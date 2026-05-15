import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
import joblib
import os
warnings.filterwarnings('ignore')

plt.rcParams['font.family'] = 'DejaVu Sans'

# ============================================================
# 1. VERİ YÜKLE VE HAZIRLA
# ============================================================
kayit_konumu = r"C:\Users\canse\OneDrive\Masaüstü\Production Forecasting"

df = pd.read_csv(f"{kayit_konumu}/kablo_uretim_veriseti_v4.csv", sep=';')
df['Tarih_DT'] = pd.to_datetime(df['Üretim Tarihi'], format='%d.%m.%Y')

gunluk = df.groupby('Tarih_DT')['Kaliteli Bobin Adedi'].sum().reset_index()
gunluk.columns = ['Tarih_DT', 'Toplam_Bobin']
gunluk = gunluk.sort_values('Tarih_DT').reset_index(drop=True)

print("=" * 70)
print("🎯 XGBoost (Tuned) - FİNAL EĞİTİM")
print("=" * 70)
print(f"Veri: {len(gunluk)} gün | {gunluk['Tarih_DT'].min().date()} → {gunluk['Tarih_DT'].max().date()}")
print(f"Günlük ortalama: {gunluk['Toplam_Bobin'].mean():.0f} bobin")

# ============================================================
# 2. FEATURE ENGINEERING
# ============================================================
def create_features(data):
    """Tahmin anında bilinebilecek feature'ları üret"""
    df = data.copy()
    
    # Takvim
    df['Gun_No'] = df['Tarih_DT'].dt.dayofweek
    df['Ay'] = df['Tarih_DT'].dt.month
    df['Hafta'] = df['Tarih_DT'].dt.isocalendar().week.astype(int)
    df['Ay_Gunu'] = df['Tarih_DT'].dt.day
    df['Pazartesi'] = (df['Gun_No'] == 0).astype(int)
    df['Cumartesi'] = (df['Gun_No'] == 5).astype(int)
    
    # Lag feature'lar (geçmiş üretim)
    for lag in [1, 2, 3, 7]:
        df[f'Lag_{lag}'] = df['Toplam_Bobin'].shift(lag)
    
    # Rolling mean (hareketli ortalama)
    for w in [3, 7, 14]:
        df[f'RM_{w}'] = df['Toplam_Bobin'].shift(1).rolling(w).mean()
    
    df['Gecen_Hafta'] = df['Toplam_Bobin'].shift(7)
    df['Haftalik_Ort'] = df['Toplam_Bobin'].shift(1).rolling(7).mean()
    
    return df

# Feature'ları üret
full_data = create_features(gunluk)
full_data = full_data.dropna()
print(f"Feature sonrası: {len(full_data)} gün")

# ============================================================
# 3. TRAIN/TEST SPLIT
# ============================================================
feature_cols = ['Gun_No', 'Ay', 'Hafta', 'Ay_Gunu', 'Pazartesi', 'Cumartesi',
                'Lag_1', 'Lag_2', 'Lag_3', 'Lag_7', 
                'RM_3', 'RM_7', 'RM_14',
                'Gecen_Hafta', 'Haftalik_Ort']

X = full_data[feature_cols].copy()
y = full_data['Toplam_Bobin'].copy()

split = int(len(X) * 0.8)
X_train, X_test = X.iloc[:split], X.iloc[split:]
y_train, y_test = y.iloc[:split], y.iloc[split:]

print(f"\nTrain: {len(X_train)} gün | Test: {len(X_test)} gün")
print(f"Train: {full_data['Tarih_DT'].iloc[:split].min().date()} → {full_data['Tarih_DT'].iloc[:split].max().date()}")
print(f"Test:  {full_data['Tarih_DT'].iloc[split:].min().date()} → {full_data['Tarih_DT'].iloc[split:].max().date()}")

# ============================================================
# 4. XGBOOST EĞİTİMİ (En iyi parametrelerle)
# ============================================================
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error, r2_score

print("\n" + "=" * 70)
print("🚀 XGBoost Eğitiliyor...")
print("=" * 70)

# Grid search'ten gelen en iyi parametreler
model = xgb.XGBRegressor(
    n_estimators=50,
    max_depth=2,
    learning_rate=0.01,
    reg_alpha=1,
    reg_lambda=1,
    subsample=0.7,
    colsample_bytree=0.7,
    random_state=42,
    verbosity=0
)

model.fit(X_train, y_train)

# ============================================================
# 5. TAHMİN VE SKORLAR
# ============================================================
y_pred_train = model.predict(X_train)
y_pred_test = model.predict(X_test)

print("\n" + "=" * 70)
print("📊 MODEL PERFORMANSI")
print("=" * 70)

def rapor(y_true, y_pred, set_name):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mape = mean_absolute_percentage_error(y_true, y_pred) * 100
    r2 = r2_score(y_true, y_pred)
    
    print(f"\n🔹 {set_name}")
    print(f"   MAE:  {mae:.1f} bobin")
    print(f"   RMSE: {rmse:.1f} bobin")
    print(f"   MAPE: %{mape:.1f}")
    print(f"   R²:   {r2:.4f}")
    
    return mae, r2

train_mae, train_r2 = rapor(y_train, y_pred_train, "TRAIN")
test_mae, test_r2 = rapor(y_test, y_pred_test, "TEST")

# Overfitting kontrolü
print(f"\n🔍 OVERFITTING:")
print(f"   Train MAE: {train_mae:.0f} | Test MAE: {test_mae:.0f} | Fark: {test_mae-train_mae:.0f}")
print(f"   Train R²: {train_r2:.3f} | Test R²: {test_r2:.3f} | Fark: {train_r2-test_r2:.3f}")

if test_r2 > 0 and train_r2 - test_r2 < 0.5:
    print("   ✅ Model dengeli — overfitting yok")
else:
    print("   ⚠️  İyileştirme yapılabilir")

# ============================================================
# 6. FEATURE IMPORTANCE
# ============================================================
print("\n" + "=" * 70)
print("⭐ FEATURE IMPORTANCE")
print("=" * 70)

imp_df = pd.DataFrame({
    'Feature': feature_cols,
    'Importance': model.feature_importances_
}).sort_values('Importance', ascending=False)

print(imp_df.to_string(index=False))

# ============================================================
# 7. GÖRSELLEŞTİRME
# ============================================================
fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle('XGBoost (Tuned) - Final Model', fontsize=16, fontweight='bold')

# Feature importance
top10 = imp_df.head(10)
colors = plt.cm.get_cmap('viridis')(np.linspace(0.2, 0.9, 10))
axes[0, 0].barh(range(10), top10['Importance'].values, color=colors, edgecolor='black')
axes[0, 0].set_yticks(range(10))
axes[0, 0].set_yticklabels(top10['Feature'].values, fontsize=9)
axes[0, 0].set_title('Top 10 Feature', fontsize=12, fontweight='bold')
axes[0, 0].invert_yaxis()

# Gerçek vs Tahmin (Test)
axes[0, 1].scatter(y_test, y_pred_test, alpha=0.6, c='steelblue', s=30, edgecolor='white')
axes[0, 1].plot([y.min(), y.max()], [y.min(), y.max()], 'r--', linewidth=2)
axes[0, 1].set_title(f'Gerçek vs Tahmin (R² = {test_r2:.3f})', fontsize=12, fontweight='bold')
axes[0, 1].set_xlabel('Gerçek Bobin')
axes[0, 1].set_ylabel('Tahmin Bobin')

# Zaman serisi
test_tarih = full_data['Tarih_DT'].iloc[split:].values
axes[1, 0].plot(test_tarih, y_test.values, 'black', linewidth=2, label='Gerçek', alpha=0.8)
axes[1, 0].plot(test_tarih, y_pred_test, 'green', linewidth=1.5, label='Tahmin', alpha=0.8)
axes[1, 0].fill_between(test_tarih, y_pred_test - test_mae, y_pred_test + test_mae, 
                         alpha=0.1, color='green', label=f'±{test_mae:.0f} bobin')
axes[1, 0].set_title('Test Dönemi Tahminleri', fontsize=12, fontweight='bold')
axes[1, 0].set_xlabel('Tarih')
axes[1, 0].set_ylabel('Bobin')
axes[1, 0].legend()
axes[1, 0].tick_params(axis='x', rotation=45)

# Hata dağılımı
hatalar = y_test - y_pred_test
axes[1, 1].hist(hatalar, bins=20, color='steelblue', edgecolor='black', alpha=0.7)
axes[1, 1].axvline(0, color='red', linestyle='--', linewidth=2)
axes[1, 1].axvline(hatalar.mean(), color='orange', linestyle='--', linewidth=2, 
                   label=f'Ort: {hatalar.mean():.0f}')
axes[1, 1].set_title(f'Hata Dağılımı (σ = {hatalar.std():.0f})', fontsize=12, fontweight='bold')
axes[1, 1].set_xlabel('Hata (Gerçek - Tahmin)')
axes[1, 1].set_ylabel('Frekans')
axes[1, 1].legend()

plt.tight_layout()
plt.savefig(f"{kayit_konumu}/Final_Model.png", dpi=150, bbox_inches='tight')
plt.show()

# ============================================================
# 8. MODELİ KAYDET
# ============================================================
model_dosyasi = os.path.join(kayit_konumu, 'xgb_model.pkl')
feature_dosyasi = os.path.join(kayit_konumu, 'feature_list.pkl')

joblib.dump(model, model_dosyasi)
joblib.dump(feature_cols, feature_dosyasi)

print(f"\n💾 Model kaydedildi:")
print(f"   📄 {model_dosyasi}")
print(f"   📄 {feature_dosyasi}")

# ============================================================
# 9. ÖZET
# ============================================================
print("\n" + "=" * 70)
print("✅ FİNAL EĞİTİM TAMAMLANDI!")
print("=" * 70)
print(f"""
📊 PERFORMANS:
   • Train MAE: {train_mae:.0f} bobin | R²: {train_r2:.3f}
   • Test MAE:  {test_mae:.0f} bobin | R²: {test_r2:.3f}
   • Günlük ~{int(y.mean()):,} bobin üretimde ±{test_mae:.0f} hata

⭐ TOP 3 FEATURE:
   1. {imp_df.iloc[0]['Feature']} (%{imp_df.iloc[0]['Importance']*100:.1f})
   2. {imp_df.iloc[1]['Feature']} (%{imp_df.iloc[1]['Importance']*100:.1f})
   3. {imp_df.iloc[2]['Feature']} (%{imp_df.iloc[2]['Importance']*100:.1f})

🔮 SIRADA: Streamlit dashboard!
""")