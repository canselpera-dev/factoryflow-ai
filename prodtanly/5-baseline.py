import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Görsel ayarları
plt.rcParams['font.family'] = 'DejaVu Sans'
sns.set_style("whitegrid")

# ============================================================
# 1. VERİ SETİNİ YÜKLEME VE HAZIRLIK
# ============================================================
kayit_konumu = r"C:\Users\canse\OneDrive\Masaüstü\Production Forecasting"
df = pd.read_csv(f"{kayit_konumu}/kablo_uretim_veriseti_v4.csv", sep=';')
df['Tarih_DT'] = pd.to_datetime(df['Üretim Tarihi'], format='%d.%m.%Y')

print("=" * 70)
print("🎯 XGBOOST - FEATURE SELECTION & MODEL EĞİTİMİ")
print("=" * 70)

# ============================================================
# 2. GÜNLÜK AGREGASYON
# ============================================================
gunluk_df = df.groupby('Tarih_DT').agg(
    Toplam_Bobin=('Kaliteli Bobin Adedi', 'sum'),
    Toplam_Metraj=('Kaliteli Toplam Metraj', 'sum'),
    Toplam_Fire_Bobin=('Fire Bobin Adedi', 'sum'),
    Is_Emri_Sayisi=('İş Emri No', 'count'),
    Ortalama_OEE=('OEE Değeri (%)', 'mean'),
    Ortalama_Zorluk=('Üretim Zorluk Derecesi', 'mean'),
    Toplam_Bakim=('Bakım Duruşu (Saat)', 'sum'),
    Toplam_Ariza=('Arıza Duruşu (Saat)', 'sum'),
    Ortalama_Hiz=('Operasyon Hızı (m/dk)', 'mean'),
    Ortalama_Setup=('Setup Süresi (Saat)', 'mean'),
    Aktif_Hat_Sayisi=('Hat Kodu', 'nunique'),
    Aktif_Operator_Sayisi=('Operatör Sicil No', 'nunique')
).reset_index()

gunluk_df['Fire_Orani'] = (gunluk_df['Toplam_Fire_Bobin'] / 
                           (gunluk_df['Toplam_Bobin'] + gunluk_df['Toplam_Fire_Bobin']) * 100)

gunluk_df['Gun_No'] = gunluk_df['Tarih_DT'].dt.dayofweek
gunluk_df['Ay'] = gunluk_df['Tarih_DT'].dt.month
gunluk_df['Hafta'] = gunluk_df['Tarih_DT'].dt.isocalendar().week.astype(int)
gunluk_df['Ay_Gunu'] = gunluk_df['Tarih_DT'].dt.day

gunluk_df['Haftasonu'] = gunluk_df['Gun_No'].apply(lambda x: 1 if x >= 5 else 0)
gunluk_df['Pazartesi'] = gunluk_df['Gun_No'].apply(lambda x: 1 if x == 0 else 0)
gunluk_df['Cumartesi'] = gunluk_df['Gun_No'].apply(lambda x: 1 if x == 5 else 0)
gunluk_df['Bakim_Gunu'] = gunluk_df['Toplam_Bakim'].apply(lambda x: 1 if x > 0 else 0)
gunluk_df['Ariza_Gunu'] = gunluk_df['Toplam_Ariza'].apply(lambda x: 1 if x > 0 else 0)

# Lag ve rolling feature'lar
for lag in [1, 2, 3, 7, 14]:
    gunluk_df[f'Uretim_Lag_{lag}'] = gunluk_df['Toplam_Bobin'].shift(lag)

for window in [3, 7, 14]:
    gunluk_df[f'Uretim_Rolling_Mean_{window}'] = gunluk_df['Toplam_Bobin'].rolling(window).mean()
    gunluk_df[f'Uretim_Rolling_Std_{window}'] = gunluk_df['Toplam_Bobin'].rolling(window).std()
    gunluk_df[f'OEE_Rolling_Mean_{window}'] = gunluk_df['Ortalama_OEE'].rolling(window).mean()

gunluk_df['Gecen_Hafta_Ayni_Gun'] = gunluk_df['Toplam_Bobin'].shift(7)

# Temizlik
gunluk_df_temiz = gunluk_df.dropna().copy()

print(f"Model verisi: {len(gunluk_df_temiz)} gün × {len(gunluk_df_temiz.columns)} sütun")

# ============================================================
# 3. XGBOOST İÇİN VERİ HAZIRLIĞI
# ============================================================

target = 'Toplam_Bobin'

drop_cols = ['Tarih_DT', 'Toplam_Metraj', 'Toplam_Fire_Bobin']
feature_cols = [col for col in gunluk_df_temiz.columns if col not in drop_cols + [target]]

X = gunluk_df_temiz[feature_cols].copy()
y = gunluk_df_temiz[target].copy()

print(f"\nFeature sayısı: {len(feature_cols)}")

# Train/Test split (zamansal)
train_size = int(len(X) * 0.8)
X_train, X_test = X.iloc[:train_size], X.iloc[train_size:]
y_train, y_test = y.iloc[:train_size], y.iloc[train_size:]

print(f"Train: {len(X_train)} gün | Test: {len(X_test)} gün")

# ============================================================
# 4. XGBOOST MODEL EĞİTİMİ
# ============================================================
print("\n" + "=" * 70)
print("🚀 XGBoost eğitiliyor...")
print("=" * 70)

import xgboost as xgb
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error, r2_score

model = xgb.XGBRegressor(
    n_estimators=200,
    max_depth=5,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    verbosity=0
)

model.fit(X_train, y_train)

y_pred_train = model.predict(X_train)
y_pred_test = model.predict(X_test)

# ============================================================
# 5. MODEL PERFORMANSI
# ============================================================
print("\n" + "=" * 70)
print("📊 MODEL PERFORMANSI")
print("=" * 70)

def print_metrics(y_true, y_pred, set_name):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mape = mean_absolute_percentage_error(y_true, y_pred) * 100
    r2 = r2_score(y_true, y_pred)
    
    print(f"\n🔹 {set_name}")
    print(f"   MAE:  {mae:.1f} bobin")
    print(f"   RMSE: {rmse:.1f} bobin")
    print(f"   MAPE: %{mape:.1f}")
    print(f"   R²:   {r2:.4f}")
    
    return mae, rmse, mape, r2

train_mae, _, train_mape, train_r2 = print_metrics(y_train, y_pred_train, "TRAIN")
test_mae, test_rmse, test_mape, test_r2 = print_metrics(y_test, y_pred_test, "TEST")

baseline_mae = 180.6
iyilesme = (baseline_mae - test_mae) / baseline_mae * 100

print(f"\n📈 BASELINE KARŞILAŞTIRMASI:")
print(f"   Baseline MAE: {baseline_mae:.0f} bobin")
print(f"   XGBoost MAE:  {test_mae:.0f} bobin")
print(f"   İyileşme:     %{iyilesme:.1f}")

# ============================================================
# 6. FEATURE IMPORTANCE
# ============================================================
print("\n" + "=" * 70)
print("⭐ FEATURE IMPORTANCE (İLK 20)")
print("=" * 70)

importance_df = pd.DataFrame({
    'Feature': feature_cols,
    'Importance': model.feature_importances_
}).sort_values('Importance', ascending=False)

print(importance_df.head(20).to_string(index=False))

# ============================================================
# 7. GÖRSELLEŞTİRME
# ============================================================
print("\n📊 Görseller oluşturuluyor...")

fig, axes = plt.subplots(2, 2, figsize=(18, 12))
fig.suptitle('XGBOOST - KABLO ÜRETİM TAHMİNİ', fontsize=18, fontweight='bold')

# 7.1 Feature Importance
top_features = importance_df.head(15)
# DÜZELTME: colormap string olarak
colors = plt.cm.get_cmap('RdYlGn')(np.linspace(0.2, 0.8, len(top_features)))[::-1]
axes[0, 0].barh(range(len(top_features)), top_features['Importance'].values, color=colors, edgecolor='black')
axes[0, 0].set_yticks(range(len(top_features)))
axes[0, 0].set_yticklabels(top_features['Feature'].values, fontsize=9)
axes[0, 0].set_title('Top 15 Feature Importance', fontsize=13, fontweight='bold')
axes[0, 0].set_xlabel('Önem Skoru')
axes[0, 0].invert_yaxis()

# 7.2 Gerçek vs Tahmin (Test)
axes[0, 1].scatter(y_test, y_pred_test, alpha=0.6, c='steelblue', s=30, edgecolor='white')
axes[0, 1].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', linewidth=2)
axes[0, 1].set_title(f'Gerçek vs Tahmin (R² = {test_r2:.3f})', fontsize=13, fontweight='bold')
axes[0, 1].set_xlabel('Gerçek Bobin')
axes[0, 1].set_ylabel('Tahmin Bobin')

# 7.3 Zaman Serisi - Train + Test
tarihler = gunluk_df_temiz['Tarih_DT'].values
axes[1, 0].plot(tarihler[:train_size], y_train.values, 'steelblue', alpha=0.6, linewidth=0.8, label='Train Gerçek')
axes[1, 0].plot(tarihler[train_size:], y_test.values, 'red', alpha=0.8, linewidth=1.2, label='Test Gerçek')
axes[1, 0].plot(tarihler[train_size:], y_pred_test, 'green', alpha=0.8, linewidth=1.5, label='Test Tahmin')
axes[1, 0].axvline(tarihler[train_size], color='gray', linestyle='--', alpha=0.7)
axes[1, 0].set_title('XGBoost Tahminleri', fontsize=13, fontweight='bold')
axes[1, 0].set_xlabel('Tarih')
axes[1, 0].set_ylabel('Bobin')
axes[1, 0].legend()
axes[1, 0].tick_params(axis='x', rotation=45)

# 7.4 Hata Dağılımı
hatalar = y_test - y_pred_test
axes[1, 1].hist(hatalar, bins=25, color='steelblue', edgecolor='black', alpha=0.7)
axes[1, 1].axvline(0, color='red', linestyle='--', linewidth=2)
axes[1, 1].axvline(hatalar.mean(), color='orange', linestyle='--', linewidth=2, label=f'Ortalama Hata: {hatalar.mean():.0f}')
axes[1, 1].set_title('Tahmin Hata Dağılımı', fontsize=13, fontweight='bold')
axes[1, 1].set_xlabel('Hata (Gerçek - Tahmin)')
axes[1, 1].set_ylabel('Frekans')
axes[1, 1].legend()

plt.tight_layout()
plt.savefig(f"{kayit_konumu}/XGBoost_Results.png", dpi=150, bbox_inches='tight')
plt.show()

# ============================================================
# 8. SEÇİLMİŞ FEATURE'LAR İLE MODEL
# ============================================================
print("\n" + "=" * 70)
print("🎯 SEÇİLMİŞ FEATURE'LAR İLE MODEL")
print("=" * 70)

# Eşik değerin üzerindeki feature'ları seç
esik = 0.01
selected_features = importance_df[importance_df['Importance'] > esik]['Feature'].tolist()
print(f"Eşik > {esik}: {len(selected_features)} feature seçildi")
print(f"Seçilenler: {selected_features}")

X_train_selected = X_train[selected_features]
X_test_selected = X_test[selected_features]

model_selected = xgb.XGBRegressor(
    n_estimators=200,
    max_depth=5,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    verbosity=0
)

model_selected.fit(X_train_selected, y_train)
y_pred_selected = model_selected.predict(X_test_selected)

selected_mae = mean_absolute_error(y_test, y_pred_selected)
selected_r2 = r2_score(y_test, y_pred_selected)
selected_mape = mean_absolute_percentage_error(y_test, y_pred_selected) * 100

print(f"\n🔹 SEÇİLMİŞ FEATURE MODELİ ({len(selected_features)} feature)")
print(f"   MAE:  {selected_mae:.1f} bobin")
print(f"   MAPE: %{selected_mape:.1f}")
print(f"   R²:   {selected_r2:.4f}")

# Final karşılaştırma
print("\n" + "=" * 70)
print("🏆 FİNAL KARŞILAŞTIRMA")
print("=" * 70)
print(f"{'Model':<30} {'MAE':>8} {'MAPE':>8} {'R²':>8}")
print("-" * 55)
print(f"{'Baseline (7-Gün MA)':<30} {180.6:>8.0f} {21.3:>7.1f}% {0.22:>7.3f}")
print(f"{'XGBoost (Tüm Feature)':<30} {test_mae:>8.0f} {test_mape:>7.1f}% {test_r2:>7.3f}")
print(f"{'XGBoost (Seçili Feature)':<30} {selected_mae:>8.0f} {selected_mape:>7.1f}% {selected_r2:>7.3f}")

print(f"\n📁 Görseller: {kayit_konumu}\\XGBoost_Results.png")
print("=" * 70)
print("✅ XGBOOST MODEL EĞİTİMİ TAMAMLANDI!") 