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
sns.set_palette("husl")

# ============================================================
# 1. VERİ SETİNİ YÜKLEME
# ============================================================
kayit_konumu = r"C:\Users\canse\OneDrive\Masaüstü\Production Forecasting"
df = pd.read_csv(f"{kayit_konumu}/kablo_uretim_veriseti_v4.csv", sep=';')

# Tarih sütununu datetime'a çevir
df['Tarih_DT'] = pd.to_datetime(df['Üretim Tarihi'], format='%d.%m.%Y')

print("=" * 70)
print("📊 FEATURE ENGINEERING & BASELINE MODEL")
print("=" * 70)
print(f"Orijinal veri: {len(df):,} iş emri")

# ============================================================
# 2. GÜNLÜK AGREGASYON
# ============================================================
print("\n📅 Günlük agregasyon yapılıyor...")

gunluk_df = df.groupby('Tarih_DT').agg(
    Toplam_Bobin=('Kaliteli Bobin Adedi', 'sum'),
    Toplam_Metraj=('Kaliteli Toplam Metraj', 'sum'),
    Toplam_Fire_Bobin=('Fire Bobin Adedi', 'sum'),
    Toplam_Fire_Metraj=('Fire Toplam Metraj', 'sum'),
    Is_Emri_Sayisi=('İş Emri No', 'count'),
    Ortalama_OEE=('OEE Değeri (%)', 'mean'),
    Ortalama_Zorluk=('Üretim Zorluk Derecesi', 'mean'),
    Toplam_Bakim=('Bakım Duruşu (Saat)', 'sum'),
    Toplam_Ariza=('Arıza Duruşu (Saat)', 'sum'),
    Toplam_Plansiz_Durus=('Plansız Duruş (Saat)', 'sum'),
    Ortalama_Hiz=('Operasyon Hızı (m/dk)', 'mean'),
    Ortalama_Setup=('Setup Süresi (Saat)', 'mean'),
    Aktif_Hat_Sayisi=('Hat Kodu', 'nunique'),
    Aktif_Operator_Sayisi=('Operatör Sicil No', 'nunique')
).reset_index()

# Fire oranı hesapla
gunluk_df['Fire_Orani'] = (gunluk_df['Toplam_Fire_Bobin'] / 
                           (gunluk_df['Toplam_Bobin'] + gunluk_df['Toplam_Fire_Bobin']) * 100)

# Haftanın günü, ay, yıl bilgileri
gunluk_df['Gun_No'] = gunluk_df['Tarih_DT'].dt.dayofweek  # 0=Pazartesi
gunluk_df['Gun_Adi'] = gunluk_df['Tarih_DT'].dt.day_name()
gunluk_df['Ay'] = gunluk_df['Tarih_DT'].dt.month
gunluk_df['Yil'] = gunluk_df['Tarih_DT'].dt.year
gunluk_df['Hafta'] = gunluk_df['Tarih_DT'].dt.isocalendar().week.astype(int)
gunluk_df['Yil_Ay'] = gunluk_df['Tarih_DT'].dt.to_period('M').astype(str)
gunluk_df['Ay_Gunu'] = gunluk_df['Tarih_DT'].dt.day

# Binary feature'lar
gunluk_df['Haftasonu'] = gunluk_df['Gun_No'].apply(lambda x: 1 if x >= 5 else 0)
gunluk_df['Pazartesi'] = gunluk_df['Gun_No'].apply(lambda x: 1 if x == 0 else 0)
gunluk_df['Cumartesi'] = gunluk_df['Gun_No'].apply(lambda x: 1 if x == 5 else 0)
gunluk_df['Bakim_Gunu'] = gunluk_df['Toplam_Bakim'].apply(lambda x: 1 if x > 0 else 0)
gunluk_df['Ariza_Gunu'] = gunluk_df['Toplam_Ariza'].apply(lambda x: 1 if x > 0 else 0)

print(f"Günlük veri: {len(gunluk_df)} gün")
print(f"Tarih aralığı: {gunluk_df['Tarih_DT'].min().date()} → {gunluk_df['Tarih_DT'].max().date()}")

# ============================================================
# 3. LAG FEATURE'LARI VE ROLLING İSTATİSTİKLER
# ============================================================
print("\n🔄 Lag ve rolling feature'lar oluşturuluyor...")

# Lag feature'lar (geçmiş günler)
for lag in [1, 2, 3, 7, 14]:
    gunluk_df[f'Uretim_Lag_{lag}'] = gunluk_df['Toplam_Bobin'].shift(lag)
    gunluk_df[f'Fire_Lag_{lag}'] = gunluk_df['Toplam_Fire_Bobin'].shift(lag)

# Rolling istatistikler
for window in [3, 7, 14]:
    gunluk_df[f'Uretim_Rolling_Mean_{window}'] = gunluk_df['Toplam_Bobin'].rolling(window).mean()
    gunluk_df[f'Uretim_Rolling_Std_{window}'] = gunluk_df['Toplam_Bobin'].rolling(window).std()
    gunluk_df[f'Fire_Rolling_Mean_{window}'] = gunluk_df['Toplam_Fire_Bobin'].rolling(window).mean()
    gunluk_df[f'OEE_Rolling_Mean_{window}'] = gunluk_df['Ortalama_OEE'].rolling(window).mean()

# Haftalık ortalama (geçen hafta)
gunluk_df['Gecen_Hafta_Ortalama'] = gunluk_df['Toplam_Bobin'].shift(7).rolling(7).mean()

# Ay başından beri kümülatif ortalama
gunluk_df['Ay_Kumulatif_Ortalama'] = gunluk_df.groupby('Yil_Ay')['Toplam_Bobin'].transform(
    lambda x: x.expanding().mean()
)

# Bir önceki aynı gün (haftalık pattern)
gunluk_df['Gecen_Hafta_Ayni_Gun'] = gunluk_df['Toplam_Bobin'].shift(7)

print(f"Toplam feature sayısı: {len(gunluk_df.columns)}")

# ============================================================
# 4. EKSİK VERİ TEMİZLİĞİ
# ============================================================
print("\n🧹 Eksik veri temizliği...")

# İlk 14 gün lag feature'ları için NaN olacak, bunları düşürelim
gunluk_df_temiz = gunluk_df.dropna().copy()
print(f"Temizlenmiş veri: {len(gunluk_df_temiz)} gün (orijinal: {len(gunluk_df)})")

# ============================================================
# 5. BASELINE MODEL - NAIVE FORECAST
# ============================================================
print("\n" + "=" * 70)
print("📈 5. BASELINE MODELLER")
print("=" * 70)

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error, r2_score

# Veriyi train/test olarak ayır (son %20 test)
train_size = int(len(gunluk_df_temiz) * 0.8)
train = gunluk_df_temiz.iloc[:train_size]
test = gunluk_df_temiz.iloc[train_size:]

print(f"\n📊 Train: {len(train)} gün | Test: {len(test)} gün")
print(f"Train: {train['Tarih_DT'].min().date()} → {train['Tarih_DT'].max().date()}")
print(f"Test:  {test['Tarih_DT'].min().date()} → {test['Tarih_DT'].max().date()}")

# Hedef değişken
y_train = train['Toplam_Bobin'].values
y_test = test['Toplam_Bobin'].values

# ---------- MODEL 1: Naive (Geçen hafta aynı gün) ----------
y_pred_naive = test['Gecen_Hafta_Ayni_Gun'].values

# ---------- MODEL 2: 7 Günlük Hareketli Ortalama ----------
y_pred_ma7 = test['Uretim_Rolling_Mean_7'].values

# ---------- MODEL 3: Geçen Hafta Ortalama ----------
y_pred_weekly = test['Gecen_Hafta_Ortalama'].values

def evaluate_model(y_true, y_pred, model_name):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mape = mean_absolute_percentage_error(y_true, y_pred) * 100
    r2 = r2_score(y_true, y_pred)
    
    print(f"\n🔹 {model_name}")
    print(f"   MAE:  {mae:.1f} bobin")
    print(f"   RMSE: {rmse:.1f} bobin")
    print(f"   MAPE: %{mape:.1f}")
    print(f"   R²:   {r2:.4f}")
    
    return {'Model': model_name, 'MAE': mae, 'RMSE': rmse, 'MAPE': mape, 'R2': r2}

sonuclar = []
sonuclar.append(evaluate_model(y_test, y_pred_naive, "Naive (Geçen Hafta Aynı Gün)"))
sonuclar.append(evaluate_model(y_test, y_pred_ma7, "7 Günlük Hareketli Ortalama"))
sonuclar.append(evaluate_model(y_test, y_pred_weekly, "Geçen Hafta Ortalaması"))

# ============================================================
# 6. GÖRSELLEŞTİRME
# ============================================================
print("\n" + "=" * 70)
print("📊 6. GÖRSELLEŞTİRMELER")
print("=" * 70)

fig, axes = plt.subplots(3, 2, figsize=(18, 14))
fig.suptitle('KABLO ÜRETİM TAHMİNİ - FEATURE ENGINEERING & BASELINE', fontsize=18, fontweight='bold')

# 6.1 Günlük Üretim - Train/Test ayrımı
axes[0, 0].plot(train['Tarih_DT'], train['Toplam_Bobin'], 'steelblue', alpha=0.7, linewidth=0.8, label='Train')
axes[0, 0].plot(test['Tarih_DT'], test['Toplam_Bobin'], 'red', alpha=0.7, linewidth=0.8, label='Test')
axes[0, 0].axvline(test['Tarih_DT'].min(), color='gray', linestyle='--', alpha=0.7)
axes[0, 0].set_title('Günlük Bobin Üretimi - Train/Test Split', fontsize=12, fontweight='bold')
axes[0, 0].set_xlabel('Tarih')
axes[0, 0].set_ylabel('Bobin Adedi')
axes[0, 0].legend()
axes[0, 0].tick_params(axis='x', rotation=45)

# 6.2 Baseline Karşılaştırma (Test dönemi)
axes[0, 1].plot(test['Tarih_DT'], y_test, 'black', linewidth=2, label='Gerçek', alpha=0.8)
axes[0, 1].plot(test['Tarih_DT'], y_pred_naive, 'steelblue', linewidth=1, alpha=0.7, label='Naive')
axes[0, 1].plot(test['Tarih_DT'], y_pred_ma7, 'orange', linewidth=1, alpha=0.7, label='MA-7')
axes[0, 1].set_title('Baseline Modeller - Test Dönemi', fontsize=12, fontweight='bold')
axes[0, 1].set_xlabel('Tarih')
axes[0, 1].set_ylabel('Bobin Adedi')
axes[0, 1].legend()
axes[0, 1].tick_params(axis='x', rotation=45)

# 6.3 Haftanın Günlerine Göre Üretim
gun_sirasi = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday']
gun_ort = gunluk_df_temiz.groupby('Gun_Adi')['Toplam_Bobin'].mean().reindex(gun_sirasi)
renkler = ['#e74c3c' if g == 'Monday' else '#f39c12' if g == 'Saturday' else '#3498db' for g in gun_sirasi]
axes[1, 0].bar(range(len(gun_sirasi)), gun_ort.values, color=renkler, edgecolor='black')
axes[1, 0].set_title('Gün Bazlı Ortalama Üretim', fontsize=12, fontweight='bold')
axes[1, 0].set_xticks(range(len(gun_sirasi)))
axes[1, 0].set_xticklabels(['Pzt', 'Sal', 'Çar', 'Per', 'Cum', 'Cmt'])
axes[1, 0].set_ylabel('Ortalama Bobin')
for i, v in enumerate(gun_ort.values):
    if not np.isnan(v):
        axes[1, 0].text(i, v + 5, f'{v:.0f}', ha='center', fontsize=9)

# 6.4 Fire Oranı vs Zorluk
zorluk_ort = gunluk_df_temiz.groupby('Ortalama_Zorluk')['Fire_Orani'].mean()
axes[1, 1].scatter(gunluk_df_temiz['Ortalama_Zorluk'], gunluk_df_temiz['Fire_Orani'], 
                   alpha=0.3, s=10, c='steelblue')
z = np.polyfit(gunluk_df_temiz['Ortalama_Zorluk'].dropna(), gunluk_df_temiz['Fire_Orani'].dropna(), 1)
p = np.poly1d(z)
x_trend = np.linspace(1, 7, 50)
axes[1, 1].plot(x_trend, p(x_trend), 'r--', linewidth=2)
axes[1, 1].set_title('Ortalama Zorluk vs Fire Oranı', fontsize=12, fontweight='bold')
axes[1, 1].set_xlabel('Ortalama Zorluk Derecesi')
axes[1, 1].set_ylabel('Fire Oranı (%)')

# 6.5 Model Hata Karşılaştırması
model_isimleri = [s['Model'][:20] for s in sonuclar]
mae_degerleri = [s['MAE'] for s in sonuclar]
axes[2, 0].barh(model_isimleri, mae_degerleri, color=['#3498db', '#f39c12', '#2ecc71'], edgecolor='black')
axes[2, 0].set_title('Model MAE Karşılaştırması', fontsize=12, fontweight='bold')
axes[2, 0].set_xlabel('MAE (Bobin)')
for i, v in enumerate(mae_degerleri):
    axes[2, 0].text(v + 1, i, f'{v:.0f}', va='center', fontsize=10)

# 6.6 Feature Importance (Korelasyon)
korelasyon_features = ['Toplam_Bobin', 'Is_Emri_Sayisi', 'Ortalama_OEE', 'Ortalama_Zorluk',
                       'Toplam_Bakim', 'Toplam_Ariza', 'Aktif_Hat_Sayisi',
                       'Uretim_Lag_1', 'Uretim_Rolling_Mean_7', 'Gecen_Hafta_Ayni_Gun']
korelasyon = gunluk_df_temiz[korelasyon_features].corr()['Toplam_Bobin'].drop('Toplam_Bobin').sort_values()
renkler_kor = ['#2ecc71' if x > 0 else '#e74c3c' for x in korelasyon.values]
axes[2, 1].barh(range(len(korelasyon)), korelasyon.values, color=renkler_kor, edgecolor='black')
axes[2, 1].set_yticks(range(len(korelasyon)))
axes[2, 1].set_yticklabels(korelasyon.index, fontsize=9)
axes[2, 1].set_title('Feature-Target Korelasyonu', fontsize=12, fontweight='bold')
axes[2, 1].set_xlabel('Korelasyon')
axes[2, 1].axvline(0, color='black', linewidth=0.5)
for i, v in enumerate(korelasyon.values):
    axes[2, 1].text(v + 0.01, i, f'{v:.3f}', va='center', fontsize=8)

plt.tight_layout()
plt.savefig(f"{kayit_konumu}/Feature_Engineering_Baseline.png", dpi=150, bbox_inches='tight')
plt.show()

# ============================================================
# 7. ÖZET VE SONRAKİ ADIMLAR
# ============================================================
print("\n" + "=" * 70)
print("📋 7. ÖZET")
print("=" * 70)

best_model = min(sonuclar, key=lambda x: x['MAE'])
print(f"""
✅ VERİ SETİ:
   • Orijinal: {len(df):,} iş emri
   • Günlük: {len(gunluk_df)} gün
   • Model için: {len(gunluk_df_temiz)} gün (NaN'ler çıkarıldı)
   • Feature sayısı: {len(gunluk_df_temiz.columns)}

✅ EN İYİ BASELINE: {best_model['Model']}
   • MAE: {best_model['MAE']:.0f} bobin
   • MAPE: %{best_model['MAPE']:.1f}
   
✅ SONRAKİ ADIMLAR:
   1. XGBoost ile ML modeli
   2. LSTM ile DL modeli
   3. Hiperparametre optimizasyonu
   4. Streamlit dashboard
""")

print("=" * 70)
print("✅ FEATURE ENGINEERING TAMAMLANDI!")
print(f"📁 Görseller: {kayit_konumu}\\Feature_Engineering_Baseline.png")
print("=" * 70)