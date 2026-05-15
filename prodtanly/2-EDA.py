import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Türkçe karakter desteği için
plt.rcParams['font.family'] = 'DejaVu Sans'
sns.set_style("whitegrid")
sns.set_palette("husl")

# ============================================================
# 1. VERİ SETİNİ YÜKLEME
# ============================================================
kayit_konumu = r"C:\Users\canse\OneDrive\Masaüstü\Production Forecasting"
df = pd.read_csv(f"{kayit_konumu}/kablo_uretim_veriseti_v3.csv", sep=';')

# Tarih sütununu datetime'a çevir
df['Tarih_DT'] = pd.to_datetime(df['Üretim Tarihi'], format='%d.%m.%Y')

print("=" * 70)
print("📊 KABLO ÜRETİM VERİ SETİ - KEŞİFÇİ VERİ ANALİZİ (EDA)")
print("=" * 70)

# ============================================================
# 2. TEMEL VERİ SETİ BİLGİLERİ
# ============================================================
print("\n" + "=" * 70)
print("📋 2. TEMEL VERİ SETİ BİLGİLERİ")
print("=" * 70)

print(f"\n🔢 Boyut: {df.shape[0]:,} satır × {df.shape[1]} sütun")
print(f"📅 Tarih Aralığı: {df['Üretim Tarihi'].min()} - {df['Üretim Tarihi'].max()}")
print(f"📆 Toplam Çalışma Günü: {df['Üretim Tarihi'].nunique()}")
print(f"💾 Bellek Kullanımı: {df.memory_usage(deep=True).sum() / 1024 / 1024:.2f} MB")

print("\n📊 SÜTUN TİPLERİ:")
print(df.dtypes.value_counts())

print("\n🔍 EKSİK VERİ KONTROLÜ:")
eksik_veri = df.isnull().sum()
eksik_veri = eksik_veri[eksik_veri > 0]
if len(eksik_veri) > 0:
    print(eksik_veri)
else:
    print("✅ Eksik veri yok!")

# ============================================================
# 3. TEMEL İSTATİSTİKLER
# ============================================================
print("\n" + "=" * 70)
print("📈 3. SAYISAL SÜTUNLAR İÇİN TEMEL İSTATİSTİKLER")
print("=" * 70)

onemli_sutunlar = [
    'Kaliteli Bobin Adedi', 'Kaliteli Toplam Metraj', 
    'Fire Bobin Adedi', 'Fire Toplam Metraj',
    'Setup Süresi (Saat)', 'Operasyon Süresi (Saat)', 
    'Operasyon Hızı (m/dk)', 'Bakım Duruşu (Saat)',
    'Arıza Duruşu (Saat)', 'Üretim Zorluk Derecesi',
    'Kullanım Oranı (%)', 'Performans Oranı (%)', 
    'Kalite Oranı (%)', 'OEE Değeri (%)'
]

print(df[onemli_sutunlar].describe().round(2))

# ============================================================
# 4. KATEGORİK DEĞİŞKEN ANALİZİ
# ============================================================
print("\n" + "=" * 70)
print("🏷️ 4. KATEGORİK DEĞİŞKEN ANALİZİ")
print("=" * 70)

print(f"\n🏭 TESİSLER:")
for tesis in df['Tesis Kodu'].unique():
    tesis_df = df[df['Tesis Kodu'] == tesis]
    print(f"  {tesis}: {len(tesis_df):,} iş emri (%{len(tesis_df)/len(df)*100:.1f})")

print(f"\n🔧 ÜRETİM HATLARI:")
hat_ozet = df.groupby('Hat Kodu').agg(
    is_emri_sayisi=('İş Emri No', 'count'),
    ortalama_uretim=('Kaliteli Bobin Adedi', 'mean'),
    ortalama_oee=('OEE Değeri (%)', 'mean'),
    toplam_ariza=('Arıza Duruşu (Saat)', 'sum')
).round(2)
print(hat_ozet)

print(f"\n👨‍🔧 OPERATÖRLER (İlk 10):")
op_ozet = df.groupby('Operatör Adı Soyadı').agg(
    is_emri_sayisi=('İş Emri No', 'count'),
    ortalama_uretim=('Kaliteli Bobin Adedi', 'mean'),
    ortalama_oee=('OEE Değeri (%)', 'mean'),
    ortalama_fire=('Fire Bobin Adedi', 'mean')
).round(2).sort_values('is_emri_sayisi', ascending=False)
print(op_ozet)

print(f"\n📦 ÜRÜN GRUPLARI (Zorluk seviyesine göre):")
urun_ozet = df.groupby(['Ürün Kodu', 'Üretim Zorluk Derecesi']).agg(
    is_emri_sayisi=('İş Emri No', 'count'),
    ortalama_uretim=('Kaliteli Bobin Adedi', 'mean'),
    ortalama_setup=('Setup Süresi (Saat)', 'mean'),
    ortalama_oee=('OEE Değeri (%)', 'mean')
).round(2).sort_values('Üretim Zorluk Derecesi')
print(urun_ozet)

# ============================================================
# 5. GÖRSELLEŞTİRMELER
# ============================================================
print("\n" + "=" * 70)
print("📊 5. GÖRSELLEŞTİRMELER OLUŞTURULUYOR...")
print("=" * 70)

fig, axes = plt.subplots(3, 3, figsize=(20, 16))
fig.suptitle('KABLO ÜRETİM TESİSİ - KEŞİFÇİ VERİ ANALİZİ', fontsize=18, fontweight='bold', y=0.98)

# 5.1 Günlük Toplam Üretim Trendi
gunluk_uretim = df.groupby('Tarih_DT')['Kaliteli Bobin Adedi'].sum().reset_index()
axes[0, 0].plot(gunluk_uretim['Tarih_DT'], gunluk_uretim['Kaliteli Bobin Adedi'], 
                linewidth=0.8, color='steelblue', alpha=0.8)
axes[0, 0].fill_between(gunluk_uretim['Tarih_DT'], gunluk_uretim['Kaliteli Bobin Adedi'], 
                         alpha=0.3, color='steelblue')
# 7 günlük hareketli ortalama
gunluk_uretim['MA7'] = gunluk_uretim['Kaliteli Bobin Adedi'].rolling(7).mean()
axes[0, 0].plot(gunluk_uretim['Tarih_DT'], gunluk_uretim['MA7'], 
                linewidth=2, color='red', label='7 Günlük Ortalama')
axes[0, 0].set_title('Günlük Toplam Bobin Üretimi', fontsize=13, fontweight='bold')
axes[0, 0].set_xlabel('Tarih')
axes[0, 0].set_ylabel('Bobin Adedi')
axes[0, 0].legend()
axes[0, 0].tick_params(axis='x', rotation=45)

# 5.2 Haftanın Günlerine Göre Üretim
gun_sirasi = ['PAZARTESİ', 'SALI', 'ÇARŞAMBA', 'PERŞEMBE', 'CUMA', 'CUMARTESİ']
gunluk_ortalama = df.groupby('Gün Adı')['Kaliteli Bobin Adedi'].mean().reindex(gun_sirasi)
renkler = ['#e74c3c' if gun == 'PAZARTESİ' else '#f39c12' if gun == 'CUMARTESİ' else '#3498db' 
           for gun in gun_sirasi]
axes[0, 1].bar(gun_sirasi, gunluk_ortalama.values, color=renkler, edgecolor='black', linewidth=0.5)
axes[0, 1].set_title('Gün Bazlı Ortalama Bobin Üretimi', fontsize=13, fontweight='bold')
axes[0, 1].set_xlabel('Gün')
axes[0, 1].set_ylabel('Ortalama Bobin Adedi')
axes[0, 1].tick_params(axis='x', rotation=45)
# Değerleri çubukların üzerine yaz
for i, (gun, deger) in enumerate(zip(gun_sirasi, gunluk_ortalama.values)):
    if not pd.isna(deger):
        axes[0, 1].text(i, deger + 5, f'{deger:.0f}', ha='center', fontsize=9)

# 5.3 Tesis Bazlı Üretim Dağılımı
tesis_uretim = df.groupby('Tesis Adı')['Kaliteli Bobin Adedi'].sum()
axes[0, 2].pie(tesis_uretim.values, labels=tesis_uretim.index, autopct='%1.1f%%',
               colors=['#3498db', '#e74c3c'], startangle=90, explode=(0.02, 0.02))
axes[0, 2].set_title('Tesis Bazlı Toplam Üretim Dağılımı', fontsize=13, fontweight='bold')

# 5.4 Zorluk Derecesi vs OEE
axes[1, 0].scatter(df['Üretim Zorluk Derecesi'], df['OEE Değeri (%)'], 
                   alpha=0.3, c='steelblue', s=20)
# Trend çizgisi
z = np.polyfit(df['Üretim Zorluk Derecesi'], df['OEE Değeri (%)'], 1)
p = np.poly1d(z)
x_trend = np.linspace(0, 10, 100)
axes[1, 0].plot(x_trend, p(x_trend), "r--", linewidth=2, label='Trend')
axes[1, 0].set_title('Üretim Zorluk Derecesi vs OEE', fontsize=13, fontweight='bold')
axes[1, 0].set_xlabel('Zorluk Derecesi (1-9)')
axes[1, 0].set_ylabel('OEE (%)')
axes[1, 0].legend()
axes[1, 0].grid(True, alpha=0.3)

# 5.5 Bakım ve Arıza Duruşları - Aylık Trend
df['Ay_Yil'] = df['Tarih_DT'].dt.to_period('M')
aylik_durus = df.groupby('Ay_Yil').agg(
    bakim=('Bakım Duruşu (Saat)', 'sum'),
    ariza=('Arıza Duruşu (Saat)', 'sum')
)
aylik_durus.index = aylik_durus.index.astype(str)
ax = axes[1, 1]
x = np.arange(len(aylik_durus))
width = 0.35
ax.bar(x - width/2, aylik_durus['bakim'], width, label='Planlı Bakım', color='#3498db', alpha=0.8)
ax.bar(x + width/2, aylik_durus['ariza'], width, label='Arıza', color='#e74c3c', alpha=0.8)
ax.set_title('Aylık Bakım ve Arıza Duruşları', fontsize=13, fontweight='bold')
ax.set_xlabel('Ay')
ax.set_ylabel('Toplam Duruş (Saat)')
ax.set_xticks(x)
ax.set_xticklabels(aylik_durus.index, rotation=45, ha='right')
ax.legend()
ax.grid(True, alpha=0.3, axis='y')

# 5.6 Operasyon Hızı Dağılımı
axes[1, 2].hist(df['Operasyon Hızı (m/dk)'], bins=40, color='steelblue', 
                edgecolor='black', alpha=0.7, linewidth=0.5)
axes[1, 2].axvline(df['Operasyon Hızı (m/dk)'].mean(), color='red', linestyle='--', 
                   linewidth=2, label=f"Ortalama: {df['Operasyon Hızı (m/dk)'].mean():.0f} m/dk")
axes[1, 2].set_title('Operasyon Hızı Dağılımı', fontsize=13, fontweight='bold')
axes[1, 2].set_xlabel('Operasyon Hızı (m/dk)')
axes[1, 2].set_ylabel('Frekans')
axes[1, 2].legend()

# 5.7 Fire Oranı - Zorluk Derecesine Göre
zorluk_fire = df.groupby('Üretim Zorluk Derecesi').agg(
    toplam_uretim=('Kaliteli Bobin Adedi', 'sum'),
    toplam_fire=('Fire Bobin Adedi', 'sum')
)
zorluk_fire['Fire_Orani'] = (zorluk_fire['toplam_fire'] / 
                             (zorluk_fire['toplam_uretim'] + zorluk_fire['toplam_fire']) * 100)
axes[2, 0].bar(zorluk_fire.index, zorluk_fire['Fire_Orani'], 
              color=['#2ecc71' if x <= 3 else '#f39c12' if x <= 6 else '#e74c3c' 
                     for x in zorluk_fire.index],
              edgecolor='black', linewidth=0.5)
axes[2, 0].set_title('Zorluk Derecesine Göre Fire Oranı', fontsize=13, fontweight='bold')
axes[2, 0].set_xlabel('Üretim Zorluk Derecesi')
axes[2, 0].set_ylabel('Fire Oranı (%)')
axes[2, 0].set_xticks(zorluk_fire.index)
# Değerleri yaz
for i, (index, row) in enumerate(zorluk_fire.iterrows()):
    axes[2, 0].text(index, row['Fire_Orani'] + 0.1, f"%{row['Fire_Orani']:.1f}", 
                    ha='center', fontsize=9)

# 5.8 OEE Bileşenleri - Boxplot
oee_bilesenler = df[['Kullanım Oranı (%)', 'Performans Oranı (%)', 'Kalite Oranı (%)']]
axes[2, 1].boxplot([oee_bilesenler['Kullanım Oranı (%)'].dropna(),
                    oee_bilesenler['Performans Oranı (%)'].dropna(),
                    oee_bilesenler['Kalite Oranı (%)'].dropna()],
                   labels=['Kullanım', 'Performans', 'Kalite'],
                   patch_artist=True,
                   boxprops=dict(facecolor='steelblue', alpha=0.6),
                   medianprops=dict(color='red', linewidth=2))
axes[2, 1].set_title('OEE Bileşenleri Dağılımı', fontsize=13, fontweight='bold')
axes[2, 1].set_ylabel('Oran (%)')
axes[2, 1].grid(True, alpha=0.3, axis='y')

# 5.9 Korelasyon Isı Haritası
korelasyon_sutunlari = [
    'Kaliteli Bobin Adedi', 'Fire Bobin Adedi', 'Setup Süresi (Saat)',
    'Bakım Duruşu (Saat)', 'Arıza Duruşu (Saat)', 'Operasyon Süresi (Saat)',
    'Operasyon Hızı (m/dk)', 'Üretim Zorluk Derecesi',
    'Kullanım Oranı (%)', 'Performans Oranı (%)', 'Kalite Oranı (%)', 'OEE Değeri (%)'
]
korelasyon_matrisi = df[korelasyon_sutunlari].corr()
im = axes[2, 2].imshow(korelasyon_matrisi, cmap='RdYlBu_r', aspect='auto', vmin=-1, vmax=1)
axes[2, 2].set_xticks(range(len(korelasyon_sutunlari)))
axes[2, 2].set_yticks(range(len(korelasyon_sutunlari)))
axes[2, 2].set_xticklabels([s.replace(' (Saat)', '').replace(' (%)', '')[:12] 
                            for s in korelasyon_sutunlari], rotation=90, fontsize=7)
axes[2, 2].set_yticklabels([s.replace(' (Saat)', '').replace(' (%)', '')[:12] 
                            for s in korelasyon_sutunlari], fontsize=7)
axes[2, 2].set_title('Değişkenler Arası Korelasyon', fontsize=13, fontweight='bold')
plt.colorbar(im, ax=axes[2, 2], shrink=0.8)

plt.tight_layout()
plt.savefig(f"{kayit_konumu}/EDA_Gorsellestirme.png", dpi=150, bbox_inches='tight')
plt.show()
print("✅ Görseller kaydedildi: EDA_Gorsellestirme.png")

# ============================================================
# 6. ZAMAN SERİSİ ANALİZİ
# ============================================================
print("\n" + "=" * 70)
print("📈 6. ZAMAN SERİSİ ÖZETİ")
print("=" * 70)

gunluk_ozet = df.groupby('Tarih_DT').agg(
    toplam_bobin=('Kaliteli Bobin Adedi', 'sum'),
    toplam_metraj=('Kaliteli Toplam Metraj', 'sum'),
    toplam_fire=('Fire Bobin Adedi', 'sum'),
    ortalama_oee=('OEE Değeri (%)', 'mean'),
    toplam_bakim=('Bakım Duruşu (Saat)', 'sum'),
    toplam_ariza=('Arıza Duruşu (Saat)', 'sum'),
    is_emri_sayisi=('İş Emri No', 'count')
).reset_index()

print(f"\n📊 Günlük Ortalamalar:")
print(f"  Bobin Üretimi: {gunluk_ozet['toplam_bobin'].mean():.0f} adet/gün")
print(f"  Metraj: {gunluk_ozet['toplam_metraj'].mean():.0f} metre/gün")
print(f"  Fire: {gunluk_ozet['toplam_fire'].mean():.0f} bobin/gün")
print(f"  OEE: %{gunluk_ozet['ortalama_oee'].mean():.1f}")
print(f"  İş Emri: {gunluk_ozet['is_emri_sayisi'].mean():.0f} adet/gün")
print(f"  Bakım Duruşu: {gunluk_ozet['toplam_bakim'].mean():.1f} saat/gün")
print(f"  Arıza Duruşu: {gunluk_ozet['toplam_ariza'].mean():.1f} saat/gün")

# Haftalık pattern - DÜZELTİLEN KISIM
gunluk_ozet['gun_no'] = gunluk_ozet['Tarih_DT'].dt.dayofweek  # 0=Pazartesi, 5=Cumartesi
gun_isimleri = {0: 'Pazartesi', 1: 'Salı', 2: 'Çarşamba', 3: 'Perşembe', 4: 'Cuma', 5: 'Cumartesi'}
gunluk_ozet['gun_adi'] = gunluk_ozet['gun_no'].map(gun_isimleri)

gun_ozet = gunluk_ozet.groupby('gun_adi')['toplam_bobin'].agg(['mean', 'std']).round(0)
gun_sirasi_tr = ['Pazartesi', 'Salı', 'Çarşamba', 'Perşembe', 'Cuma', 'Cumartesi']
gun_ozet = gun_ozet.reindex(gun_sirasi_tr)

print(f"\n📅 Gün Bazlı Üretim Pattern'i:")
for gun in gun_sirasi_tr:
    if gun in gun_ozet.index:
        print(f"  {gun}: {gun_ozet.loc[gun, 'mean']:.0f} ± {gun_ozet.loc[gun, 'std']:.0f} bobin")

# ============================================================
# 7. ÖNEMLİ BULGULAR VE ÖZET
# ============================================================
print("\n" + "=" * 70)
print("🔍 7. ÖNEMLİ BULGULAR")
print("=" * 70)

print(f"""
📌 VERİ SETİ ÖZETİ:
  • {df['Tesis Kodu'].nunique()} tesis, {df['Hat Kodu'].nunique()} üretim hattı
  • {df['Ürün Kodu'].nunique()} farklı ürün (Zorluk: {df['Üretim Zorluk Derecesi'].min()}-{df['Üretim Zorluk Derecesi'].max()})
  • {df['Operatör Sicil No'].nunique()} operatör
  • Toplam {df['Kaliteli Bobin Adedi'].sum():,} bobin üretim
  • Fire oranı: %{(df['Fire Bobin Adedi'].sum() / (df['Kaliteli Bobin Adedi'].sum() + df['Fire Bobin Adedi'].sum()) * 100):.2f}

📌 DRAMATİK ETKİLER:
  • Bakım duruşu olan iş emri: {(df['Bakım Duruşu (Saat)'] > 0).sum():,} adet
  • Arıza duruşu olan iş emri: {(df['Arıza Duruşu (Saat)'] > 0).sum():,} adet
  • Toplam kayıp süre: {df['Bakım Duruşu (Saat)'].sum() + df['Arıza Duruşu (Saat)'].sum():.0f} saat

📌 MODEL İÇİN ÖNEMLİ DEĞİŞKENLER:
  • En yüksek korelasyon (Üretim ←→ OEE): {df['Kaliteli Bobin Adedi'].corr(df['OEE Değeri (%)']):.3f}
  • Zorluk derecesi ←→ Setup süresi korelasyonu: {df['Üretim Zorluk Derecesi'].corr(df['Setup Süresi (Saat)']):.3f}
  • Arıza ←→ OEE korelasyonu: {df['Arıza Duruşu (Saat)'].corr(df['OEE Değeri (%)']):.3f}
""")

print("=" * 70)
print("✅ EDA TAMAMLANDI!")
print(f"📁 Görseller: {kayit_konumu}\\EDA_Gorsellestirme.png")
print("=" * 70)