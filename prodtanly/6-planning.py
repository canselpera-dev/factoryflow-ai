import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import random
import os

# ============================================================
# 0. KAYIT KONUMU
# ============================================================
kayit_konumu = r"C:\Users\canse\OneDrive\Masaüstü\Production Forecasting"
if not os.path.exists(kayit_konumu):
    os.makedirs(kayit_konumu)

np.random.seed(42)
random.seed(42)

# ============================================================
# 1. ÜRETİM TESİSİ TANIMLARI
# ============================================================

# Üretim Hatları ve Kapasiteleri
hat_kapasiteleri = {
    'XTR-1001': {
        'hat_adi': 'Mikro Extruder Hattı Alfa',
        'hat_tipi': 'Mikro',
        'max_islem_hizi': 120,  # m/dk
        'gunluk_calisma_saati': 20,  # 3 vardiya - bakım hariç
        'max_gunluk_metraj': 120 * 60 * 20,  # Teorik max
        'urun_tipleri': ['KBL-101', 'KBL-102', 'KBL-601', 'KBL-602', 'KBL-801', 'KBL-902'],
        'operatörler': ['OPR-1001', 'OPR-1002', 'OPR-1003']
    },
    'XTR-1002': {
        'hat_adi': 'Standart Extruder Hattı Beta',
        'hat_tipi': 'Standart',
        'max_islem_hizi': 80,
        'gunluk_calisma_saati': 20,
        'max_gunluk_metraj': 80 * 60 * 20,
        'urun_tipleri': ['KBL-201', 'KBL-202', 'KBL-501', 'KBL-502', 'KBL-901'],
        'operatörler': ['OPR-1002', 'OPR-1005', 'OPR-1003']
    },
    'XTR-1003': {
        'hat_adi': 'Ağır Hizmet Extruder Hattı Gamma',
        'hat_tipi': 'Ağır Hizmet',
        'max_islem_hizi': 40,
        'gunluk_calisma_saati': 18,
        'max_gunluk_metraj': 40 * 60 * 18,
        'urun_tipleri': ['KBL-301', 'KBL-302', 'KBL-401'],
        'operatörler': ['OPR-1004', 'OPR-1005', 'OPR-2005']
    },
    'XTR-1004': {
        'hat_adi': 'Yüksek Hızlı Extruder Hattı Delta',
        'hat_tipi': 'Mikro',
        'max_islem_hizi': 130,
        'gunluk_calisma_saati': 20,
        'max_gunluk_metraj': 130 * 60 * 20,
        'urun_tipleri': ['KBL-101', 'KBL-102', 'KBL-601', 'KBL-602', 'KBL-801', 'KBL-802'],
        'operatörler': ['OPR-1001', 'OPR-2001', 'OPR-2002']
    },
    'XTR-2001': {
        'hat_adi': 'Özel Üretim Extruder Hattı Epsilon',
        'hat_tipi': 'Özel Üretim',
        'max_islem_hizi': 50,
        'gunluk_calisma_saati': 18,
        'max_gunluk_metraj': 50 * 60 * 18,
        'urun_tipleri': ['KBL-701', 'KBL-702'],
        'operatörler': ['OPR-2001', 'OPR-2004', 'OPR-2005']
    },
    'XTR-2002': {
        'hat_adi': 'Standart Extruder Hattı Zeta',
        'hat_tipi': 'Standart',
        'max_islem_hizi': 90,
        'gunluk_calisma_saati': 20,
        'max_gunluk_metraj': 90 * 60 * 20,
        'urun_tipleri': ['KBL-201', 'KBL-202', 'KBL-501', 'KBL-502', 'KBL-901'],
        'operatörler': ['OPR-2002', 'OPR-2003', 'OPR-2005']
    },
    'XTR-2003': {
        'hat_adi': 'Ağır Hizmet Extruder Hattı Eta',
        'hat_tipi': 'Ağır Hizmet',
        'max_islem_hizi': 45,
        'gunluk_calisma_saati': 18,
        'max_gunluk_metraj': 45 * 60 * 18,
        'urun_tipleri': ['KBL-301', 'KBL-302', 'KBL-401'],
        'operatörler': ['OPR-2003', 'OPR-2004', 'OPR-2005']
    },
    'XTR-2004': {
        'hat_adi': 'Kombine Extruder Hattı Theta',
        'hat_tipi': 'Kombine',
        'max_islem_hizi': 110,
        'gunluk_calisma_saati': 20,
        'max_gunluk_metraj': 110 * 60 * 20,
        'urun_tipleri': ['KBL-101', 'KBL-102', 'KBL-201', 'KBL-202', 'KBL-501', 'KBL-601', 'KBL-602', 'KBL-801', 'KBL-802', 'KBL-902'],
        'operatörler': ['OPR-2001', 'OPR-2002', 'OPR-2003']
    },
    'XTR-2005': {
        'hat_adi': 'Süper Ağır Hizmet Extruder Hattı Iota',
        'hat_tipi': 'Süper Ağır',
        'max_islem_hizi': 30,
        'gunluk_calisma_saati': 16,
        'max_gunluk_metraj': 30 * 60 * 16,
        'urun_tipleri': ['KBL-401', 'KBL-302', 'KBL-301'],
        'operatörler': ['OPR-2004', 'OPR-2005']
    },
}

# Ürün Kataloğu
urun_katalogu = {
    'KBL-101': {'urun_adi': 'NYA 2x1.5 mm² Bina İçi Tesisat', 'standart_bobin_metraj': 100, 'uretim_zorluk_puani': 1, 'izolasyon_malzemesi': 'PVC', 'malzeme_kodu': 'CU-ETP'},
    'KBL-102': {'urun_adi': 'NYA 3x2.5 mm² Bina İçi Tesisat', 'standart_bobin_metraj': 100, 'uretim_zorluk_puani': 2, 'izolasyon_malzemesi': 'PVC', 'malzeme_kodu': 'CU-ETP'},
    'KBL-201': {'urun_adi': 'NYY 4x6 mm² Yeraltı Enerji', 'standart_bobin_metraj': 200, 'uretim_zorluk_puani': 3, 'izolasyon_malzemesi': 'PVC', 'malzeme_kodu': 'CU-ETP'},
    'KBL-202': {'urun_adi': 'NYY 4x16 mm² Yeraltı Enerji', 'standart_bobin_metraj': 150, 'uretim_zorluk_puani': 4, 'izolasyon_malzemesi': 'XLPE', 'malzeme_kodu': 'CU-ETP'},
    'KBL-301': {'urun_adi': 'NYY 5x25 mm² Endüstriyel Enerji', 'standart_bobin_metraj': 100, 'uretim_zorluk_puani': 5, 'izolasyon_malzemesi': 'XLPE', 'malzeme_kodu': 'CU-ETP'},
    'KBL-302': {'urun_adi': 'YVV 3x35 mm² Ağır Sanayi', 'standart_bobin_metraj': 80, 'uretim_zorluk_puani': 6, 'izolasyon_malzemesi': 'XLPE', 'malzeme_kodu': 'AL-1350'},
    'KBL-401': {'urun_adi': 'YVV 4x50 mm² Çok Ağır Sanayi', 'standart_bobin_metraj': 60, 'uretim_zorluk_puani': 7, 'izolasyon_malzemesi': 'XLPE', 'malzeme_kodu': 'AL-1350'},
    'KBL-501': {'urun_adi': 'NHXMH 3x1.5 mm² Halogen-Free Bina', 'standart_bobin_metraj': 200, 'uretim_zorluk_puani': 2, 'izolasyon_malzemesi': 'Halogen-Free', 'malzeme_kodu': 'CU-ETP'},
    'KBL-502': {'urun_adi': 'NHXMH 5x2.5 mm² Halogen-Free Endüstriyel', 'standart_bobin_metraj': 150, 'uretim_zorluk_puani': 4, 'izolasyon_malzemesi': 'Halogen-Free', 'malzeme_kodu': 'CU-ETP'},
    'KBL-601': {'urun_adi': 'LIHCH 2x0.75 mm² Sinyal Kontrol', 'standart_bobin_metraj': 300, 'uretim_zorluk_puani': 1, 'izolasyon_malzemesi': 'PVC', 'malzeme_kodu': 'CU-ETP'},
    'KBL-602': {'urun_adi': 'LIHCH 4x1.0 mm² Kontrol Kumanda', 'standart_bobin_metraj': 250, 'uretim_zorluk_puani': 2, 'izolasyon_malzemesi': 'PVC', 'malzeme_kodu': 'CU-ETP'},
    'KBL-701': {'urun_adi': 'Fiber Optik SM 12C Dış Tesisat', 'standart_bobin_metraj': 1000, 'uretim_zorluk_puani': 8, 'izolasyon_malzemesi': 'LSZH', 'malzeme_kodu': 'FO-SM-9'},
    'KBL-702': {'urun_adi': 'Fiber Optik MM 24C Data Center', 'standart_bobin_metraj': 800, 'uretim_zorluk_puani': 9, 'izolasyon_malzemesi': 'LSZH', 'malzeme_kodu': 'FO-SM-9'},
    'KBL-801': {'urun_adi': 'CAT6 UTP 4P Data İletişim', 'standart_bobin_metraj': 305, 'uretim_zorluk_puani': 3, 'izolasyon_malzemesi': 'PE', 'malzeme_kodu': 'CU-ETP'},
    'KBL-802': {'urun_adi': 'CAT7 SFTP 4P Yüksek Performans', 'standart_bobin_metraj': 200, 'uretim_zorluk_puani': 5, 'izolasyon_malzemesi': 'LSZH', 'malzeme_kodu': 'CU-ETP'},
    'KBL-901': {'urun_adi': 'H07RN-F 5G6 mm² Kauçuk Endüstriyel', 'standart_bobin_metraj': 100, 'uretim_zorluk_puani': 6, 'izolasyon_malzemesi': 'Kauçuk', 'malzeme_kodu': 'CU-ETP'},
    'KBL-902': {'urun_adi': 'H07RN-F 3G2.5 mm² Kauçuk Seyyar', 'standart_bobin_metraj': 150, 'uretim_zorluk_puani': 3, 'izolasyon_malzemesi': 'Kauçuk', 'malzeme_kodu': 'CU-ETP'},
}

# Operatör havuzu
operatorler = {
    'OPR-1001': {'ad_soyad': 'AHMET YILMAZ', 'yetkinlik': 5, 'performans': 1.15},
    'OPR-1002': {'ad_soyad': 'MEHMET DEMİR', 'yetkinlik': 4, 'performans': 1.08},
    'OPR-1003': {'ad_soyad': 'MUSTAFA ŞAHİN', 'yetkinlik': 2, 'performans': 0.85},
    'OPR-1004': {'ad_soyad': 'HÜSEYİN ÇELİK', 'yetkinlik': 5, 'performans': 1.20},
    'OPR-1005': {'ad_soyad': 'ALİ KAYA', 'yetkinlik': 3, 'performans': 0.95},
    'OPR-2001': {'ad_soyad': 'İSMAİL ÖZDEMİR', 'yetkinlik': 4, 'performans': 1.10},
    'OPR-2002': {'ad_soyad': 'OSMAN YILDIZ', 'yetkinlik': 1, 'performans': 0.70},
    'OPR-2003': {'ad_soyad': 'HASAN AYDIN', 'yetkinlik': 4, 'performans': 1.05},
    'OPR-2004': {'ad_soyad': 'İBRAHİM ÖZTÜRK', 'yetkinlik': 5, 'performans': 1.25},
    'OPR-2005': {'ad_soyad': 'SÜLEYMAN KOÇ', 'yetkinlik': 3, 'performans': 0.90},
}

# ============================================================
# 2. BASKI PLANI PROGRAMI (İŞ EMRİ PLANLAMA MODÜLÜ)
# ============================================================

class IsEmriPlanlamaModulu:
    """
    Kablo Üretim Tesisi - İş Emri Planlama Modülü
    Her gün için: hangi makine, hangi işleri, kaç adet üretecek?
    """
    
    def __init__(self, hat_kapasiteleri, urun_katalogu, operatorler):
        self.hat_kapasiteleri = hat_kapasiteleri
        self.urun_katalogu = urun_katalogu
        self.operatorler = operatorler
        
    def makine_gunluk_kapasite(self, hat_kodu, bakim_var_mi=False):
        """Bir makinenin günlük teorik üretim kapasitesi (metre cinsinden)"""
        hat = self.hat_kapasiteleri[hat_kodu]
        calisma_saati = hat['gunluk_calisma_saati']
        
        # Bakım günü ise kapasite düşer
        if bakim_var_mi:
            calisma_saati -= 3  # 3 saat bakım
        
        return hat['max_islem_hizi'] * 60 * calisma_saati
    
    def is_emri_uretim_suresi(self, urun_kodu, bobin_adedi, hat_kodu, operator_kodu):
        """Bir iş emrinin tahmini üretim süresi (saat)"""
        urun = self.urun_katalogu[urun_kodu]
        hat = self.hat_kapasiteleri[hat_kodu]
        operator = self.operatorler[operator_kodu]
        
        toplam_metraj = bobin_adedi * urun['standart_bobin_metraj']
        # Efektif hız = max_hiz * operatör_performans - zorluk_etkisi
        efektif_hiz = hat['max_islem_hizi'] * operator['performans'] - (urun['uretim_zorluk_puani'] * 2.5)
        efektif_hiz = max(hat['max_islem_hizi'] * 0.3, efektif_hiz)
        
        setup_suresi = urun['uretim_zorluk_puani'] * 0.08 + 0.2  # saat
        uretim_suresi = toplam_metraj / efektif_hiz / 60  # saat
        
        return setup_suresi + uretim_suresi
    
    def gunluk_plan_olustur(self, tarih):
        """
        Belirli bir tarih için günlük iş emri planı oluşturur.
        
        Returns:
            DataFrame: O günün iş emri planı
        """
        gun_no = tarih.weekday()
        bakim_gunu = (gun_no == 0 or gun_no == 5)  # Pazartesi veya Cumartesi
        
        plan_listesi = []
        is_emri_sayaci = 0
        
        for hat_kodu, hat in self.hat_kapasiteleri.items():
            # O günkü kapasite
            gunluk_kapasite = self.makine_gunluk_kapasite(hat_kodu, bakim_gunu)
            kalan_kapasite = gunluk_kapasite
            
            # Operatör ata
            operator_kodu = random.choice(hat['operatörler'])
            
            # İş emirlerini sırayla ekle (kapasite dolana kadar)
            while kalan_kapasite > 500:  # En az 500 metre kaldıysa yeni iş ekle
                # Rastgele ürün seç
                urun_kodu = random.choice(hat['urun_tipleri'])
                urun = self.urun_katalogu[urun_kodu]
                
                # Bobin adedi belirle (kapasiteye göre)
                max_bobin = int(kalan_kapasite / urun['standart_bobin_metraj'])
                if max_bobin < 5:
                    break
                
                bobin_adedi = random.randint(5, min(max_bobin, 60))
                toplam_metraj = bobin_adedi * urun['standart_bobin_metraj']
                
                # Üretim süresini hesapla
                uretim_suresi = self.is_emri_uretim_suresi(urun_kodu, bobin_adedi, hat_kodu, operator_kodu)
                
                # Kapasiteden düş
                kalan_kapasite -= toplam_metraj
                
                is_emri_sayaci += 1
                plan_listesi.append({
                    'Plan_Tarihi': tarih.strftime('%d.%m.%Y'),
                    'Hat_Kodu': hat_kodu,
                    'Hat_Adi': hat['hat_adi'],
                    'Is_Emri_Sira': is_emri_sayaci,
                    'Urun_Kodu': urun_kodu,
                    'Urun_Adi': urun['urun_adi'],
                    'Malzeme_Kodu': urun['malzeme_kodu'],
                    'Izolasyon_Cinsi': urun['izolasyon_malzemesi'],
                    'Zorluk_Derecesi': urun['uretim_zorluk_puani'],
                    'Planlanan_Bobin': bobin_adedi,
                    'Planlanan_Metraj': toplam_metraj,
                    'Bobin_Metraj': urun['standart_bobin_metraj'],
                    'Operator_Kodu': operator_kodu,
                    'Tahmini_Sure_Saat': round(uretim_suresi, 2),
                    'Bakim_Gunu': bakim_gunu,
                    'Kapasite_Kullanim': round((1 - kalan_kapasite/gunluk_kapasite) * 100, 1)
                })
        
        return pd.DataFrame(plan_listesi)
    
    def donem_plani_olustur(self, baslangic_tarih, bitis_tarih):
        """Tarih aralığı için tam plan oluşturur"""
        tum_plan = []
        tarih = baslangic_tarih
        
        while tarih <= bitis_tarih:
            if tarih.weekday() < 6:  # Pazar hariç
                gunluk_plan = self.gunluk_plan_olustur(tarih)
                tum_plan.append(gunluk_plan)
            tarih += timedelta(days=1)
        
        return pd.concat(tum_plan, ignore_index=True)


# ============================================================
# 3. PLANI OLUŞTUR VE KAYDET
# ============================================================

print("=" * 70)
print("🏭 BASKI PLANI PROGRAMI (İŞ EMRİ PLANLAMA MODÜLÜ)")
print("=" * 70)

planlama = IsEmriPlanlamaModulu(hat_kapasiteleri, urun_katalogu, operatorler)

baslangic = datetime(2025, 6, 2)
bitis = datetime(2026, 5, 30)

print(f"📅 Plan dönemi: {baslangic.date()} → {bitis.date()}")
print("🔄 Plan oluşturuluyor...")

df_plan = planlama.donem_plani_olustur(baslangic, bitis)

print(f"\n✅ Plan hazır!")
print(f"   Toplam iş emri: {len(df_plan):,}")
print(f"   Çalışma günü: {df_plan['Plan_Tarihi'].nunique()}")
print(f"   Aktif hat: {df_plan['Hat_Kodu'].nunique()}")
print(f"   Farklı ürün: {df_plan['Urun_Kodu'].nunique()}")

# Günlük özet
gunluk_ozet = df_plan.groupby('Plan_Tarihi').agg(
    Toplam_Is_Emri=('Is_Emri_Sira', 'count'),
    Toplam_Bobin=('Planlanan_Bobin', 'sum'),
    Toplam_Metraj=('Planlanan_Metraj', 'sum'),
    Ortalama_Zorluk=('Zorluk_Derecesi', 'mean'),
    Aktif_Hat=('Hat_Kodu', 'nunique'),
    Aktif_Operator=('Operator_Kodu', 'nunique'),
    Bakim_Gunu=('Bakim_Gunu', 'first')
).reset_index()

print(f"\n📊 Günlük Ortalamalar:")
print(f"   İş emri: {gunluk_ozet['Toplam_Is_Emri'].mean():.0f}")
print(f"   Bobin:   {gunluk_ozet['Toplam_Bobin'].mean():.0f}")
print(f"   Metraj:  {gunluk_ozet['Toplam_Metraj'].mean():.0f} m")
print(f"   Zorluk:  {gunluk_ozet['Ortalama_Zorluk'].mean():.2f}")

# ============================================================
# 4. KAYDET
# ============================================================

plan_dosyasi = os.path.join(kayit_konumu, 'is_emri_plani.csv')
ozet_dosyasi = os.path.join(kayit_konumu, 'gunluk_plan_ozeti.csv')

df_plan.to_csv(plan_dosyasi, index=False, sep=';')
gunluk_ozet.to_csv(ozet_dosyasi, index=False, sep=';')

print(f"\n📁 Kaydedildi:")
print(f"   📄 {plan_dosyasi}")
print(f"   📄 {ozet_dosyasi}")

print("\n📋 ÖRNEK PLAN (İLK 10 SATIR):")
print(df_plan[['Plan_Tarihi', 'Hat_Kodu', 'Urun_Kodu', 'Planlanan_Bobin', 'Tahmini_Sure_Saat']].head(10).to_string())

print("\n📋 GÜNLÜK ÖZET (İLK 5 GÜN):")
print(gunluk_ozet[['Plan_Tarihi', 'Toplam_Is_Emri', 'Toplam_Bobin', 'Ortalama_Zorluk', 'Bakim_Gunu']].head().to_string())

print("\n" + "=" * 70)
print("✅ BASKI PLANI PROGRAMI HAZIR!")
print("=" * 70)