import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import random
import os

# Rastgelelik sabiti - tekrarlanabilirlik için
np.random.seed(42)
random.seed(42)

# ============================================================
# 0. KAYIT KONUMU
# ============================================================
kayit_konumu = r"C:\Users\canse\OneDrive\Masaüstü\Production Forecasting"

# Klasör yoksa oluştur
if not os.path.exists(kayit_konumu):
    os.makedirs(kayit_konumu)
    print(f"📁 Klasör oluşturuldu: {kayit_konumu}")

# ============================================================
# 1. KABLO FABRİKASI PARAMETRELERİ
# ============================================================

# Tesis bilgileri
tesisler = {
    'KABLO-IST': {
        'tesis_adi': 'İstanbul Kablo Üretim Tesisi',
        'bolge': 'Marmara',
        'vardiya_sistemi': '3 Vardiya'
    },
    'KABLO-ANK': {
        'tesis_adi': 'Ankara Kablo Üretim Tesisi', 
        'bolge': 'İç Anadolu',
        'vardiya_sistemi': '2 Vardiya'
    }
}

# Bakım takvimi - Her hat için planlı bakım günleri
# Pazartesi sabah 06:00-08:00 ve Cumartesi gece 22:00-00:00
bakim_takvimi = {
    'Pazartesi_Sabah': {'baslangic_saat': 6, 'bitis_saat': 8, 'min_sure': 2.0, 'max_sure': 3.0},
    'Cumartesi_Gece': {'baslangic_saat': 22, 'bitis_saat': 0, 'min_sure': 2.5, 'max_sure': 4.0}
}

# Ekstrüzyon Hatları - Yeni kodlama sistemi
uretim_hatlari = {
    'XTR-1001': {
        'hat_adi': 'Mikro Extruder Hattı Alfa',
        'max_islem_hizi': 120, 
        'min_islem_hizi': 60, 
        'tesis': 'KABLO-IST', 
        'hat_tipi': 'Mikro',
        'teknoloji_seviyesi': 'Endüstri 4.0',
        'kurulum_yili': 2018,
        'ariza_egilimi': 0.08,  # %8 ariza ihtimali
        'bakim_kritikligi': 'Düşük'
    },
    'XTR-1002': {
        'hat_adi': 'Standart Extruder Hattı Beta',
        'max_islem_hizi': 80, 
        'min_islem_hizi': 40, 
        'tesis': 'KABLO-IST', 
        'hat_tipi': 'Standart',
        'teknoloji_seviyesi': 'Endüstri 3.0',
        'kurulum_yili': 2015,
        'ariza_egilimi': 0.15,  # %15 ariza ihtimali (eski hat)
        'bakim_kritikligi': 'Orta'
    },
    'XTR-1003': {
        'hat_adi': 'Ağır Hizmet Extruder Hattı Gamma',
        'max_islem_hizi': 40, 
        'min_islem_hizi': 15, 
        'tesis': 'KABLO-IST', 
        'hat_tipi': 'Ağır Hizmet',
        'teknoloji_seviyesi': 'Endüstri 3.0',
        'kurulum_yili': 2012,
        'ariza_egilimi': 0.22,  # %22 ariza ihtimali (en eski hat)
        'bakim_kritikligi': 'Yüksek'
    },
    'XTR-1004': {
        'hat_adi': 'Yüksek Hızlı Extruder Hattı Delta',
        'max_islem_hizi': 130, 
        'min_islem_hizi': 70, 
        'tesis': 'KABLO-IST', 
        'hat_tipi': 'Mikro',
        'teknoloji_seviyesi': 'Endüstri 4.0',
        'kurulum_yili': 2022,
        'ariza_egilimi': 0.05,  # %5 ariza ihtimali (yeni hat)
        'bakim_kritikligi': 'Düşük'
    },
    'XTR-2001': {
        'hat_adi': 'Özel Üretim Extruder Hattı Epsilon',
        'max_islem_hizi': 50, 
        'min_islem_hizi': 20, 
        'tesis': 'KABLO-ANK', 
        'hat_tipi': 'Özel Üretim',
        'teknoloji_seviyesi': 'Endüstri 4.0',
        'kurulum_yili': 2020,
        'ariza_egilimi': 0.10,  # %10 ariza ihtimali
        'bakim_kritikligi': 'Orta'
    },
    'XTR-2002': {
        'hat_adi': 'Standart Extruder Hattı Zeta',
        'max_islem_hizi': 90, 
        'min_islem_hizi': 45, 
        'tesis': 'KABLO-ANK', 
        'hat_tipi': 'Standart',
        'teknoloji_seviyesi': 'Endüstri 3.0',
        'kurulum_yili': 2016,
        'ariza_egilimi': 0.18,  # %18 ariza ihtimali
        'bakim_kritikligi': 'Orta'
    },
    'XTR-2003': {
        'hat_adi': 'Ağır Hizmet Extruder Hattı Eta',
        'max_islem_hizi': 45, 
        'min_islem_hizi': 20, 
        'tesis': 'KABLO-ANK', 
        'hat_tipi': 'Ağır Hizmet',
        'teknoloji_seviyesi': 'Endüstri 3.0',
        'kurulum_yili': 2014,
        'ariza_egilimi': 0.20,  # %20 ariza ihtimali
        'bakim_kritikligi': 'Yüksek'
    },
    'XTR-2004': {
        'hat_adi': 'Kombine Extruder Hattı Theta',
        'max_islem_hizi': 110, 
        'min_islem_hizi': 55, 
        'tesis': 'KABLO-ANK', 
        'hat_tipi': 'Kombine',
        'teknoloji_seviyesi': 'Endüstri 4.0',
        'kurulum_yili': 2021,
        'ariza_egilimi': 0.07,  # %7 ariza ihtimali
        'bakim_kritikligi': 'Düşük'
    },
    'XTR-2005': {
        'hat_adi': 'Süper Ağır Hizmet Extruder Hattı Iota',
        'max_islem_hizi': 30, 
        'min_islem_hizi': 10, 
        'tesis': 'KABLO-ANK', 
        'hat_tipi': 'Süper Ağır',
        'teknoloji_seviyesi': 'Endüstri 3.0',
        'kurulum_yili': 2010,
        'ariza_egilimi': 0.25,  # %25 ariza ihtimali (en eski ve en sorunlu hat)
        'bakim_kritikligi': 'Çok Yüksek'
    },
}

# İletken Malzeme Tipleri
malzeme_tipleri = {
    'CU-ETP': {
        'malzeme_adi': 'Elektrolitik Bakır', 
        'fire_toleransi_alt': 0.01, 
        'fire_toleransi_ust': 0.05, 
        'birim_maliyet': 125.5,
        'sicaklik_dayanimi': '90°C'
    },
    'AL-1350': {
        'malzeme_adi': 'Alüminyum Alaşım', 
        'fire_toleransi_alt': 0.03, 
        'fire_toleransi_ust': 0.09, 
        'birim_maliyet': 78.3,
        'sicaklik_dayanimi': '75°C'
    },
    'FO-SM-9': {
        'malzeme_adi': 'Fiber Optik Single Mode', 
        'fire_toleransi_alt': 0.005, 
        'fire_toleransi_ust': 0.02, 
        'birim_maliyet': 340.0,
        'sicaklik_dayanimi': '85°C'
    },
}

# Ürün Kataloğu - Kablo spesifikasyonları (ZORLUK DERECESİ EKLENDİ)
urun_katalogu = {
    'KBL-101': {
        'urun_adi': 'NYA 2x1.5 mm² Bina İçi Tesisat',
        'cap_sinifi': 'Mikro',
        'standart_bobin_metraj': 100,
        'agirlik_sinifi': 'Hafif',
        'izolasyon_malzemesi': 'PVC',
        'uretim_zorluk_puani': 1,
        'max_sicaklik': '70°C',
        'sertifikasi': 'TSE, CE',
        'ozel_gereksinim': False
    },
    'KBL-102': {
        'urun_adi': 'NYA 3x2.5 mm² Bina İçi Tesisat',
        'cap_sinifi': 'Mikro',
        'standart_bobin_metraj': 100,
        'agirlik_sinifi': 'Hafif',
        'izolasyon_malzemesi': 'PVC',
        'uretim_zorluk_puani': 2,
        'max_sicaklik': '70°C',
        'sertifikasi': 'TSE, CE',
        'ozel_gereksinim': False
    },
    'KBL-201': {
        'urun_adi': 'NYY 4x6 mm² Yeraltı Enerji',
        'cap_sinifi': 'Standart',
        'standart_bobin_metraj': 200,
        'agirlik_sinifi': 'Orta',
        'izolasyon_malzemesi': 'PVC',
        'uretim_zorluk_puani': 3,
        'max_sicaklik': '70°C',
        'sertifikasi': 'TSE, CE, VDE',
        'ozel_gereksinim': False
    },
    'KBL-202': {
        'urun_adi': 'NYY 4x16 mm² Yeraltı Enerji',
        'cap_sinifi': 'Standart',
        'standart_bobin_metraj': 150,
        'agirlik_sinifi': 'Orta',
        'izolasyon_malzemesi': 'XLPE',
        'uretim_zorluk_puani': 4,
        'max_sicaklik': '90°C',
        'sertifikasi': 'TSE, CE, VDE',
        'ozel_gereksinim': True  # XLPE özel sıcaklık kontrolü gerektirir
    },
    'KBL-301': {
        'urun_adi': 'NYY 5x25 mm² Endüstriyel Enerji',
        'cap_sinifi': 'Ağır',
        'standart_bobin_metraj': 100,
        'agirlik_sinifi': 'Ağır',
        'izolasyon_malzemesi': 'XLPE',
        'uretim_zorluk_puani': 5,
        'max_sicaklik': '90°C',
        'sertifikasi': 'TSE, CE, VDE, UL',
        'ozel_gereksinim': True
    },
    'KBL-302': {
        'urun_adi': 'YVV 3x35 mm² Ağır Sanayi',
        'cap_sinifi': 'Ağır',
        'standart_bobin_metraj': 80,
        'agirlik_sinifi': 'Ağır',
        'izolasyon_malzemesi': 'XLPE',
        'uretim_zorluk_puani': 6,
        'max_sicaklik': '90°C',
        'sertifikasi': 'TSE, CE, VDE',
        'ozel_gereksinim': True
    },
    'KBL-401': {
        'urun_adi': 'YVV 4x50 mm² Çok Ağır Sanayi',
        'cap_sinifi': 'Süper Ağır',
        'standart_bobin_metraj': 60,
        'agirlik_sinifi': 'Çok Ağır',
        'izolasyon_malzemesi': 'XLPE',
        'uretim_zorluk_puani': 7,
        'max_sicaklik': '90°C',
        'sertifikasi': 'TSE, CE, VDE, UL',
        'ozel_gereksinim': True
    },
    'KBL-501': {
        'urun_adi': 'NHXMH 3x1.5 mm² Halogen-Free Bina',
        'cap_sinifi': 'Mikro',
        'standart_bobin_metraj': 200,
        'agirlik_sinifi': 'Hafif',
        'izolasyon_malzemesi': 'Halogen-Free',
        'uretim_zorluk_puani': 2,
        'max_sicaklik': '90°C',
        'sertifikasi': 'TSE, CE, CPR',
        'ozel_gereksinim': False
    },
    'KBL-502': {
        'urun_adi': 'NHXMH 5x2.5 mm² Halogen-Free Endüstriyel',
        'cap_sinifi': 'Standart',
        'standart_bobin_metraj': 150,
        'agirlik_sinifi': 'Orta',
        'izolasyon_malzemesi': 'Halogen-Free',
        'uretim_zorluk_puani': 4,
        'max_sicaklik': '90°C',
        'sertifikasi': 'TSE, CE, CPR',
        'ozel_gereksinim': True
    },
    'KBL-601': {
        'urun_adi': 'LIHCH 2x0.75 mm² Sinyal Kontrol',
        'cap_sinifi': 'Mikro',
        'standart_bobin_metraj': 300,
        'agirlik_sinifi': 'Çok Hafif',
        'izolasyon_malzemesi': 'PVC',
        'uretim_zorluk_puani': 1,
        'max_sicaklik': '70°C',
        'sertifikasi': 'TSE, CE',
        'ozel_gereksinim': False
    },
    'KBL-602': {
        'urun_adi': 'LIHCH 4x1.0 mm² Kontrol Kumanda',
        'cap_sinifi': 'Mikro',
        'standart_bobin_metraj': 250,
        'agirlik_sinifi': 'Hafif',
        'izolasyon_malzemesi': 'PVC',
        'uretim_zorluk_puani': 2,
        'max_sicaklik': '70°C',
        'sertifikasi': 'TSE, CE',
        'ozel_gereksinim': False
    },
    'KBL-701': {
        'urun_adi': 'Fiber Optik SM 12C Dış Tesisat',
        'cap_sinifi': 'Özel',
        'standart_bobin_metraj': 1000,
        'agirlik_sinifi': 'Çok Hafif',
        'izolasyon_malzemesi': 'LSZH',
        'uretim_zorluk_puani': 8,
        'max_sicaklik': '85°C',
        'sertifikasi': 'TSE, CE, ITU-T',
        'ozel_gereksinim': True
    },
    'KBL-702': {
        'urun_adi': 'Fiber Optik MM 24C Data Center',
        'cap_sinifi': 'Özel',
        'standart_bobin_metraj': 800,
        'agirlik_sinifi': 'Çok Hafif',
        'izolasyon_malzemesi': 'LSZH',
        'uretim_zorluk_puani': 9,
        'max_sicaklik': '85°C',
        'sertifikasi': 'TSE, CE, ITU-T, TIA',
        'ozel_gereksinim': True
    },
    'KBL-801': {
        'urun_adi': 'CAT6 UTP 4P Data İletişim',
        'cap_sinifi': 'Mikro',
        'standart_bobin_metraj': 305,
        'agirlik_sinifi': 'Hafif',
        'izolasyon_malzemesi': 'PE',
        'uretim_zorluk_puani': 3,
        'max_sicaklik': '60°C',
        'sertifikasi': 'TSE, CE, ISO 11801',
        'ozel_gereksinim': False
    },
    'KBL-802': {
        'urun_adi': 'CAT7 SFTP 4P Yüksek Performans',
        'cap_sinifi': 'Mikro',
        'standart_bobin_metraj': 200,
        'agirlik_sinifi': 'Orta',
        'izolasyon_malzemesi': 'LSZH',
        'uretim_zorluk_puani': 5,
        'max_sicaklik': '60°C',
        'sertifikasi': 'TSE, CE, ISO 11801',
        'ozel_gereksinim': True
    },
    'KBL-901': {
        'urun_adi': 'H07RN-F 5G6 mm² Kauçuk Endüstriyel',
        'cap_sinifi': 'Standart',
        'standart_bobin_metraj': 100,
        'agirlik_sinifi': 'Ağır',
        'izolasyon_malzemesi': 'Kauçuk',
        'uretim_zorluk_puani': 6,
        'max_sicaklik': '60°C',
        'sertifikasi': 'TSE, CE, VDE',
        'ozel_gereksinim': True
    },
    'KBL-902': {
        'urun_adi': 'H07RN-F 3G2.5 mm² Kauçuk Seyyar',
        'cap_sinifi': 'Mikro',
        'standart_bobin_metraj': 150,
        'agirlik_sinifi': 'Orta',
        'izolasyon_malzemesi': 'Kauçuk',
        'uretim_zorluk_puani': 3,
        'max_sicaklik': '60°C',
        'sertifikasi': 'TSE, CE, VDE',
        'ozel_gereksinim': False
    },
}

# İzolasyon Malzeme Tipleri
izolasyon_malzemeleri = ['PVC', 'XLPE', 'Halogen-Free', 'LSZH', 'PE', 'Kauçuk', 'Silikon']

# Operatör havuzu - yeni ID sistemi
operator_havuzu = {
    'OPR-1001': {
        'ad_soyad': 'AHMET YILMAZ',
        'yetkinlik_seviyesi': 5,
        'uzmanlik_alani': ['Mikro'],
        'performans_katsayisi': 1.15,
        'tecrube_yili': 12,
        'vardiya_tercihi': 'Sabah'
    },
    'OPR-1002': {
        'ad_soyad': 'MEHMET DEMİR',
        'yetkinlik_seviyesi': 4,
        'uzmanlik_alani': ['Mikro', 'Standart'],
        'performans_katsayisi': 1.08,
        'tecrube_yili': 8,
        'vardiya_tercihi': 'Öğle'
    },
    'OPR-1003': {
        'ad_soyad': 'MUSTAFA ŞAHİN',
        'yetkinlik_seviyesi': 2,
        'uzmanlik_alani': ['Mikro'],
        'performans_katsayisi': 0.85,
        'tecrube_yili': 2,
        'vardiya_tercihi': 'Sabah'
    },
    'OPR-1004': {
        'ad_soyad': 'HÜSEYİN ÇELİK',
        'yetkinlik_seviyesi': 5,
        'uzmanlik_alani': ['Ağır', 'Süper Ağır'],
        'performans_katsayisi': 1.20,
        'tecrube_yili': 15,
        'vardiya_tercihi': 'Gece'
    },
    'OPR-1005': {
        'ad_soyad': 'ALİ KAYA',
        'yetkinlik_seviyesi': 3,
        'uzmanlik_alani': ['Standart'],
        'performans_katsayisi': 0.95,
        'tecrube_yili': 5,
        'vardiya_tercihi': 'Öğle'
    },
    'OPR-2001': {
        'ad_soyad': 'İSMAİL ÖZDEMİR',
        'yetkinlik_seviyesi': 4,
        'uzmanlik_alani': ['Mikro', 'Özel'],
        'performans_katsayisi': 1.10,
        'tecrube_yili': 10,
        'vardiya_tercihi': 'Sabah'
    },
    'OPR-2002': {
        'ad_soyad': 'OSMAN YILDIZ',
        'yetkinlik_seviyesi': 1,
        'uzmanlik_alani': ['Mikro'],
        'performans_katsayisi': 0.70,
        'tecrube_yili': 1,
        'vardiya_tercihi': 'Sabah'
    },
    'OPR-2003': {
        'ad_soyad': 'HASAN AYDIN',
        'yetkinlik_seviyesi': 4,
        'uzmanlik_alani': ['Standart', 'Ağır'],
        'performans_katsayisi': 1.05,
        'tecrube_yili': 9,
        'vardiya_tercihi': 'Öğle'
    },
    'OPR-2004': {
        'ad_soyad': 'İBRAHİM ÖZTÜRK',
        'yetkinlik_seviyesi': 5,
        'uzmanlik_alani': ['Özel', 'Süper Ağır'],
        'performans_katsayisi': 1.25,
        'tecrube_yili': 18,
        'vardiya_tercihi': 'Gece'
    },
    'OPR-2005': {
        'ad_soyad': 'SÜLEYMAN KOÇ',
        'yetkinlik_seviyesi': 3,
        'uzmanlik_alani': ['Standart', 'Ağır'],
        'performans_katsayisi': 0.90,
        'tecrube_yili': 4,
        'vardiya_tercihi': 'Öğle'
    },
}

# Proses Operasyon Kodları
operasyon_kodlari = {
    'OP-310': {'operasyon_adi': 'İLETKEN TEL ÇEKİM', 'birim_sure_dk': 15, 'ekipman_gereksinim': 'Tel Çekme Makinesi'},
    'OP-320': {'operasyon_adi': 'İZOLASYON EKSTRÜZYON', 'birim_sure_dk': 25, 'ekipman_gereksinim': 'Extruder'},
    'OP-330': {'operasyon_adi': 'DOLGU KAPLAMA', 'birim_sure_dk': 20, 'ekipman_gereksinim': 'Dolgu Extruder'},
    'OP-340': {'operasyon_adi': 'DIŞ KILIF SIVAMA', 'birim_sure_dk': 30, 'ekipman_gereksinim': 'Kılıf Extruder'},
    'OP-350': {'operasyon_adi': 'ÇELİK ZIRH SARMAL', 'birim_sure_dk': 40, 'ekipman_gereksinim': 'Zırh Sarma Makinesi'},
}

# ============================================================
# 2. TARİH ARALIĞI VE İŞ EMRİ ÜRETİMİ
# ============================================================

baslangic_tarih = datetime(2025, 6, 1)  # 1 yıl öncesi
bitis_tarih = datetime(2026, 5, 31)

def is_gunu_mu(tarih):
    """Haftasonu kontrolü - Pazar hariç çalışma var"""
    return tarih.weekday() < 6  # 0=Pazartesi, 6=Pazar

def bakim_kontrol(tarih, hat_kodu):
    """
    Bakım durumu kontrolü
    - Pazartesi sabah: 2-3 saat bakım
    - Cumartesi gece: 2.5-4 saat bakım
    """
    gun_adi = tarih.strftime('%A')
    
    if gun_adi == 'Monday':  # Pazartesi
        return np.random.uniform(2.0, 3.0)  # 2-3 saat
    elif gun_adi == 'Saturday':  # Cumartesi
        return np.random.uniform(2.5, 4.0)  # 2.5-4 saat
    
    return 0

def ariza_kontrol(hat_kodu):
    """Rastgele arıza kontrolü - 1-3 saat arası"""
    hat = uretim_hatlari[hat_kodu]
    if np.random.random() < hat['ariza_egilimi']:
        return np.random.uniform(1.0, 3.0)  # 1-3 saat arıza
    return 0

def is_emri_olustur(tarih, emir_no):
    """Tek bir üretim iş emri (satır) oluştur"""
    
    # Tesis ve üretim hattı seçimi
    tesis_kodu = random.choice(list(tesisler.keys()))
    uygun_hatlar = {k: v for k, v in uretim_hatlari.items() if v['tesis'] == tesis_kodu}
    hat_kodu = random.choice(list(uygun_hatlar.keys()))
    hat = uygun_hatlar[hat_kodu]
    
    # Ürün seçimi (hat tipine uygun)
    uygun_urunler = {
        k: v for k, v in urun_katalogu.items() 
        if (v['cap_sinifi'] == hat['hat_tipi'] or 
            (hat['hat_tipi'] == 'Kombine' and v['cap_sinifi'] in ['Mikro', 'Standart']) or
            (hat['hat_tipi'] == 'Mikro' and v['cap_sinifi'] in ['Mikro']))
    }
    
    if not uygun_urunler:
        uygun_urunler = urun_katalogu
    
    urun_kodu = random.choice(list(uygun_urunler.keys()))
    urun = uygun_urunler[urun_kodu]
    
    # Malzeme tipi
    if 'Fiber' in urun['urun_adi']:
        malzeme_kodu = 'FO-SM-9'
    elif 'Data' in urun['urun_adi'] or 'CAT' in urun['urun_adi']:
        malzeme_kodu = 'CU-ETP'
    else:
        malzeme_kodu = random.choice(['CU-ETP', 'AL-1350'])
    
    malzeme = malzeme_tipleri[malzeme_kodu]
    
    # Operatör seçimi (hat tipine uygunluk)
    uygun_operatorler = {
        k: v for k, v in operator_havuzu.items() 
        if hat['hat_tipi'] in v['uzmanlik_alani'] or 
        (hat['hat_tipi'] == 'Kombine' and any(tip in v['uzmanlik_alani'] for tip in ['Mikro', 'Standart']))
    }
    if not uygun_operatorler:
        uygun_operatorler = operator_havuzu
    
    operator_kodu = random.choice(list(uygun_operatorler.keys()))
    operator = uygun_operatorler[operator_kodu]
    
    # İzolasyon malzemesi
    izolasyon = urun['izolasyon_malzemesi']
    
    # Üretim miktarı hesaplama
    hedef_bobin_sayisi = random.randint(8, 60)
    standart_metraj = urun['standart_bobin_metraj']
    hedef_toplam_metraj = hedef_bobin_sayisi * standart_metraj
    
    # Fire oranı hesaplama
    fire_alt = malzeme['fire_toleransi_alt']
    fire_ust = malzeme['fire_toleransi_ust']
    
    # Operatör yetkinliği fire oranını etkiler
    operator_fire_etkisi = (5 - operator['yetkinlik_seviyesi']) * 0.004
    # Hat teknolojisi fire oranını etkiler
    hat_fire_etkisi = 0.01 if hat['teknoloji_seviyesi'] == 'Endüstri 4.0' else 0.03
    # Ürün zorluk derecesi fire oranını etkiler (YENİ)
    zorluk_fire_etkisi = urun['uretim_zorluk_puani'] * 0.002
    
    fire_orani = np.random.uniform(fire_alt, fire_ust) + operator_fire_etkisi + hat_fire_etkisi + zorluk_fire_etkisi
    fire_orani = max(0.001, min(fire_orani, 0.12))
    
    fire_bobin_adedi = int(hedef_bobin_sayisi * fire_orani)
    fire_toplam_metraj = int(hedef_toplam_metraj * fire_orani)
    kaliteli_bobin_adedi = hedef_bobin_sayisi - fire_bobin_adedi
    kaliteli_toplam_metraj = hedef_toplam_metraj - fire_toplam_metraj
    
    # Metrekare hesabı (kablo çapına göre yaklaşık yüzey alanı)
    m2_katsayilari = {
        'Mikro': 0.008, 'Standart': 0.025, 'Ağır': 0.065, 
        'Özel': 0.004, 'Süper Ağır': 0.095, 'Kombine': 0.020
    }
    m2_katsayi = m2_katsayilari.get(hat['hat_tipi'], 0.015)
    toplam_m2 = kaliteli_toplam_metraj * m2_katsayi
    fire_m2 = fire_toplam_metraj * m2_katsayi
    
    # Süre hesaplamaları
    hazirlik_temel_sureler = {
        'Mikro': 0.25, 'Standart': 0.50, 'Ağır': 0.75, 
        'Özel': 1.50, 'Süper Ağır': 1.25, 'Kombine': 0.60
    }
    temel_hazirlik = hazirlik_temel_sureler.get(hat['hat_tipi'], 0.50)
    
    # Zorluk derecesi setup süresini etkiler (YENİ)
    zorluk_hazirlik_etkisi = urun['uretim_zorluk_puani'] * 0.05
    
    # Operatör etkisi ve rastgele değişkenlik
    hat_hazirlik_suresi = temel_hazirlik * (1.5 - operator['performans_katsayisi'] * 0.25) + zorluk_hazirlik_etkisi
    hat_hazirlik_suresi = max(0.08, hat_hazirlik_suresi + np.random.uniform(-0.15, 0.25))
    
    # İşlem hızı (hat kapasitesi + ürün zorluğu)
    islem_hizi = hat['max_islem_hizi'] - (urun['uretim_zorluk_puani'] * 2.5)
    islem_hizi = max(hat['min_islem_hizi'], islem_hizi + np.random.uniform(-4, 4))
    
    # Operatör performansı işlem hızını etkiler
    efektif_islem_hizi = islem_hizi * operator['performans_katsayisi']
    efektif_islem_hizi = min(efektif_islem_hizi, hat['max_islem_hizi'] * 1.12)
    
    # Üretim süresi (saat)
    toplam_operasyon_suresi = kaliteli_toplam_metraj / efektif_islem_hizi / 60
    
    # BAKIM KONTROLÜ (YENİ)
    bakim_nedeniyle_durus = bakim_kontrol(tarih, hat_kodu)
    
    # ARIZA KONTROLÜ (YENİ)
    ariza_nedeniyle_durus = ariza_kontrol(hat_kodu)
    
    # Plansız duruş (diğer nedenler)
    plansiz_durus_suresi = 0
    if np.random.random() < 0.04:  # %4 diğer duruş ihtimali
        plansiz_durus_suresi = np.random.uniform(0.10, 0.50)
    
    # OEE Hesaplama (bakım ve arıza etkisiyle)
    planli_calisma_suresi = 8  # 8 saat
    toplam_durus = bakim_nedeniyle_durus + ariza_nedeniyle_durus + plansiz_durus_suresi
    efektif_calisma_suresi = planli_calisma_suresi - toplam_durus
    efektif_calisma_suresi = max(0.5, efektif_calisma_suresi)  # En az 0.5 saat
    
    kullanim_orani = min(100, ((toplam_operasyon_suresi + hat_hazirlik_suresi) / efektif_calisma_suresi) * 100 * np.random.uniform(0.82, 1.18))
    performans_orani = min(100, (efektif_islem_hizi / hat['max_islem_hizi']) * 100 * np.random.uniform(0.85, 1.12))
    
    # Zorluk derecesi kaliteyi etkiler (YENİ)
    zorluk_kalite_etkisi = urun['uretim_zorluk_puani'] * 0.5
    kalite_orani = max(0, min(100, (1 - fire_orani) * 100 * np.random.uniform(0.94, 1.04) - zorluk_kalite_etkisi))
    
    oee_degeri = (kullanim_orani / 100) * (performans_orani / 100) * (kalite_orani / 100) * 100
    
    # Tarih formatları
    tarih_format = tarih.strftime('%d.%m.%Y')
    gun_isimleri = {
        0: 'PAZARTESİ', 1: 'SALI', 2: 'ÇARŞAMBA', 
        3: 'PERŞEMBE', 4: 'CUMA', 5: 'CUMARTESİ', 6: 'PAZAR'
    }
    
    # İş emri numarası (özel format)
    is_emri_no = f"WO-{tarih.year}{tarih.month:02d}-{random.randint(10000, 99999)}"
    
    # Bobin eşdeğer hesaplamaları
    planlanan_bobin_miktari = kaliteli_bobin_adedi * np.random.uniform(0.85, 1.15)
    fire_bobin_karsiligi = fire_bobin_adedi * np.random.uniform(0.85, 1.15)
    
    # Bakım ve arıza türlerini belirleme
    if bakim_nedeniyle_durus > 0:
        if tarih.weekday() == 0:  # Pazartesi
            bakim_aciklamasi = f"Planlı Bakım (Pazartesi Sabah Periyodu)"
        else:  # Cumartesi
            bakim_aciklamasi = f"Planlı Bakım (Cumartesi Gece Periyodu)"
    else:
        bakim_aciklamasi = ""
    
    if ariza_nedeniyle_durus > 0:
        ariza_aciklamasi = f"Anlık Arıza ({hat['hat_adi']} - {hat['bakim_kritikligi']} Risk)"
    else:
        ariza_aciklamasi = ""
    
    return {
        'İş Emri No': is_emri_no,
        'Operasyon Kodu': random.choice(list(operasyon_kodlari.keys())),
        'Tesis Kodu': tesis_kodu,
        'Tesis Adı': tesisler[tesis_kodu]['tesis_adi'],
        'Üretim Tarihi': tarih_format,
        'Gün Adı': gun_isimleri[tarih.weekday()],
        'Hat Kodu': hat_kodu,
        'Hat Tanımı': hat['hat_adi'],
        'Operasyon Tanımı': random.choice(list(operasyon_kodlari.values()))['operasyon_adi'],
        'Fire Miktar (Bobin)': round(fire_bobin_karsiligi, 1),
        'Üretim Miktarı (Bobin)': round(planlanan_bobin_miktari, 1),
        'Setup Süresi (Saat)': round(hat_hazirlik_suresi, 3),
        'Bakım Duruşu (Saat)': round(bakim_nedeniyle_durus, 3),  # Maliyet duruş karşılığı
        'Arıza Duruşu (Saat)': round(ariza_nedeniyle_durus, 3),
        'Plansız Duruş (Saat)': round(plansiz_durus_suresi, 3),
        'Operasyon Süresi (Saat)': round(toplam_operasyon_suresi, 3),
        'Operasyon Hızı (m/dk)': round(efektif_islem_hizi, 1),
        'Operatör Adı Soyadı': operator['ad_soyad'],
        'Operatör Sicil No': operator_kodu,
        'Üretim Yılı': tarih.year,
        'Üretim Ayı': tarih.month,
        'İzolasyon Cinsi': izolasyon,
        'Ürün Kodu': urun_kodu,
        'Ürün Açıklaması': urun['urun_adi'],
        'Üretim Zorluk Derecesi': urun['uretim_zorluk_puani'],  # YENİ SÜTUN
        'Malzeme Kodu': malzeme_kodu,
        'Malzeme Açıklaması': malzeme['malzeme_adi'],
        'Kaliteli Bobin Adedi': kaliteli_bobin_adedi,
        'Kaliteli Toplam Metraj': kaliteli_toplam_metraj,
        'Kaliteli Toplam m²': round(toplam_m2, 2),
        'Fire Bobin Adedi': fire_bobin_adedi, 
        'Fire Toplam Metraj': fire_toplam_metraj,
        'Fire Toplam m²': round(fire_m2, 2),
        'Günlük İş Emri Sayısı': 1,
        'Parti Sayısı': 1,
        'Hat Başına İş Emri': 1,
        'Hat Günlük İş Emri': 1,
        'Ortalama Çekim Miktarı': round(planlanan_bobin_miktari, 1),
        'Ortalama Setup Süresi (Saat)': round(hat_hazirlik_suresi, 1),
        'Hat Nominal Hızı (m/dk)': round(hat['max_islem_hizi'], 1),
        'Bakım Açıklaması': bakim_aciklamasi,  # YENİ SÜTUN
        'Arıza Açıklaması': ariza_aciklamasi,  # YENİ SÜTUN
        'Kullanım Oranı (%)': round(kullanim_orani, 2),
        'Performans Oranı (%)': round(performans_orani, 2),
        'Kalite Oranı (%)': round(kalite_orani, 2),
        'OEE Değeri (%)': round(oee_degeri, 2),
    }

# ============================================================
# 3. VERİ SETİNİ OLUŞTURMA
# ============================================================

print("🔄 Dramatik kablo üretim veri seti oluşturuluyor...")
print("⚙️  Bakım periyotları: Pazartesi Sabah + Cumartesi Gece")
print("🔧 Rastgele makine arızaları: %5-25 ihtimal (hat bazlı)")
print("📊 Ürün zorluk dereceleri: 1-9 arası")
print("=" * 70)

tum_is_emirleri = []
is_emri_sayaci = 0

tarih = baslangic_tarih
while tarih <= bitis_tarih:
    if is_gunu_mu(tarih):
        # Günde 18-42 arası iş emri
        gunluk_emir_sayisi = random.randint(18, 42)
        for _ in range(gunluk_emir_sayisi):
            is_emri_sayaci += 1
            emir = is_emri_olustur(tarih, is_emri_sayaci)
            tum_is_emirleri.append(emir)
    
    tarih += timedelta(days=1)

# DataFrame oluşturma
df = pd.DataFrame(tum_is_emirleri)

# Yeni özgün sütun sıralaması (zorluk derecesi ve açıklama sütunları eklendi)
yeni_sutun_sirasi = [
    'İş Emri No',
    'Operasyon Kodu', 
    'Tesis Kodu',
    'Tesis Adı',
    'Üretim Tarihi',
    'Gün Adı',
    'Hat Kodu',
    'Hat Tanımı',
    'Operasyon Tanımı',
    'Fire Miktar (Bobin)',
    'Üretim Miktarı (Bobin)',
    'Setup Süresi (Saat)',
    'Bakım Duruşu (Saat)',  # Planlı bakım - Pazartesi/Cumartesi
    'Arıza Duruşu (Saat)',  # Rastgele arızalar
    'Plansız Duruş (Saat)',
    'Operasyon Süresi (Saat)',
    'Operasyon Hızı (m/dk)',
    'Operatör Adı Soyadı',
    'Operatör Sicil No',
    'Üretim Yılı',
    'Üretim Ayı',
    'İzolasyon Cinsi',
    'Ürün Kodu',
    'Ürün Açıklaması',
    'Üretim Zorluk Derecesi',  # ⭐ YENİ - 1'den 9'a
    'Malzeme Kodu',
    'Malzeme Açıklaması',
    'Kaliteli Bobin Adedi',
    'Kaliteli Toplam Metraj',
    'Kaliteli Toplam m²',
    'Fire Bobin Adedi', 
    'Fire Toplam Metraj',
    'Fire Toplam m²',
    'Günlük İş Emri Sayısı',
    'Parti Sayısı',
    'Hat Başına İş Emri',
    'Hat Günlük İş Emri',
    'Ortalama Çekim Miktarı',
    'Ortalama Setup Süresi (Saat)',
    'Hat Nominal Hızı (m/dk)',
    'Bakım Açıklaması',  # ⭐ YENİ - Bakım detayı
    'Arıza Açıklaması',  # ⭐ YENİ - Arıza detayı
    'Kullanım Oranı (%)',
    'Performans Oranı (%)',
    'Kalite Oranı (%)',
    'OEE Değeri (%)',
]

df = df[yeni_sutun_sirasi]

# Tarihe göre sırala
df['Tarih_Siralama'] = pd.to_datetime(df['Üretim Tarihi'], format='%d.%m.%Y')
df = df.sort_values(['Tarih_Siralama', 'İş Emri No']).reset_index(drop=True)
df = df.drop('Tarih_Siralama', axis=1)

# Son kontroller ve temizlik
df['İş Emri No'] = [f"WO-{datetime.strptime(t, '%d.%m.%Y').year}{datetime.strptime(t, '%d.%m.%Y').month:02d}-{10000+i}" 
                    for i, t in enumerate(df['Üretim Tarihi'])]

# ============================================================
# 4. VERİ SETİNİ BELİRTİLEN KONUMA KAYDET
# ============================================================

csv_dosyasi = os.path.join(kayit_konumu, 'kablo_uretim_veriseti_v3.csv')
xlsx_dosyasi = os.path.join(kayit_konumu, 'kablo_uretim_veriseti_v3.xlsx')
bilgi_dosyasi = os.path.join(kayit_konumu, 'veri_seti_aciklama_v3.txt')
kod_sozlugu_dosyasi = os.path.join(kayit_konumu, 'kod_sozlugu_v3.txt')
senaryo_dosyasi = os.path.join(kayit_konumu, 'uretim_senaryosu.txt')

df.to_csv(csv_dosyasi, index=False, sep=';')
df.to_excel(xlsx_dosyasi, index=False)

# Detaylı senaryo dosyası (YENİ)
with open(senaryo_dosyasi, 'w', encoding='utf-8') as f:
    f.write("=" * 70 + "\n")
    f.write("KABLO ÜRETİM TESİSİ - DRAMATİK SENARYO\n")
    f.write("=" * 70 + "\n\n")
    f.write("BAKIM PERİYOTLARI:\n")
    f.write("─" * 40 + "\n")
    f.write("📅 Pazartesi Sabah (06:00-08:00): 2-3 saat planlı bakım\n")
    f.write("📅 Cumartesi Gece (22:00-00:00): 2.5-4 saat planlı bakım\n")
    f.write("Etki: Bakım Duruşu (Saat) sütununa yansır\n\n")
    
    f.write("RASTGELE ARIZALAR:\n")
    f.write("─" * 40 + "\n")
    for hat_kodu, hat in uretim_hatlari.items():
        f.write(f"🔧 {hat_kodu} - {hat['hat_adi']}:\n")
        f.write(f"   Arıza İhtimali: %{hat['ariza_egilimi']*100:.0f}\n")
        f.write(f"   Arıza Süresi: 1-3 saat (rastgele)\n")
        f.write(f"   Risk Seviyesi: {hat['bakim_kritikligi']}\n")
        f.write(f"   Teknoloji: {hat['teknoloji_seviyesi']}\n\n")
    
    f.write("ÜRÜN ZORLUK DERECELERİ:\n")
    f.write("─" * 40 + "\n")
    f.write("⭐ 1-2: Kolay (Standart PVC kablolar)\n")
    f.write("⭐ 3-4: Orta (XLPE başlangıç, Halogen-Free)\n")
    f.write("⭐ 5-6: Zor (Kalın kesitler, özel malzemeler)\n")
    f.write("⭐ 7-8: Çok Zor (Süper ağır, fiber optik)\n")
    f.write("⭐ 9: Uzman (Fiber optik multi-mode)\n\n")
    
    f.write("ZORLUK ETKİLERİ:\n")
    f.write("─" * 40 + "\n")
    f.write("• Fire oranını artırır (+0.002 her zorluk puanı)\n")
    f.write("• Setup süresini uzatır (+0.05 saat her zorluk puanı)\n")
    f.write("• Operasyon hızını düşürür (-2.5 m/dk her zorluk puanı)\n")
    f.write("• Kalite oranını düşürür (-0.5% her zorluk puanı)\n\n")
    
    f.write(f"TOPLAM SENARYO İSTATİSTİKLERİ:\n")
    f.write(f"─" * 40 + "\n")
    bakim_sayisi = (df['Bakım Duruşu (Saat)'] > 0).sum()
    ariza_sayisi = (df['Arıza Duruşu (Saat)'] > 0).sum()
    f.write(f"Bakım yapılan iş emri: {bakim_sayisi} (Toplam: {df['Bakım Duruşu (Saat)'].sum():.1f} saat)\n")
    f.write(f"Arıza yaşanan iş emri: {ariza_sayisi} (Toplam: {df['Arıza Duruşu (Saat)'].sum():.1f} saat)\n")

# Detaylı açıklama dosyası
with open(bilgi_dosyasi, 'w', encoding='utf-8') as f:
    f.write("=" * 70 + "\n")
    f.write("KABLO ÜRETİM TESİSİ - VERİ SETİ DOKÜMANTASYONU v3.0\n")
    f.write("DRAMATİK SENARYO EDİSYONU\n")
    f.write("=" * 70 + "\n\n")
    f.write(f"Oluşturulma Tarihi: {datetime.now().strftime('%d.%m.%Y %H:%M')}\n")
    f.write(f"Sürüm: 3.0 - Bakım + Arıza + Zorluk Derecesi\n\n")
    f.write(f"TOPLAM İSTATİSTİKLER:\n")
    f.write(f"{'─' * 40}\n")
    f.write(f"Toplam İş Emri: {len(df):,}\n")
    f.write(f"Tarih Aralığı: {df['Üretim Tarihi'].min()} - {df['Üretim Tarihi'].max()}\n")
    f.write(f"Çalışma Günü: {df['Üretim Tarihi'].nunique()}\n")
    f.write(f"Tesis Sayısı: {df['Tesis Kodu'].nunique()}\n")
    f.write(f"Hat Sayısı: {df['Hat Kodu'].nunique()}\n")
    f.write(f"Operatör: {df['Operatör Sicil No'].nunique()}\n")
    f.write(f"Ürün Çeşidi: {df['Ürün Kodu'].nunique()}\n")
    f.write(f"Sütun Sayısı: {len(df.columns)}\n\n")
    
    f.write(f"ÜRETİM RAKAMLARI:\n")
    f.write(f"{'─' * 40}\n")
    f.write(f"Toplam Bobin: {df['Kaliteli Bobin Adedi'].sum():,}\n")
    f.write(f"Toplam Fire: {df['Fire Bobin Adedi'].sum():,}\n")
    f.write(f"Fire Oranı: %{((df['Fire Bobin Adedi'].sum() / (df['Kaliteli Bobin Adedi'].sum() + df['Fire Bobin Adedi'].sum())) * 100):.2f}\n")
    f.write(f"Toplam Metraj: {df['Kaliteli Toplam Metraj'].sum():,} m\n")
    f.write(f"Ortalama OEE: %{df['OEE Değeri (%)'].mean():.2f}\n\n")
    
    f.write(f"DRAMATİK EKLEMELER:\n")
    f.write(f"{'─' * 40}\n")
    f.write(f"Bakım Duruşu (Toplam): {df['Bakım Duruşu (Saat)'].sum():.1f} saat\n")
    f.write(f"Arıza Duruşu (Toplam): {df['Arıza Duruşu (Saat)'].sum():.1f} saat\n")
    f.write(f"Ortalama Zorluk: {df['Üretim Zorluk Derecesi'].mean():.2f}\n")

print("\n" + "=" * 70)
print("✅ DRAMATİK KABLO ÜRETİM VERİ SETİ HAZIR!")
print("=" * 70)
print(f"📁 Konum: {kayit_konumu}")
print(f"📄 CSV: kablo_uretim_veriseti_v3.csv")
print(f"📊 Excel: kablo_uretim_veriseti_v3.xlsx")
print(f"📝 Açıklama: veri_seti_aciklama_v3.txt")
print(f"📋 Kod Sözlüğü: kod_sozlugu_v3.txt")
print(f"🎭 Senaryo: uretim_senaryosu.txt")
print("=" * 70)
print(f"📊 TOPLAM: {len(df):,} iş emri, {len(df.columns)} sütun")
print(f"📅 Tarih: {df['Üretim Tarihi'].min()} - {df['Üretim Tarihi'].max()}")
print(f"🏭 Tesis: {df['Tesis Kodu'].nunique()} | Hat: {df['Hat Kodu'].nunique()}")
print(f"👷 Operatör: {df['Operatör Sicil No'].nunique()}")
print(f"📦 Ürün: {df['Ürün Kodu'].nunique()} | ⭐ Ort. Zorluk: {df['Üretim Zorluk Derecesi'].mean():.1f}")
print(f"🔧 Bakım: {df['Bakım Duruşu (Saat)'].sum():.0f} saat")
print(f"⚠️  Arıza: {df['Arıza Duruşu (Saat)'].sum():.0f} saat")
print(f"📈 OEE: %{df['OEE Değeri (%)'].mean():.1f}")

print("\n🎭 DRAMATİK SENARYO ÖZETİ:")
print("─" * 40)
print("📅 Her Pazartesi sabah: 2-3 saat bakım")
print("📅 Her Cumartesi gece: 2.5-4 saat bakım")
print("🔧 Rastgele arızalar: 1-3 saat (%5-25 ihtimal)")
print("⭐ Zorluk derecesi: Fire, setup, hız, kalite etkiler")
print("👴 Eski hatlar daha sık arıza yapar")
print("💪 Tecrübeli operatörler durumu kurtarır!")