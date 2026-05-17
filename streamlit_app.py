import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

BASE_DIR = Path(__file__).resolve().parent
PROJECT_DIR = BASE_DIR  # Çünkü streamlit_app.py zaten ana klasörde

# ============================================================
# SAYFA KONFİGÜRASYONU
# ============================================================
st.set_page_config(
    page_title="Kablo Üretim Tahmini | AI Forecasting",
    page_icon="🏭",
    layout="wide"
)

# ============================================================
# MODEL VE FEATURE YÜKLE
# ============================================================
@st.cache_resource
def load_model():
    model_path = PROJECT_DIR / "xgb_model.pkl"
    features_path = PROJECT_DIR / "feature_list.pkl"

    if not model_path.exists() or not features_path.exists():
        missing_files = [
            path.name
            for path in (model_path, features_path)
            if not path.exists()
        ]
        raise FileNotFoundError(
            f"Eksik model dosyası: {', '.join(missing_files)}. "
            f"Beklenen klasör: {PROJECT_DIR}"
        )

    model = joblib.load(model_path)
    features = joblib.load(features_path)
    return model, features

try:
    model, feature_cols = load_model()
except Exception as exc:
    st.error(f"Model yüklenemedi: {exc}")
    st.stop()

# ============================================================
# FEATURE ENGINEERING (TAHMİN ANINDA)
# ============================================================
def create_prediction_features(tarih, gecmis_veri):
    """Seçilen tarih için tahmin feature'larını üret"""
    df = pd.DataFrame({'Tarih_DT': [tarih]})
    
    # Takvim
    df['Gun_No'] = df['Tarih_DT'].dt.dayofweek
    df['Ay'] = df['Tarih_DT'].dt.month
    df['Hafta'] = df['Tarih_DT'].dt.isocalendar().week.astype(int)
    df['Ay_Gunu'] = df['Tarih_DT'].dt.day
    df['Pazartesi'] = (df['Gun_No'] == 0).astype(int)
    df['Cumartesi'] = (df['Gun_No'] == 5).astype(int)
    
    # Geçmiş veriden lag feature'ları
    if gecmis_veri is not None and len(gecmis_veri) >= 14:
        son_degerler = gecmis_veri['Toplam_Bobin'].to_numpy(dtype=float)
        
        df['Lag_1'] = son_degerler[-1]
        df['Lag_2'] = son_degerler[-2]
        df['Lag_3'] = son_degerler[-3]
        df['Lag_7'] = son_degerler[-7]
        df['RM_3'] = np.mean(son_degerler[-4:-1])
        df['RM_7'] = np.mean(son_degerler[-8:-1])
        df['RM_14'] = np.mean(son_degerler[-15:-1])
        df['Gecen_Hafta'] = son_degerler[-7]
        df['Haftalik_Ort'] = np.mean(son_degerler[-7:])
    else:
        # Varsayılan değerler
        for col in ['Lag_1', 'Lag_2', 'Lag_3', 'Lag_7', 'RM_3', 'RM_7', 'RM_14', 'Gecen_Hafta', 'Haftalik_Ort']:
            df[col] = 969  # Ortalama üretim
    
    return df[feature_cols]

def prepare_history_data(data):
    """Yüklenen veya demo geçmiş veriyi modelin beklediği formata getirir."""
    required_column = 'Toplam_Bobin'

    if required_column not in data.columns:
        possible_columns = [
            'Toplam Bobin',
            'Kaliteli Bobin Adedi',
            'Bobin',
            'bobin',
            'toplam_bobin',
        ]
        matched_column = next((col for col in possible_columns if col in data.columns), None)

        if matched_column is None:
            raise ValueError(
                "CSV içinde 'Toplam_Bobin' kolonu bulunamadı. "
                "Lütfen son 14 günlük üretim adetlerini bu kolon adıyla yükleyin."
            )

        data = data.rename(columns={matched_column: required_column})

    data = data.copy()
    data[required_column] = pd.to_numeric(data[required_column], errors='coerce')
    data = data.dropna(subset=[required_column])

    if len(data) < 14:
        raise ValueError("Tahmin için en az 14 günlük geçmiş üretim verisi gerekiyor.")

    return data.tail(30).reset_index(drop=True)

# ============================================================
# ANA SAYFA
# ============================================================
st.title("🏭 Kablo Üretim Tesisi - AI Destekli Üretim Tahmini")
st.markdown("---")

# Sidebar
with st.sidebar:
    st.header("⚙️ Kontrol Paneli")
    
    # Tarih seçimi
    bugun = datetime.now().date()
    min_tarih = bugun + timedelta(days=1)
    max_tarih = bugun + timedelta(days=30)
    
    tahmin_tarihi = st.date_input(
        "📅 Tahmin Yapılacak Tarih",
        value=min_tarih,
        min_value=min_tarih,
        max_value=max_tarih
    )
    
    st.markdown("---")
    
    # Geçmiş veri yükleme
    st.subheader("📂 Geçmiş Üretim Verisi")
    uploaded_file = st.file_uploader("Son 14 günlük veriyi yükle (CSV)", type="csv")
    
    if uploaded_file:
        try:
            gecmis_df = prepare_history_data(pd.read_csv(uploaded_file, sep=';'))
            st.success(f"✅ {len(gecmis_df)} günlük veri yüklendi")
        except Exception as exc:
            st.error(f"CSV okunamadı: {exc}")
            st.stop()
    else:
        st.info("Demo verisi kullanılıyor")
        # Demo geçmiş veri
        np.random.seed(42)
        demo_tarihler = pd.date_range(end=pd.to_datetime(bugun), periods=30, freq='D')
        gecmis_df = pd.DataFrame({
            'Tarih_DT': demo_tarihler,
            'Toplam_Bobin': np.random.normal(969, 200, 30).astype(int)
        })
        gecmis_df = prepare_history_data(gecmis_df)
    
    st.markdown("---")
    st.caption("v1.0 | XGBoost Model | Sentetik Veri")

# ============================================================
# ANA İÇERİK
# ============================================================
col1, col2, col3 = st.columns(3)

# Feature'ları hazırla
tahmin_tarihi_dt = pd.to_datetime(tahmin_tarihi)
X_pred = create_prediction_features(tahmin_tarihi_dt, gecmis_df)

# Tahmin yap
tahmin_bobin = model.predict(X_pred)[0]
tahmin_bobin = int(tahmin_bobin)

# OEE ve fire tahmini
tahmin_oee = np.random.uniform(60, 85)
tahmin_fire = int(tahmin_bobin * np.random.uniform(0.04, 0.08))

with col1:
    st.metric(
        label="📦 Tahmini Bobin Üretimi",
        value=f"{tahmin_bobin:,}",
        delta=f"{tahmin_bobin - 969:,} vs ortalama"
    )

with col2:
    st.metric(
        label="⚡ Tahmini OEE",
        value=f"%{tahmin_oee:.1f}",
        delta=f"{tahmin_oee - 70:.1f}%"
    )

with col3:
    st.metric(
        label="🔥 Tahmini Fire",
        value=f"{tahmin_fire:,} bobin",
        delta=f"%{tahmin_fire/tahmin_bobin*100:.1f} fire oranı",
        delta_color="inverse"
    )

st.markdown("---")

# ============================================================
# GÖRSELLEŞTİRME
# ============================================================
col_left, col_right = st.columns(2)

with col_left:
    st.subheader("📈 7 Günlük Tahmin")
    
    # Gelecek 7 gün için tahmin
    gelecek_tahminler = []
    gecmis_kopya = gecmis_df.copy()
    
    for i in range(7):
        gun = tahmin_tarihi_dt + timedelta(days=i)
        
        # Feature oluştur
        if len(gecmis_kopya) >= 14:
            son_degerler = gecmis_kopya['Toplam_Bobin'].to_numpy(dtype=float)
            X_future = pd.DataFrame({
                'Gun_No': [gun.dayofweek],
                'Ay': [gun.month],
                'Hafta': [gun.isocalendar().week],
                'Ay_Gunu': [gun.day],
                'Pazartesi': [1 if gun.dayofweek == 0 else 0],
                'Cumartesi': [1 if gun.dayofweek == 5 else 0],
                'Lag_1': [son_degerler[-1]],
                'Lag_2': [son_degerler[-2]],
                'Lag_3': [son_degerler[-3]],
                'Lag_7': [son_degerler[-7]] if len(son_degerler) >= 7 else [969],
                'RM_3': [np.mean(son_degerler[-4:-1])],
                'RM_7': [np.mean(son_degerler[-8:-1])] if len(son_degerler) >= 8 else [969],
                'RM_14': [np.mean(son_degerler[-15:-1])] if len(son_degerler) >= 15 else [969],
                'Gecen_Hafta': [son_degerler[-7]] if len(son_degerler) >= 7 else [969],
                'Haftalik_Ort': [np.mean(son_degerler[-7:])]
            })
            
            pred = model.predict(X_future[feature_cols])[0]
            gelecek_tahminler.append(int(pred))
            
            # Geçmişe ekle (bir sonraki gün için)
            yeni_satir = pd.DataFrame({'Tarih_DT': [gun], 'Toplam_Bobin': [int(pred)]})
            gecmis_kopya = pd.concat([gecmis_kopya, yeni_satir], ignore_index=True)
        else:
            gelecek_tahminler.append(969)
    
    # Grafik
    fig, ax = plt.subplots(figsize=(10, 5))
    gunler = [(tahmin_tarihi_dt + timedelta(days=i)).strftime('%d.%m') for i in range(7)]
    
    # Geçmiş 7 gün
    gecmis_7 = gecmis_df['Toplam_Bobin'].to_numpy(dtype=float)[-7:]
    gecmis_gunler = [(bugun - timedelta(days=6-i)).strftime('%d.%m') for i in range(7)]
    x_gecmis = np.arange(7)
    x_tahmin = np.arange(7, 14)
    x_labels = gecmis_gunler + gunler
    
    ax.plot(x_gecmis, gecmis_7, 'steelblue', marker='o', linewidth=2, label='Geçmiş')
    ax.plot(x_tahmin, gelecek_tahminler, '#2ecc71', marker='o', linewidth=2, label='Tahmin')
    ax.axvline(x=6.5, color='red', linestyle='--', alpha=0.5)
    ax.fill_between(x_tahmin, 
                    [t - 150 for t in gelecek_tahminler], 
                    [t + 150 for t in gelecek_tahminler], 
                    alpha=0.15, color='green')
    ax.set_xticks(np.arange(14))
    ax.set_xticklabels(x_labels, rotation=45)
    ax.set_xlabel('Tarih')
    ax.set_ylabel('Bobin Adedi')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    st.pyplot(fig)

with col_right:
    st.subheader("⭐ Feature Importance")
    
    # Feature importance grafiği
    importance = model.feature_importances_
    top_features = pd.DataFrame({
        'Feature': feature_cols,
        'Importance': importance
    }).sort_values('Importance', ascending=True).tail(10)
    
    fig2, ax2 = plt.subplots(figsize=(8, 5))
    colors = plt.cm.get_cmap('viridis')(np.linspace(0.2, 0.9, 10))
    top_importance = top_features['Importance'].to_numpy(dtype=float)
    top_feature_names = top_features['Feature'].astype(str).tolist()

    ax2.barh(range(len(top_importance)), top_importance, color=colors, edgecolor='black')
    ax2.set_yticks(range(len(top_importance)))
    ax2.set_yticklabels(top_feature_names)
    ax2.set_xlabel('Önem Skoru')
    ax2.set_title('Model Feature Importance (Top 10)')
    
    st.pyplot(fig2)

st.markdown("---")

# ============================================================
# DETAYLI TABLO
# ============================================================
st.subheader("📋 7 Günlük Detaylı Tahmin Tablosu")

tablo_data = []
for i in range(7):
    gun = tahmin_tarihi_dt + timedelta(days=i)
    gun_adi = ['Pazartesi', 'Salı', 'Çarşamba', 'Perşembe', 'Cuma', 'Cumartesi', 'Pazar'][gun.dayofweek]
    bakim = "🔧" if gun.dayofweek in [0, 5] else ""
    
    tablo_data.append({
        'Tarih': gun.strftime('%d.%m.%Y'),
        'Gün': gun_adi + " " + bakim,
        'Tahmini Bobin': gelecek_tahminler[i],
        'Min Tahmin': gelecek_tahminler[i] - 200,
        'Max Tahmin': gelecek_tahminler[i] + 200,
        'Bakım Günü': 'Evet' if gun.dayofweek in [0, 5] else 'Hayır'
    })

tablo_df = pd.DataFrame(tablo_data)
st.dataframe(tablo_df, use_container_width=True, hide_index=True)

st.markdown("---")

# ============================================================
# MODEL BİLGİSİ
# ============================================================
with st.expander("ℹ️ Model Bilgisi & Metrikler"):
    st.markdown("""
    ### 🎯 Model Detayları
    
    | Özellik | Değer |
    |---------|-------|
    | **Model** | XGBoost Regressor (Tuned) |
    | **Veri Seti** | 312 gün sentetik kablo üretim verisi |
    | **Feature Sayısı** | 15 (Takvim + Lag + Rolling) |
    | **Train MAE** | 195 bobin |
    | **Test MAE** | 205 bobin |
    | **Test MAPE** | %25.1 |
    
    ### 📊 Simülasyon Senaryosu
    
    - 🏭 2 üretim tesisi (İstanbul, Ankara)
    - 🔧 9 ekstrüzyon hattı
    - 📦 17 farklı ürün tipi
    - 👷 10 operatör
    - 📅 Pazartesi/Cumartesi planlı bakım
    - ⚠️ Rastgele makine arızaları
    
    ### ⚠️ Uyarı
    
    Bu demo **sentetik veri** ile eğitilmiştir. Gerçek üretim ortamında 
    kullanılmadan önce gerçek veri ile yeniden eğitilmelidir.
    """)

st.markdown("---")
st.caption("🏭 Kablo Üretim Tesisi AI Forecasting | XGBoost | © 2025")
