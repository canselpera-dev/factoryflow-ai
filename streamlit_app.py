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
SEARCH_DIRS = [
    BASE_DIR,
    BASE_DIR.parent,
    BASE_DIR / "prodtanly",
]

def find_project_file(file_name):
    for folder in SEARCH_DIRS:
        candidate = folder / file_name
        if candidate.exists():
            return candidate

    searched = "\n".join(f"- {folder / file_name}" for folder in SEARCH_DIRS)
    raise FileNotFoundError(f"{file_name} bulunamadı. Aranan yollar:\n{searched}")

# ============================================================
# SAYFA KONFİGÜRASYONU
# ============================================================
st.set_page_config(
    page_title="Kablo Üretim Tahmini | AI Forecasting",
    page_icon="🏭",
    layout="wide"
)

# ============================================================
# MODEL VE VERİ YÜKLE
# ============================================================
@st.cache_resource
def load_model():
    model_path = find_project_file("xgb_model.pkl")
    features_path = find_project_file("feature_list.pkl")
    
    model = joblib.load(model_path)
    features = joblib.load(features_path)
    return model, features

@st.cache_data
def load_historical_data():
    """V4 üretim verisini yükle"""
    try:
        data_path = find_project_file("kablo_uretim_veriseti_v4.csv")
        df = pd.read_csv(data_path, sep=';')
        df['Tarih_DT'] = pd.to_datetime(df['Üretim Tarihi'], format='%d.%m.%Y')
        return df
    except FileNotFoundError:
        return None

try:
    model, feature_cols = load_model()
    ham_veri = load_historical_data()
except Exception as exc:
    st.error(f"Yükleme hatası: {exc}")
    st.stop()

# ============================================================
# FEATURE ENGINEERING
# ============================================================
def create_prediction_features(tarih, gecmis_veri):
    df = pd.DataFrame({'Tarih_DT': [tarih]})
    
    df['Gun_No'] = df['Tarih_DT'].dt.dayofweek
    df['Ay'] = df['Tarih_DT'].dt.month
    df['Hafta'] = df['Tarih_DT'].dt.isocalendar().week.astype(int)
    df['Ay_Gunu'] = df['Tarih_DT'].dt.day
    df['Pazartesi'] = (df['Gun_No'] == 0).astype(int)
    df['Cumartesi'] = (df['Gun_No'] == 5).astype(int)
    
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
        for col in ['Lag_1', 'Lag_2', 'Lag_3', 'Lag_7', 'RM_3', 'RM_7', 'RM_14', 'Gecen_Hafta', 'Haftalik_Ort']:
            df[col] = 969
    
    return df[feature_cols]

def prepare_history_data(data):
    required_column = 'Toplam_Bobin'
    if required_column not in data.columns:
        possible_columns = ['Toplam Bobin', 'Kaliteli Bobin Adedi', 'Bobin', 'bobin', 'toplam_bobin']
        matched_column = next((col for col in possible_columns if col in data.columns), None)
        if matched_column is None:
            raise ValueError("CSV içinde 'Toplam_Bobin' kolonu bulunamadı.")
        data = data.rename(columns={matched_column: required_column})
    
    data = data.copy()
    data[required_column] = pd.to_numeric(data[required_column], errors='coerce')
    data = data.dropna(subset=[required_column])
    if len(data) < 14:
        raise ValueError("Tahmin için en az 14 günlük geçmiş üretim verisi gerekiyor.")
    return data.tail(30).reset_index(drop=True)

# ============================================================
# YARDIMCI FONKSİYONLAR (Hat & Operatör Analizi)
# ============================================================
def get_hat_operatör_analizi(ham_veri, secili_tarih=None):
    """Hat ve operatör bazlı üretim analizi"""
    if ham_veri is None:
        return None, None, None
    
    if secili_tarih:
        df_filtre = ham_veri[ham_veri['Tarih_DT'] == secili_tarih]
    else:
        df_filtre = ham_veri
    
    # Hat bazlı
    hat_ozet = df_filtre.groupby('Hat Kodu').agg(
        Hat_Adi=('Hat Tanımı', 'first'),
        Toplam_Bobin=('Kaliteli Bobin Adedi', 'sum'),
        Ortalama_OEE=('OEE Değeri (%)', 'mean'),
        Toplam_Arıza=('Arıza Duruşu (Saat)', 'sum'),
        Toplam_Bakım=('Bakım Duruşu (Saat)', 'sum'),
        Is_Emri_Sayisi=('İş Emri No', 'count')
    ).round(1)
    
    # Operatör bazlı
    op_ozet = df_filtre.groupby('Operatör Adı Soyadı').agg(
        Toplam_Bobin=('Kaliteli Bobin Adedi', 'sum'),
        Ortalama_OEE=('OEE Değeri (%)', 'mean'),
        Is_Emri_Sayisi=('İş Emri No', 'count'),
        Fire_Adet=('Fire Bobin Adedi', 'sum')
    ).round(1)
    op_ozet['Fire_Orani'] = (op_ozet['Fire_Adet'] / (op_ozet['Toplam_Bobin'] + op_ozet['Fire_Adet']) * 100).round(1)
    
    # Tesis özeti
    tesis_ozet = df_filtre.groupby('Tesis Adı').agg(
        Toplam_Bobin=('Kaliteli Bobin Adedi', 'sum'),
        Ortalama_OEE=('OEE Değeri (%)', 'mean'),
        Toplam_Arıza=('Arıza Duruşu (Saat)', 'sum'),
        Toplam_Bakım=('Bakım Duruşu (Saat)', 'sum')
    ).round(1)
    
    return hat_ozet, op_ozet, tesis_ozet

# ============================================================
# SIDEBAR
# ============================================================
with st.sidebar:
    st.header("⚙️ Kontrol Paneli")
    
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
    
    # Demo veri toggle
    use_demo = st.checkbox("Demo verisi kullan", value=True)
    
    st.subheader("📂 Geçmiş Üretim Verisi")
    if not use_demo:
        uploaded_file = st.file_uploader("Son 14 günlük veriyi yükle (CSV)", type="csv")
        if uploaded_file:
            try:
                gecmis_df = prepare_history_data(pd.read_csv(uploaded_file, sep=';'))
                st.success(f"✅ {len(gecmis_df)} gün yüklendi")
            except Exception as exc:
                st.error(f"Hata: {exc}")
                st.stop()
        else:
            st.warning("Lütfen CSV yükleyin veya demo modu açın")
            st.stop()
    else:
        np.random.seed(42)
        demo_tarihler = pd.date_range(end=pd.to_datetime(bugun), periods=30, freq='D')
        gecmis_df = pd.DataFrame({
            'Tarih_DT': demo_tarihler,
            'Toplam_Bobin': np.random.normal(969, 200, 30).astype(int)
        })
        gecmis_df = prepare_history_data(gecmis_df)
        st.info("Demo verisi aktif")
    
    st.markdown("---")
    
    # Quick stats
    st.subheader("📊 Son 7 Gün")
    son_7 = gecmis_df['Toplam_Bobin'].tail(7)
    st.metric("Ortalama Üretim", f"{son_7.mean():.0f} bobin")
    st.metric("Min", f"{son_7.min():.0f} | Max: {son_7.max():.0f}")
    
    st.markdown("---")
    st.caption("v2.0 | XGBoost | FactoryFlow-AI")

# ============================================================
# ANA SAYFA
# ============================================================
st.title("🏭 Kablo Üretim Tesisi - AI Destekli Üretim Tahmini")
st.markdown("---")

# ============================================================
# ÜST METRİKLER
# ============================================================
tahmin_tarihi_dt = pd.to_datetime(tahmin_tarihi)
X_pred = create_prediction_features(tahmin_tarihi_dt, gecmis_df)
tahmin_bobin = int(model.predict(X_pred)[0])

# Tarihe göre gerçek veri varsa onu göster
if ham_veri is not None:
    o_tarih_veri = ham_veri[ham_veri['Tarih_DT'] == tahmin_tarihi_dt]
    if len(o_tarih_veri) > 0:
        gercek_bobin = o_tarih_veri['Kaliteli Bobin Adedi'].sum()
        gercek_oee = o_tarih_veri['OEE Değeri (%)'].mean()
        gercek_fire = o_tarih_veri['Fire Bobin Adedi'].sum()
    else:
        gercek_bobin = None
else:
    gercek_bobin = None

col1, col2, col3, col4 = st.columns(4)

with col1:
    delta_val = f"{tahmin_bobin - 969:,}" if gercek_bobin is None else f"Gerçek: {gercek_bobin:,}"
    st.metric("📦 Tahmini Bobin", f"{tahmin_bobin:,}", delta=delta_val)

with col2:
    tahmin_oee = np.random.uniform(60, 85) if gercek_bobin is None else gercek_oee
    st.metric("⚡ OEE", f"%{tahmin_oee:.1f}", delta=f"{tahmin_oee - 70:.1f}%")

with col3:
    tahmin_fire = int(tahmin_bobin * np.random.uniform(0.04, 0.08)) if gercek_bobin is None else gercek_fire
    st.metric("🔥 Fire", f"{tahmin_fire:,} bobin", delta=f"%{tahmin_fire/tahmin_bobin*100:.1f}")

with col4:
    gun_adi_tr = ['Pzt', 'Sal', 'Çar', 'Per', 'Cum', 'Cmt', 'Paz']
    bakim_var = "🔧 Bakım" if tahmin_tarihi_dt.dayofweek in [0, 5] else "✅ Normal"
    st.metric("📅 " + gun_adi_tr[tahmin_tarihi_dt.dayofweek], bakim_var)

st.markdown("---")

# ============================================================
# HAT & OPERATÖR ANALİZİ
# ============================================================
st.subheader("🔧 Hat Bazlı Üretim & Operatör Performansı")

if ham_veri is not None:
    hat_ozet, op_ozet, tesis_ozet = get_hat_operatör_analizi(ham_veri, tahmin_tarihi_dt)
    
    tab1, tab2, tab3 = st.tabs(["🏭 Hat Bazlı", "👷 Operatör", "🏢 Tesis"])
    
    with tab1:
        if hat_ozet is not None and len(hat_ozet) > 0:
            col_h1, col_h2 = st.columns([3, 2])
            with col_h1:
                st.dataframe(hat_ozet, use_container_width=True)
            with col_h2:
                # Hat üretim bar chart
                fig_h, ax_h = plt.subplots(figsize=(6, 5))
                hat_ozet_sorted = hat_ozet.sort_values('Toplam_Bobin', ascending=True)
                hat_bobin = hat_ozet_sorted['Toplam_Bobin'].to_numpy(dtype=float)
                hat_labels = [str(label) for label in hat_ozet_sorted.index]
                bars = ax_h.barh(range(len(hat_bobin)), hat_bobin,
                                color='steelblue', edgecolor='black')
                ax_h.set_yticks(range(len(hat_bobin)))
                ax_h.set_yticklabels(hat_labels, fontsize=8)
                ax_h.set_xlabel('Bobin Adedi')
                ax_h.set_title('Hat Bazlı Üretim')
                # En yüksek üretimi vurgula
                if len(hat_ozet_sorted) > 0:
                    bars[-1].set_color('#2ecc71')
                st.pyplot(fig_h)
        else:
            st.info("Seçili tarih için hat verisi bulunamadı. Geçmiş veriden ortalama gösteriliyor.")
            hat_ozet_ort, _, _ = get_hat_operatör_analizi(ham_veri)
            if hat_ozet_ort is not None:
                st.dataframe(hat_ozet_ort, use_container_width=True)
    
    with tab2:
        if op_ozet is not None and len(op_ozet) > 0:
            col_o1, col_o2 = st.columns([3, 2])
            with col_o1:
                st.dataframe(op_ozet, use_container_width=True)
            with col_o2:
                # Operatör performans scatter
                fig_o, ax_o = plt.subplots(figsize=(6, 5))
                op_bobin = op_ozet['Toplam_Bobin'].to_numpy(dtype=float)
                op_oee = op_ozet['Ortalama_OEE'].to_numpy(dtype=float)
                op_size = op_ozet['Is_Emri_Sayisi'].to_numpy(dtype=float) * 20
                op_fire = op_ozet['Fire_Orani'].to_numpy(dtype=float)
                scatter = ax_o.scatter(op_bobin, op_oee,
                                      s=op_size, c=op_fire,
                                      cmap='RdYlGn_r', alpha=0.7, edgecolor='black')
                for idx, row in op_ozet.iterrows():
                    operator_name = str(idx)
                    ax_o.annotate(operator_name.split()[-1][:4], (row['Toplam_Bobin'], row['Ortalama_OEE']),
                                 fontsize=7, ha='center')
                ax_o.set_xlabel('Toplam Bobin')
                ax_o.set_ylabel('OEE (%)')
                ax_o.set_title('Operatör Performansı (Balon: İş Emri, Renk: Fire)')
                plt.colorbar(scatter, ax=ax_o, label='Fire Oranı (%)')
                st.pyplot(fig_o)
        else:
            st.info("Seçili tarih için operatör verisi bulunamadı.")
    
    with tab3:
        if tesis_ozet is not None and len(tesis_ozet) > 0:
            col_t1, col_t2 = st.columns(2)
            with col_t1:
                st.dataframe(tesis_ozet, use_container_width=True)
            with col_t2:
                # Tesis arıza/bakım karşılaştırma
                fig_t, ax_t = plt.subplots(figsize=(6, 5))
                x_t = np.arange(len(tesis_ozet))
                w_t = 0.35
                tesis_ariza = tesis_ozet['Toplam_Arıza'].to_numpy(dtype=float)
                tesis_bakim = tesis_ozet['Toplam_Bakım'].to_numpy(dtype=float)
                tesis_labels = [str(label) for label in tesis_ozet.index]
                ax_t.bar(x_t - w_t/2, tesis_ariza, w_t, label='Arıza', color='#e74c3c')
                ax_t.bar(x_t + w_t/2, tesis_bakim, w_t, label='Bakım', color='#f39c12')
                ax_t.set_xticks(x_t)
                ax_t.set_xticklabels(tesis_labels, fontsize=8)
                ax_t.set_ylabel('Saat')
                ax_t.set_title('Tesis Arıza & Bakım Süreleri')
                ax_t.legend()
                st.pyplot(fig_t)
        else:
            st.info("Tesis verisi bulunamadı.")
else:
    st.warning("⚠️  V4 üretim verisi bulunamadı. Hat/operatör analizi için kablo_uretim_veriseti_v4.csv gerekiyor.")

st.markdown("---")

# ============================================================
# 7 GÜNLÜK TAHMİN
# ============================================================
col_left, col_right = st.columns([3, 2])

with col_left:
    st.subheader("📈 7 Günlük Tahmin")
    
    gelecek_tahminler = []
    gecmis_kopya = gecmis_df.copy()
    
    for i in range(7):
        gun = tahmin_tarihi_dt + timedelta(days=i)
        if len(gecmis_kopya) >= 14:
            son_degerler = gecmis_kopya['Toplam_Bobin'].to_numpy(dtype=float)
            X_future = pd.DataFrame({
                'Gun_No': [gun.dayofweek], 'Ay': [gun.month],
                'Hafta': [gun.isocalendar().week], 'Ay_Gunu': [gun.day],
                'Pazartesi': [1 if gun.dayofweek == 0 else 0],
                'Cumartesi': [1 if gun.dayofweek == 5 else 0],
                'Lag_1': [son_degerler[-1]], 'Lag_2': [son_degerler[-2]],
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
            yeni_satir = pd.DataFrame({'Tarih_DT': [gun], 'Toplam_Bobin': [int(pred)]})
            gecmis_kopya = pd.concat([gecmis_kopya, yeni_satir], ignore_index=True)
        else:
            gelecek_tahminler.append(969)
    
    fig, ax = plt.subplots(figsize=(10, 5))
    gunler = [(tahmin_tarihi_dt + timedelta(days=i)).strftime('%d.%m') for i in range(7)]
    gecmis_7 = gecmis_df['Toplam_Bobin'].to_numpy(dtype=float)[-7:]
    gecmis_gunler = [(bugun - timedelta(days=6-i)).strftime('%d.%m') for i in range(7)]
    x_labels = gecmis_gunler + gunler
    x_gecmis = np.arange(7)
    x_tahmin = np.arange(7, 14)
    
    ax.plot(x_gecmis, gecmis_7, 'steelblue', marker='o', linewidth=2, label='Geçmiş')
    ax.plot(x_tahmin, gelecek_tahminler, '#2ecc71', marker='o', linewidth=2, label='Tahmin')
    ax.axvline(x=6.5, color='red', linestyle='--', alpha=0.5)
    ax.fill_between(x_tahmin, [t-150 for t in gelecek_tahminler],
                    [t+150 for t in gelecek_tahminler], alpha=0.15, color='green')
    ax.set_xticks(np.arange(14))
    ax.set_xticklabels(x_labels, rotation=45)
    ax.set_xlabel('Tarih')
    ax.set_ylabel('Bobin Adedi')
    ax.legend()
    ax.grid(True, alpha=0.3)
    st.pyplot(fig)

with col_right:
    st.subheader("⭐ Feature Importance")
    importance = model.feature_importances_
    top_features = pd.DataFrame({'Feature': feature_cols, 'Importance': importance}
                               ).sort_values('Importance', ascending=True).tail(10)
    
    fig2, ax2 = plt.subplots(figsize=(6, 5))
    colors = plt.cm.get_cmap('viridis')(np.linspace(0.2, 0.9, 10))
    top_importance = top_features['Importance'].to_numpy(dtype=float)
    top_feature_names = top_features['Feature'].astype(str).tolist()
    ax2.barh(range(len(top_importance)), top_importance, color=colors, edgecolor='black')
    ax2.set_yticks(range(len(top_importance)))
    ax2.set_yticklabels(top_feature_names, fontsize=8)
    ax2.set_xlabel('Önem Skoru')
    ax2.set_title('Top 10 Feature Importance')
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
    ariza_risk = "Yüksek" if gun.dayofweek in [0, 5] else ("Orta" if gun.dayofweek == 4 else "Düşük")
    
    tablo_data.append({
        'Tarih': gun.strftime('%d.%m.%Y'),
        'Gün': f"{gun_adi} {bakim}",
        'Tahmini Bobin': f"{gelecek_tahminler[i]:,}",
        'Min-Max': f"{gelecek_tahminler[i]-200:,} - {gelecek_tahminler[i]+200:,}",
        'Bakım': '🔧 Var' if gun.dayofweek in [0, 5] else '✅ Yok',
        'Arıza Riski': ariza_risk
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
st.caption("🏭 FactoryFlow-AI | XGBoost | © 2025 | Sentetik Veri Simülasyonu")
