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
    try:
        data_path = find_project_file("kablo_uretim_veriseti_v4.csv")
        df = pd.read_csv(data_path, sep=';')
        df['Tarih_DT'] = pd.to_datetime(df['Üretim Tarihi'], format='%d.%m.%Y')
        return df
    except FileNotFoundError:
        return None

@st.cache_data
def load_plan_data():
    try:
        plan_path = find_project_file("is_emri_plani.csv")
        plan = pd.read_csv(plan_path, sep=';', encoding='cp1254')
        plan['Tarih_DT'] = pd.to_datetime(plan['Plan_Tarihi'], format='%d.%m.%Y')
        return plan
    except FileNotFoundError:
        return None

try:
    model, feature_cols = load_model()
    ham_veri = load_historical_data()
    plan_veri = load_plan_data()
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
# AKILLI TAHMİN (PLAN ENTEGRASYONU)
# ============================================================
def akilli_tahmin(model_tahmini, plan_bobin):
    """Model tahminini plan bilgisiyle birleştir"""
    if plan_bobin is None or plan_bobin == 0:
        return int(model_tahmini), "🤖 Sadece model", "normal"
    
    alt_sinir = plan_bobin * 0.75
    ust_sinir = plan_bobin * 1.25
    duzeltilmis = max(alt_sinir, min(model_tahmini, ust_sinir))
    
    if duzeltilmis == model_tahmini:
        return int(duzeltilmis), "✅ Plan bandında", "normal"
    elif duzeltilmis > model_tahmini:
        return int(duzeltilmis), f"⬆️ Plana çekildi (+{int(duzeltilmis-model_tahmini)})", "off"
    else:
        return int(duzeltilmis), f"⬇️ Plana çekildi ({int(duzeltilmis-model_tahmini)})", "inverse"

def get_gun_plan_detay(tarih):
    """Seçili tarih için plan detayı"""
    if plan_veri is None:
        return {'var': False}
    
    gun_plan = plan_veri[plan_veri['Tarih_DT'] == tarih]
    if len(gun_plan) == 0:
        return {'var': False}
    
    return {
        'var': True,
        'is_emri': len(gun_plan),
        'bobin': gun_plan['Planlanan_Bobin'].sum(),
        'metraj': gun_plan['Planlanan_Metraj'].sum(),
        'zorluk': gun_plan['Zorluk_Derecesi'].mean(),
        'hat': gun_plan['Hat_Kodu'].nunique(),
        'operator': gun_plan['Operator_Kodu'].nunique(),
        'bakim': gun_plan['Bakim_Gunu'].iloc[0],
        'urunler': gun_plan[['Urun_Kodu', 'Urun_Adi', 'Planlanan_Bobin', 'Zorluk_Derecesi', 'Operator_Kodu']].copy()
    }

# ============================================================
# HAT & OPERATÖR ANALİZİ
# ============================================================
def get_hat_operatör_analizi(ham_veri, secili_tarih=None):
    if ham_veri is None:
        return None, None, None
    
    if secili_tarih:
        df_filtre = ham_veri[ham_veri['Tarih_DT'] == secili_tarih]
    else:
        df_filtre = ham_veri
    
    hat_ozet = df_filtre.groupby('Hat Kodu').agg(
        Hat_Adi=('Hat Tanımı', 'first'),
        Toplam_Bobin=('Kaliteli Bobin Adedi', 'sum'),
        Ortalama_OEE=('OEE Değeri (%)', 'mean'),
        Toplam_Arıza=('Arıza Duruşu (Saat)', 'sum'),
        Toplam_Bakım=('Bakım Duruşu (Saat)', 'sum'),
        Is_Emri_Sayisi=('İş Emri No', 'count')
    ).round(1)
    
    op_ozet = df_filtre.groupby('Operatör Adı Soyadı').agg(
        Toplam_Bobin=('Kaliteli Bobin Adedi', 'sum'),
        Ortalama_OEE=('OEE Değeri (%)', 'mean'),
        Is_Emri_Sayisi=('İş Emri No', 'count'),
        Fire_Adet=('Fire Bobin Adedi', 'sum')
    ).round(1)
    op_ozet['Fire_Orani'] = (op_ozet['Fire_Adet'] / (op_ozet['Toplam_Bobin'] + op_ozet['Fire_Adet']) * 100).round(1)
    
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
    
    # Plan durumu
    if plan_veri is not None:
        st.success("📋 Plan verisi yüklendi")
    else:
        st.warning("📋 Plan verisi bulunamadı")
    
    st.subheader("📊 Son 7 Gün")
    son_7 = gecmis_df['Toplam_Bobin'].tail(7)
    st.metric("Ortalama Üretim", f"{son_7.mean():.0f} bobin")
    st.metric("Min / Max", f"{son_7.min():.0f} / {son_7.max():.0f}")
    
    st.markdown("---")
    st.caption("v3.0 | XGBoost + Plan | FactoryFlow-AI")

# ============================================================
# ANA SAYFA
# ============================================================
st.title("🏭 Kablo Üretim Tesisi - AI Destekli Üretim Tahmini")
st.markdown("---")

tahmin_tarihi_dt = pd.to_datetime(tahmin_tarihi)

# Model ham tahmin
X_pred = create_prediction_features(tahmin_tarihi_dt, gecmis_df)
ham_tahmin = model.predict(X_pred)[0]

# Plan detayı
plan_detay = get_gun_plan_detay(tahmin_tarihi_dt)

# Akıllı tahmin
plan_bobin = plan_detay.get('bobin', None)
final_tahmin, duzeltme_msj, delta_color = akilli_tahmin(ham_tahmin, plan_bobin)

# Gerçek veri
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

# ============================================================
# ÜST METRİKLER (5'li)
# ============================================================
col1, col2, col3, col4, col5 = st.columns(5)

with col1:
    st.metric("🤖 Ham Tahmin", f"{int(ham_tahmin):,} bobin")

with col2:
    delta_str = f"Plan: {plan_detay['bobin']:,}" if plan_detay['var'] else "Plan yok"
    st.metric("🎯 Final Tahmin", f"{final_tahmin:,} bobin", delta=delta_str)

with col3:
    if plan_detay['var']:
        st.metric("📋 Plan Bobin", f"{plan_detay['bobin']:,} bobin",
                 delta=f"{plan_detay['is_emri']} iş emri")
    else:
        st.metric("📋 Plan", "Yok")

with col4:
    st.metric(duzeltme_msj.split()[0], duzeltme_msj.split(' ', 1)[1] if ' ' in duzeltme_msj else duzeltme_msj,
             delta_color=delta_color)

with col5:
    gun_adi_tr = ['Pzt', 'Sal', 'Çar', 'Per', 'Cum', 'Cmt', 'Paz'][tahmin_tarihi_dt.dayofweek]
    bakim_var = "🔧 Bakım" if plan_detay.get('bakim', False) or tahmin_tarihi_dt.dayofweek in [0, 5] else "✅ Normal"
    st.metric(f"📅 {gun_adi_tr}", bakim_var)

st.markdown("---")

# ============================================================
# PLAN & ÜRETİM DETAYI
# ============================================================
if plan_detay['var'] or gercek_bobin is not None:
    st.subheader("📋 Seçili Tarih Detayı")
    
    tab1, tab2, tab3 = st.tabs(["📋 Plan Detayı", "🏭 Hat & Operatör", "📊 Karşılaştırma"])
    
    with tab1:
        if plan_detay['var']:
            col_p1, col_p2 = st.columns([2, 1])
            with col_p1:
                st.markdown(f"""
                | Bilgi | Değer |
                |-------|-------|
                | **Toplam İş Emri** | {plan_detay['is_emri']} adet |
                | **Planlanan Bobin** | {plan_detay['bobin']:,} |
                | **Planlanan Metraj** | {plan_detay['metraj']:,} m |
                | **Ortalama Zorluk** | {plan_detay['zorluk']:.1f} |
                | **Aktif Hat** | {plan_detay['hat']} |
                | **Aktif Operatör** | {plan_detay['operator']} |
                | **Bakım Günü** | {'🔧 Evet' if plan_detay['bakim'] else '✅ Hayır'} |
                """)
            with col_p2:
                st.markdown("**📦 Planlanan Ürünler:**")
                plan_urunler = plan_detay.get('urunler')
                if isinstance(plan_urunler, pd.DataFrame):
                    st.dataframe(
                        plan_urunler[['Urun_Kodu', 'Planlanan_Bobin', 'Zorluk_Derecesi']],
                        use_container_width=True,
                        hide_index=True,
                    )
                else:
                    st.info("Planlanan ürün detayı bulunamadı.")
        else:
            st.info("Bu tarih için plan bulunamadı.")
    
    with tab2:
        if ham_veri is not None:
            hat_ozet, op_ozet, tesis_ozet = get_hat_operatör_analizi(ham_veri, tahmin_tarihi_dt)
            
            if hat_ozet is not None and len(hat_ozet) > 0:
                st.dataframe(hat_ozet, use_container_width=True)
            if op_ozet is not None and len(op_ozet) > 0:
                st.dataframe(op_ozet, use_container_width=True)
        else:
            st.info("Üretim verisi bulunamadı.")
    
    with tab3:
        karsilastirma = []
        if plan_detay['var']:
            karsilastirma.append(('Planlanan', plan_detay['bobin'], '#f39c12'))
        if gercek_bobin is not None:
            karsilastirma.append(('Gerçekleşen', gercek_bobin, '#2ecc71'))
        karsilastirma.append(('Tahmin', final_tahmin, '#3498db'))
        
        if len(karsilastirma) >= 2:
            fig_k, ax_k = plt.subplots(figsize=(8, 5))
            kategoriler = [k[0] for k in karsilastirma]
            degerler = [k[1] for k in karsilastirma]
            renkler = [k[2] for k in karsilastirma]
            ax_k.bar(kategoriler, degerler, color=renkler, edgecolor='black')
            for i, v in enumerate(degerler):
                ax_k.text(i, v + 10, f'{v:,}', ha='center', fontweight='bold')
            ax_k.set_title('Plan vs Gerçek vs Tahmin', fontweight='bold')
            ax_k.set_ylabel('Bobin Adedi')
            st.pyplot(fig_k)
        else:
            st.info("Karşılaştırma için yeterli veri yok.")

st.markdown("---")

# ============================================================
# 7 GÜNLÜK TAHMİN
# ============================================================
col_left, col_right = st.columns([3, 2])

with col_left:
    st.subheader("📈 7 Günlük Tahmin (Plan Entegre)")
    
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
            ham = model.predict(X_future[feature_cols])[0]
            
            # O günün planı varsa düzelt
            gun_plan_bobin = None
            if plan_veri is not None:
                gp = plan_veri[plan_veri['Tarih_DT'] == gun]
                if len(gp) > 0:
                    gun_plan_bobin = gp['Planlanan_Bobin'].sum()
            
            final, _, _ = akilli_tahmin(ham, gun_plan_bobin)
            gelecek_tahminler.append(final)
            
            yeni = pd.DataFrame({'Tarih_DT': [gun], 'Toplam_Bobin': [final]})
            gecmis_kopya = pd.concat([gecmis_kopya, yeni], ignore_index=True)
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
    ax.plot(x_tahmin, gelecek_tahminler, '#2ecc71', marker='o', linewidth=2, label='Tahmin (Planlı)')
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
    gun_adi = ['Pzt', 'Sal', 'Çar', 'Per', 'Cum', 'Cmt', 'Paz'][gun.dayofweek]
    
    # Plan kontrol
    plan_bilgi = "Yok"
    if plan_veri is not None:
        gp = plan_veri[plan_veri['Tarih_DT'] == gun]
        if len(gp) > 0:
            plan_bilgi = f"📋 {gp['Planlanan_Bobin'].sum():,}"
    
    bakim = "🔧" if gun.dayofweek in [0, 5] else ""
    ariza_risk = "Yüksek" if gun.dayofweek in [0, 5] else ("Orta" if gun.dayofweek == 4 else "Düşük")
    
    tablo_data.append({
        'Tarih': gun.strftime('%d.%m.%Y'),
        'Gün': f"{gun_adi} {bakim}",
        'Tahmin': f"{gelecek_tahminler[i]:,}",
        'Plan': plan_bilgi,
        'Min-Max': f"{gelecek_tahminler[i]-200:,} - {gelecek_tahminler[i]+200:,}",
        'Bakım': '🔧 Var' if gun.dayofweek in [0, 5] else '✅ Yok',
        'Arıza Riski': ariza_risk
    })

st.dataframe(pd.DataFrame(tablo_data), use_container_width=True, hide_index=True)

st.markdown("---")

with st.expander("ℹ️ Model Bilgisi & Nasıl Çalışır?"):
    st.markdown("""
    ### 🎯 Nasıl Çalışır?
    
    1. **Model** geçmiş 14+ günlük üretim verisine bakar
    2. **Ham tahmin** üretir (MAE ~205 bobin)
    3. **Plan yüklüyse** tahmini planın ±%25 bandına çeker
    4. **Final tahmin** hem model hem plan bilgisini kullanır
    
    ### 📊 Model Detayları
    
    | Özellik | Değer |
    |---------|-------|
    | **Model** | XGBoost Regressor (Tuned) |
    | **Veri Seti** | 312 gün sentetik kablo üretim verisi |
    | **Feature Sayısı** | 15 (Takvim + Lag + Rolling) |
    | **Train MAE** | 195 bobin |
    | **Test MAE** | 205 bobin |
    | **Test MAPE** | %25.1 |
    
    ### ⚠️ Uyarı
    Bu demo **sentetik veri** ile eğitilmiştir.
    """)

st.markdown("---")
st.caption("🏭 FactoryFlow-AI | XGBoost + Plan Entegrasyon | © 2025")
