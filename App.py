# ============================================================
# app.py - Aplikasi Prediksi Churn Pelanggan
# ============================================================

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import plotly.graph_objects as go

# ===== SET HALAMAN =====
st.set_page_config(
    page_title="Prediksi Churn Pelanggan",
    page_icon="📊",
    layout="wide"
)

# ===== LOAD MODEL =====
@st.cache_resource
def load_model():
    model = joblib.load('best_churn_model.pkl')
    scaler = joblib.load('scaler.pkl')
    features = joblib.load('feature_columns.pkl')
    return model, scaler, features

try:
    model, scaler, features = load_model()
    model_loaded = True
except:
    model_loaded = False
    st.error("⚠️ Model tidak ditemukan! Pastikan file .pkl ada di direktori yang sama.")

# ===== HEADER =====
st.title("📊 Prediksi Churn Pelanggan")
st.markdown("---")
st.markdown("""
Aplikasi ini digunakan untuk memprediksi apakah seorang pelanggan akan **churn** (berhenti berlangganan) atau tidak.
Silakan isi data pelanggan di bawah ini untuk mendapatkan prediksi.
""")

# ===== SIDEBAR =====
with st.sidebar:
    st.header("⚙️ Input Data Pelanggan")
    st.markdown("Isi semua field di bawah ini:")
    
    # Fitur-fitur yang digunakan
    lifetime_value = st.number_input(
        "💰 Lifetime Value",
        min_value=0.0,
        max_value=100000.0,
        value=5000.0,
        step=100.0,
        help="Nilai total pelanggan selama berlangganan"
    )
    
    total_spent = st.number_input(
        "💵 Total Pengeluaran",
        min_value=0.0,
        max_value=50000.0,
        value=2000.0,
        step=100.0,
        help="Total pengeluaran pelanggan"
    )
    
    avg_order_value = st.number_input(
        "🛒 Rata-rata Nilai Transaksi",
        min_value=0.0,
        max_value=10000.0,
        value=500.0,
        step=50.0,
        help="Rata-rata nilai per transaksi"
    )
    
    last_3_month_purchase_freq = st.number_input(
        "📦 Frekuensi Pembelian 3 Bulan Terakhir",
        min_value=0,
        max_value=100,
        value=5,
        step=1,
        help="Berapa kali pembelian dalam 3 bulan terakhir"
    )
    
    total_visits = st.number_input(
        "👁️ Total Kunjungan",
        min_value=0,
        max_value=1000,
        value=50,
        step=1,
        help="Total kunjungan pelanggan"
    )
    
    age = st.number_input(
        "🎂 Usia",
        min_value=18,
        max_value=100,
        value=30,
        step=1
    )
    
    avg_session_time = st.number_input(
        "⏱️ Rata-rata Waktu Sesi (menit)",
        min_value=0.0,
        max_value=120.0,
        value=15.0,
        step=0.5
    )
    
    pages_per_session = st.number_input(
        "📄 Rata-rata Halaman per Sesi",
        min_value=0.0,
        max_value=50.0,
        value=5.0,
        step=0.5
    )
    
    support_tickets = st.number_input(
        "🎫 Jumlah Tiket Dukungan",
        min_value=0,
        max_value=20,
        value=1,
        step=1
    )
    
    delivery_delay_days = st.number_input(
        "📦 Keterlambatan Pengiriman (hari)",
        min_value=0,
        max_value=30,
        value=0,
        step=1
    )
    
    satisfaction_score = st.slider(
        "⭐ Skor Kepuasan",
        min_value=1.0,
        max_value=5.0,
        value=4.0,
        step=0.5
    )
    
    nps_score = st.slider(
        "📊 NPS Score",
        min_value=0,
        max_value=10,
        value=7,
        step=1
    )
    
    email_open_rate = st.slider(
        "📧 Email Open Rate (%)",
        min_value=0.0,
        max_value=100.0,
        value=50.0,
        step=5.0
    )
    
    email_click_rate = st.slider(
        "🖱️ Email Click Rate (%)",
        min_value=0.0,
        max_value=100.0,
        value=20.0,
        step=5.0
    )
    
    marketing_spend_per_user = st.number_input(
        "📢 Biaya Pemasaran per User",
        min_value=0.0,
        max_value=1000.0,
        value=50.0,
        step=10.0
    )
    
    is_premium_user = st.selectbox(
        "👑 Status Premium",
        options=[0, 1],
        format_func=lambda x: "Ya" if x == 1 else "Tidak"
    )
    
    discount_used = st.selectbox(
        "🏷️ Penggunaan Diskon",
        options=[0, 1],
        format_func=lambda x: "Ya" if x == 1 else "Tidak"
    )
    
    refund_requested = st.selectbox(
        "🔄 Permintaan Refund",
        options=[0, 1],
        format_func=lambda x: "Ya" if x == 1 else "Tidak"
    )

# ===== MAIN CONTENT =====
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("📋 Data Pelanggan")
    
    # Buat dataframe dari input
    input_data = pd.DataFrame({
        'lifetime_value': [lifetime_value],
        'total_spent': [total_spent],
        'avg_order_value': [avg_order_value],
        'last_3_month_purchase_freq': [last_3_month_purchase_freq],
        'total_visits': [total_visits],
        'age': [age],
        'avg_session_time': [avg_session_time],
        'pages_per_session': [pages_per_session],
        'support_tickets': [support_tickets],
        'delivery_delay_days': [delivery_delay_days],
        'satisfaction_score': [satisfaction_score],
        'nps_score': [nps_score],
        'email_open_rate': [email_open_rate],
        'email_click_rate': [email_click_rate],
        'marketing_spend_per_user': [marketing_spend_per_user],
        'is_premium_user': [is_premium_user],
        'discount_used': [discount_used],
        'refund_requested': [refund_requested]
    })
    
    # Pastikan urutan fitur sesuai
    input_data = input_data[features]
    
    st.dataframe(input_data, use_container_width=True)

with col2:
    st.subheader("🎯 Hasil Prediksi")
    
    if st.button("🔮 Prediksi Churn", type="primary", use_container_width=True):
        if model_loaded:
            # Scaling
            input_scaled = scaler.transform(input_data)
            
            # Prediksi
            prediction = model.predict(input_scaled)[0]
            probability = model.predict_proba(input_scaled)[0]
            
            # Tampilkan hasil
            if prediction == 1:
                st.error(f"⚠️ **Churn**")
                st.warning(f"Probabilitas Churn: {probability[1]*100:.2f}%")
            else:
                st.success(f"✅ **Tidak Churn**")
                st.info(f"Probabilitas Churn: {probability[1]*100:.2f}%")
            
            # Progress bar
            st.progress(probability[1])
            
            # Meter
            st.metric(
                label="Probabilitas Churn",
                value=f"{probability[1]*100:.1f}%",
                delta="Rendah" if probability[1] < 0.5 else "Tinggi"
            )
            
            # Visualization
            fig = go.Figure(go.Indicator(
                mode = "gauge+number",
                value = probability[1]*100,
                title = {'text': "Probabilitas Churn"},
                domain = {'x': [0, 1], 'y': [0, 1]},
                gauge = {
                    'axis': {'range': [0, 100]},
                    'bar': {'color': "red" if probability[1] > 0.5 else "green"},
                    'steps': [
                        {'range': [0, 50], 'color': "lightgreen"},
                        {'range': [50, 100], 'color': "lightcoral"}
                    ],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': 50
                    }
                }
            ))
            fig.update_layout(height=250)
            st.plotly_chart(fig, use_container_width=True)
            
            # Feature contribution (jika model adalah Random Forest)
            if hasattr(model, 'feature_importances_'):
                st.subheader("📊 Kontribusi Fitur")
                importance_df = pd.DataFrame({
                    'Fitur': features,
                    'Importance': model.feature_importances_
                }).sort_values('Importance', ascending=False).head(10)
                
                fig2 = px.bar(
                    importance_df,
                    x='Importance',
                    y='Fitur',
                    orientation='h',
                    title='Top 10 Fitur Paling Berpengaruh',
                    color='Importance',
                    color_continuous_scale='Blues'
                )
                st.plotly_chart(fig2, use_container_width=True)
        else:
            st.error("❌ Model tidak ditemukan. Pastikan file .pkl sudah diupload.")

# ===== FOOTER =====
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: gray;">
    <p>© 2026 - Prediksi Churn Pelanggan | Dibuat dengan Streamlit ❤️</p>
    <p>Model terbaik: Random Forest dengan F1-Score 0.45</p>
</div>
""", unsafe_allow_html=True)