# app.py
import streamlit as st
import pandas as pd
import numpy as np
import datetime
import joblib
from ai_summary import generate_ai_summary  # Import fungsi ringkasan AI

# --- Konfigurasi Awal ---
ANALYSIS_DATE = pd.Timestamp('2025-01-01')

# --- Load Model & Preprocessor ---
try:
    preprocessor = joblib.load('preprocessor_pipeline_v2.pkl')
    model = joblib.load('voting_model_v3.pkl')
except FileNotFoundError:
    st.error("❗ File model atau preprocessor tidak ditemukan. Pastikan file '.pkl' berada di direktori yang sesuai.")
    st.stop()
except Exception as e:
    st.error(f"❗ Terjadi error saat memuat model atau preprocessor: {e}")
    st.stop()

# --- Konfigurasi Halaman ---
st.set_page_config(page_title="Startup Success Predictor", layout="wide")
st.title("🚀 Startup Success Predictor")
st.markdown("Masukkan detail startup untuk memprediksi kemungkinan **berhasil** (Acquired/IPO) atau **gagal** (Closed).")

# --- Formulir Input ---
col1, col2 = st.columns(2)

with col1:
    st.header("💰 Informasi Pendanaan")
    funding_total_usd = st.number_input("Total Pendanaan (USD)", min_value=0.0, value=500000.0, step=10000.0, format="%.2f")
    funding_rounds = st.number_input("Jumlah Putaran Pendanaan", min_value=0, value=1, step=1)

with col2:
    st.header("🏢 Informasi Perusahaan")
    default_founded_date = ANALYSIS_DATE.date() - datetime.timedelta(days=3*365)
    founded_at_date = st.date_input("Tanggal Didirikan", min_value=datetime.date(1900, 1, 1), max_value=datetime.date.today(), value=default_founded_date)
    country_code = st.text_input("Kode Negara (misal: USA, GBR, CAN, IND)")
    category_list_input = st.text_input("Kategori Utama (misal: Software, Biotechnology|Enterprise Software)")

# --- Tombol Prediksi ---
if st.button("🔮 Prediksi Kesuksesan", type="primary", use_container_width=True):
    if not country_code or not category_list_input:
        st.warning("⚠️ Mohon lengkapi semua kolom, terutama *Kode Negara* dan *Kategori*.")
    else:
        founded_at_dt = pd.to_datetime(founded_at_date, errors='coerce')
        is_founded_at_known = 1 if pd.notna(founded_at_dt) else 0

        startup_age_days = (ANALYSIS_DATE - founded_at_dt).days if is_founded_at_known else 0
        startup_age_years = startup_age_days // 365 if startup_age_days > 0 else 0

        primary_category = category_list_input.split('|')[0].strip() if pd.notna(category_list_input) and category_list_input != 'unknown' else 'unknown'

        input_data = pd.DataFrame([{
            "funding_total_usd_cleaned": float(funding_total_usd),
            "funding_rounds": int(funding_rounds),
            "country_code": str(country_code).upper(),
            "primary_category": str(primary_category),
            "startup_age_years": int(startup_age_years),
            "is_founded_at_known": int(is_founded_at_known)
        }])

        st.markdown("---")
        st.subheader("🧾 Data yang Dikirim ke Model:")
        st.dataframe(input_data)

        try:
            input_processed = preprocessor.transform(input_data)

            prediction_proba = model.predict_proba(input_processed)[0]
            prediction = model.predict(input_processed)[0]

            st.markdown("---")
            st.subheader("📈 Hasil Prediksi")

            prob_success = prediction_proba[1] * 100

            if prediction == 1:
                st.success("✅ Startup ini diprediksi akan **BERHASIL** (Acquired/IPO)! 🎉")
                st.metric(label="Kepercayaan Terhadap Keberhasilan", value=f"{prob_success:.2f}%")
            else:
                st.error("❌ Startup ini diprediksi akan **GAGAL** (Closed). 😔")
                st.metric(label="Kepercayaan Terhadap Keberhasilan", value=f"{prob_success:.2f}%")

            st.progress(prob_success / 100)
            st.caption("📌 Prediksi berdasarkan data yang diberikan dan model machine learning. Bukan merupakan saran keuangan.")

            # --- Ringkasan AI ---
            st.markdown("---")
            st.subheader("🧠 Ringkasan AI")
            with st.spinner("Menganalisis..."):
                summary = generate_ai_summary(input_data, prediction, prob_success)
                st.write(summary)

        except Exception as e:
            st.error(f"❗ Terjadi kesalahan saat melakukan prediksi: {e}")
            st.info("Pastikan format input sesuai. Contoh: Kode Negara dan Kategori sesuai dengan data pelatihan model.")
            st.info("Jika Anda memasukkan kategori atau negara yang tidak umum, hasil prediksi mungkin tidak optimal.")

# --- Footer ---
st.markdown("---")
st.markdown("📊 Model dilatih menggunakan data historis startup.")
st.markdown("Fitur utama: Jumlah Pendanaan, Putaran Pendanaan, Negara, Kategori, dan Usia Startup.")

# --- Insight Model ---
st.subheader("📌 Pentingnya Fitur (Model Insights)")
importance_data = {
    "Primary Category": 0.307703,
    "Funding Amount": 0.305134,
    "Startup Age": 0.173639,
    "Country": 0.118453,
    "Funding Rounds": 0.082799
}
fi_df = pd.DataFrame(list(importance_data.items()), columns=["Fitur", "Bobot"])
fi_df["Kontribusi (%)"] = fi_df["Bobot"] * 100
fi_df["Kontribusi (%)"] = fi_df["Kontribusi (%)"].map("{:.2f}%".format)
fi_df = fi_df.drop(columns=["Bobot"])

st.dataframe(fi_df, use_container_width=True)
st.caption("📖 *Semakin besar persentase, semakin besar pengaruh fitur terhadap prediksi model.*")
