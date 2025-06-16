import streamlit as st
import pandas as pd
import joblib
import numpy as np
import os


st.set_page_config(
    page_title="Prediksi Status Kelulusan Mahasiswa",
    page_icon="🎓",
    layout="centered"
)


# Fungsi untuk menambahkan fitur turunan yang sama seperti saat pelatihan model
def inference_derived_features(input_df):
    # Konversi ke skala 4.0
    input_df['Ipk_semester1'] = (input_df['Curricular_units_1st_sem_grade'] / 20) * 4
    input_df['Ipk_semester2'] = (input_df['Curricular_units_2nd_sem_grade'] / 20) * 4

    # Proporsi SKS yang diluluskan semester 1 & 2
    # Menggunakan .replace(0, 1) untuk menghindari pembagian dengan nol jika ada
    input_df['proporsi_sks_1'] = input_df['Curricular_units_1st_sem_approved'] / input_df['Curricular_units_1st_sem_enrolled'].replace(0, 1)
    input_df['proporsi_sks_2'] = input_df['Curricular_units_2nd_sem_approved'] / input_df['Curricular_units_2nd_sem_enrolled'].replace(0, 1)

    # Perubahan IPK semester 1 ke 2
    input_df['index_ipk'] = input_df['Ipk_semester2'] - input_df['Ipk_semester1']

    # Kemajuan jumlah SKS lulus semester 2 dibanding semester 1
    input_df['kemajuan_sks'] = input_df['Curricular_units_2nd_sem_approved'] - input_df['Curricular_units_1st_sem_approved']

    # Skor status pembayaran: 0 = lancar, 1 = ada salah satu kendala, 2 = keduanya bermasalah
    # Asumsi Tuition_fees_up_to_date: 1 = Lancar, 0 = Tidak Lancar.
    # Jika 0 (Tidak Lancar) maka skor 1. Jika 1 (Lancar) maka skor 0.
    # Debtor: 1 = Ya (skor 1), 0 = Tidak (skor 0).
    input_df['status_pembayaran'] = input_df['Tuition_fees_up_to_date'].apply(lambda x: 0 if x == 1 else 1) + input_df['Debtor']

    return input_df

# --- Load Model dan Preprocessor ---
@st.cache_resource # Cache resource agar model tidak dimuat ulang setiap kali ada interaksi UI
def load_model_and_preprocessor():
    model_path = 'model/best_model.joblib'
    preprocessor_path = 'model/preprocessor.joblib'

    if not os.path.exists(model_path) or not os.path.exists(preprocessor_path):
        st.error(
            "File model atau preprocessor tidak ditemukan! "
            "Pastikan Anda telah menjalankan skrip ML asli untuk melatih dan menyimpan model, "
            "dan file-file tersebut berada di folder 'model/'."
        )
        st.stop() # Hentikan eksekusi aplikasi jika file tidak ditemukan

    try:
        model = joblib.load(model_path)
        # Preprocessor sebenarnya sudah termasuk dalam pipeline model,
        # tetapi jika diperlukan terpisah, bisa dimuat juga.
        # Untuk kasus ini, pipeline model sudah menangani preprocessing.
        return model
    except Exception as e:
        st.error(f"Gagal memuat model atau preprocessor. Error: {e}")
        st.stop()

model = load_model_and_preprocessor()


st.markdown(
    """
    <style>
    .main {
        background-color: #f0f2f6;
        padding: 20px;
        border-radius: 10px;
    }
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        border-radius: 8px;
        border: none;
        padding: 10px 20px;
        font-size: 16px;
        cursor: pointer;
        transition: background-color 0.3s ease;
    }
    .stButton>button:hover {
        background-color: #45a049;
    }
    .stTextInput>div>div>input {
        border-radius: 8px;
        border: 1px solid #ccc;
        padding: 8px;
    }
    .stSelectbox>div>div>div {
        border-radius: 8px;
        border: 1px solid #ccc;
        padding: 8px;
    }
    .stNumberInput>div>div>input {
        border-radius: 8px;
        border: 1px solid #ccc;
        padding: 8px;
    }
    .prediction-box {
        background-color: #e6f7ff;
        border-left: 5px solid #2196F3;
        padding: 15px;
        margin-top: 20px;
        border-radius: 8px;
    }
    .warning-box {
        background-color: #fff3cd; /* Warna latar belakang kuning muda */
        border-left: 5px solid #ffc107; /* Warna border kuning */
        padding: 15px;
        margin-top: 20px;
        border-radius: 8px;
    }
    .success-box {
        background-color: #03A791;
        border-left: 5px solid #28a745;
        padding: 15px;
        margin-top: 20px;
        border-radius: 8px;
    }
    h1, h2, h3 {
        color: #2c3e50;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.title("🎓 Prediksi Status Kelulusan Mahasiswa")
st.markdown("Aplikasi ini memprediksi apakah seorang mahasiswa cenderung **Lulus (Graduate)** atau **Keluar (Dropout)** berdasarkan data akademik dan demografi.")

# Formulir Input Data Mahasiswa
st.header("Masukkan Data Mahasiswa")

# Menggunakan st.columns untuk layout 2 kolom
col1, col2 = st.columns(2)

with col1:
    st.subheader("Data Akademik Semester 1")
    cu_1st_sem_enrolled = st.number_input(
        "Jumlah Unit Kurikuler yang Didaftar (Semester 1)",
        min_value=0, max_value=30, value=6, step=1,
        help="Jumlah mata kuliah yang didaftarkan di semester pertama."
    )
    cu_1st_sem_approved = st.number_input(
        "Jumlah Unit Kurikuler yang Disetujui (Semester 1)",
        min_value=0, max_value=30, value=4, step=1,
        help="Jumlah mata kuliah yang berhasil diselesaikan di semester pertama."
    )
    cu_1st_sem_grade = st.number_input(
        "Nilai Rata-rata Unit Kurikuler (Semester 1)",
        min_value=0.0, max_value=20.0, value=15.5, step=0.1, format="%.1f",
        help="Nilai rata-rata dari semua mata kuliah di semester pertama (skala 0-20)."
    )
    international = st.selectbox(
        "Apakah Mahasiswa Internasional?",
        options=["Tidak", "Ya"],
        index=0, # Default: Tidak
        format_func=lambda x: "Ya" if x == "Ya" else "Tidak",
        help="Pilih 'Ya' jika mahasiswa internasional, 'Tidak' jika mahasiswa lokal."
    )
    # Mapping 'Ya' ke 1, 'Tidak' ke 0
    international_val = 1 if international == "Ya" else 0

with col2:
    st.subheader("Data Akademik Semester 2")
    cu_2nd_sem_enrolled = st.number_input(
        "Jumlah Unit Kurikuler yang Didaftar (Semester 2)",
        min_value=0, max_value=30, value=5, step=1,
        help="Jumlah mata kuliah yang didaftarkan di semester kedua."
    )
    cu_2nd_sem_approved = st.number_input(
        "Jumlah Unit Kurikuler yang Disetujui (Semester 2)",
        min_value=0, max_value=30, value=3, step=1,
        help="Jumlah mata kuliah yang berhasil diselesaikan di semester kedua."
    )
    cu_2nd_sem_grade = st.number_input(
        "Nilai Rata-rata Unit Kurikuler (Semester 2)",
        min_value=0.0, max_value=20.0, value=14.0, step=0.1, format="%.1f",
        help="Nilai rata-rata dari semua mata kuliah di semester kedua (skala 0-20)."
    )
    mothers_occupation = st.number_input(
        "Pekerjaan Ibu (Kode Numerik)",
        min_value=0, max_value=150, value=9, step=1,
        help="Kode numerik untuk pekerjaan ibu (sesuai dataset asli). Contoh: 9 (Pekerja Jasa), 12 (Pekerja Terampil), dll."
    )

st.markdown("---") # Garis pemisah untuk keterbacaan

col3, col4 = st.columns(2)

with col3:
    st.subheader("Informasi Umum & Finansial")
    age_at_enrollment = st.number_input(
        "Usia Saat Pendaftaran",
        min_value=17, max_value=70, value=21, step=1,
        help="Usia mahasiswa saat pertama kali mendaftar."
    )
    tuition_fees_up_to_date = st.selectbox(
        "Pembayaran Uang Kuliah Tepat Waktu?",
        options=["Ya", "Tidak"],
        index=0, # Default: Ya
        format_func=lambda x: "Ya" if x == "Ya" else "Tidak",
        help="Status pembayaran uang kuliah hingga tanggal saat ini."
    )
    # Mapping 'Ya' ke 1, 'Tidak' ke 0
    tuition_fees_up_to_date_val = 1 if tuition_fees_up_to_date == "Ya" else 0

with col4:
    st.subheader("Status Lainnya")
    scholarship_holder = st.selectbox(
        "Penerima Beasiswa?",
        options=["Tidak", "Ya"],
        index=0, # Default: Tidak
        format_func=lambda x: "Ya" if x == "Ya" else "Tidak",
        help="Apakah mahasiswa saat ini menerima beasiswa?"
    )
    # Mapping 'Ya' ke 1, 'Tidak' ke 0
    scholarship_holder_val = 1 if scholarship_holder == "Ya" else 0

    debtor = st.selectbox(
        "Memiliki Hutang?",
        options=["Tidak", "Ya"],
        index=0, # Default: Tidak
        format_func=lambda x: "Ya" if x == "Ya" else "Tidak",
        help="Apakah mahasiswa memiliki hutang kepada institusi?"
    )
    # Mapping 'Ya' ke 1, 'Tidak' ke 0
    debtor_val = 1 if debtor == "Ya" else 0


# --- Tombol Prediksi ---
st.markdown("---")
if st.button("Prediksi Status"):
    # Buat dictionary input dari nilai-nilai UI
    input_data = {
        'Curricular_units_1st_sem_grade': cu_1st_sem_grade,
        'Curricular_units_2nd_sem_grade': cu_2nd_sem_grade,
        'Curricular_units_1st_sem_enrolled': cu_1st_sem_enrolled,
        'Curricular_units_1st_sem_approved': cu_1st_sem_approved,
        'Curricular_units_2nd_sem_enrolled': cu_2nd_sem_enrolled,
        'Curricular_units_2nd_sem_approved': cu_2nd_sem_approved,
        'Age_at_enrollment': age_at_enrollment,
        'Tuition_fees_up_to_date': tuition_fees_up_to_date_val,
        'Scholarship_holder': scholarship_holder_val,
        'Debtor': debtor_val,
        'International': international_val,
        'Mothers_occupation': mothers_occupation
    }

    # Konversi ke DataFrame
    input_df = pd.DataFrame([input_data])

    # Tambahkan fitur turunan
    processed_input_df = inference_derived_features(input_df)

    model_features_order = [
        'Curricular_units_1st_sem_enrolled', 'Curricular_units_1st_sem_approved',
        'Curricular_units_2nd_sem_enrolled', 'Curricular_units_2nd_sem_approved',
        'International', 'Mothers_occupation', 'Ipk_semester1', 'Ipk_semester2',
        'Age_at_enrollment', 'Tuition_fees_up_to_date', 'Scholarship_holder', 'Debtor',
        'proporsi_sks_1', 'proporsi_sks_2', 'index_ipk', 'kemajuan_sks', 'status_pembayaran'
    ]
    # Pastikan semua fitur ada di processed_input_df sebelum reindexing
    missing_features = [f for f in model_features_order if f not in processed_input_df.columns]
    if missing_features:
        st.error(f"Fitur yang dibutuhkan model tidak lengkap: {missing_features}. Harap laporkan masalah ini.")
    else:
        final_input_for_prediction = processed_input_df[model_features_order]

        # Lakukan prediksi
        prediction = model.predict(final_input_for_prediction)
        probabilities = model.predict_proba(final_input_for_prediction)[0]

        status = "Dropout" if prediction[0] == 1 else "Graduate" # 1 = Dropout, 0 = Graduate

        st.subheader("Hasil Prediksi")
        if status == "Dropout":
            st.markdown(
                f"<div class='warning-box'>Status Prediksi: <b>{status}</b></div>",
                unsafe_allow_html=True
            )
            st.warning("Perhatian! Mahasiswa ini memiliki risiko tinggi untuk Dropout. Intervensi mungkin diperlukan.")
        else:
            st.markdown(
                f"<div class='success-box'>Status Prediksi: <b>{status}</b></div><br>",
                unsafe_allow_html=True
            )
            st.success("Bagus! Mahasiswa ini diprediksi akan Lulus.")

        st.write("---")
        st.subheader("Probabilitas Prediksi")
        st.info(f"Probabilitas Lulus (Graduate): **{probabilities[0]*100:.2f}%**")
        st.info(f"Probabilitas Keluar (Dropout): **{probabilities[1]*100:.2f}%**")

        st.markdown("---")
        # st.subheader("Fitur yang Digunakan dalam Prediksi")
        # # Menampilkan fitur input beserta fitur turunan
        # st.json(final_input_for_prediction.iloc[0].to_dict())

# Penjelasan Singkat Aplikasi
st.sidebar.title("Tentang Aplikasi Ini")
st.sidebar.info(
    "Aplikasi ini menggunakan model Machine Learning (terbaik dari Random Forest) "
    "untuk memprediksi status kelulusan mahasiswa (Lulus/Dropout). "
    "Model dilatih dengan data historis untuk mengidentifikasi pola yang mengarah pada dropout."
    "\n\n"
    "Data yang dimasukkan akan diproses (misalnya, menghitung IPK, rasio SKS) sebelum diberikan ke model."
)
st.sidebar.markdown(
    """
    **Fitur Utama:**
    - Prediksi real-time status mahasiswa.
    - Menampilkan probabilitas untuk setiap status.
    - Memberikan gambaran fitur-fitur yang digunakan.
    """
)
st.sidebar.markdown("---")
st.sidebar.write("Dibuat dengan Spirit 🔥🔥🔥")