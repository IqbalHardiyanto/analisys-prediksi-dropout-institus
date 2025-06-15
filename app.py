# app.py
import streamlit as st
import pandas as pd
import joblib
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from io import BytesIO
import re

# Konfigurasi halaman
st.set_page_config(
    page_title="Sistem Deteksi Risiko Dropout",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load model dan komponen
@st.cache_resource
def load_components():
    model = joblib.load('model/best_model.joblib')
    return model

model = load_components()

# Fungsi untuk membersihkan dan mengkonversi data
def clean_and_convert(value):
    """Konversi nilai ke float dengan penanganan khusus"""
    try:
        # Handle format angka Eropa (1.000,00)
        if isinstance(value, str):
            value = value.replace('.', '').replace(',', '.')
        return float(value)
    except:
        return 0.0

# Fungsi untuk memproses dataframe
def preprocess_dataframe(df):
    """Lakukan preprocessing pada dataframe input"""
    # Normalisasi nama kolom: lowercase dan ganti spasi/tanda baca
    df.columns = [col.strip().lower().replace(' ', '_').replace('-', '_') 
                 for col in df.columns]
    
    # Daftar mapping kolom kritis
    column_mapping = {
        'curricular_units_1st_sem_(grade|approved|enrolled)': 'curricular_units_1st_sem_',
        'curricular_units_2nd_sem_(grade|approved|enrolled)': 'curricular_units_2nd_sem_',
        'marital_status': 'marital_status',
        'application_mode': 'application_mode',
        'age_at_enrollment': 'age_at_enrollment',
        'tuition_fees_up_to_date': 'tuition_fees_up_to_date',
        'scholarship_holder': 'scholarship_holder',
        'debtor': 'debtor',
        'international': 'international',
        'status': 'status'
    }
    
    # Standarisasi nama kolom
    new_columns = []
    for col in df.columns:
        matched = False
        for pattern, replacement in column_mapping.items():
            if re.search(pattern, col, re.IGNORECASE):
                new_col = re.sub(pattern, replacement, col, flags=re.IGNORECASE)
                new_columns.append(new_col)
                matched = True
                break
        if not matched:
            new_columns.append(col)
    
    df.columns = new_columns
    
    # Konversi tipe data untuk kolom kritis
    numeric_cols = ['curricular_units_1st_sem_grade', 
                   'curricular_units_2nd_sem_grade',
                   'age_at_enrollment']
    
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
    
    return df

# Fungsi feature engineering
def add_derived_features(df):
    """Tambahkan fitur turunan"""
    df['ipk_semester1'] = (df['curricular_units_1st_sem_grade'] / 20) * 4
    df['ipk_semester2'] = (df['curricular_units_2nd_sem_grade'] / 20) * 4
    
    # Handle pembagian oleh nol
    df['proporsi_sks_1'] = df.apply(
        lambda x: x['curricular_units_1st_sem_approved'] / 
        max(x['curricular_units_1st_sem_enrolled'], 1), 
        axis=1
    )
    
    df['proporsi_sks_2'] = df.apply(
        lambda x: x['curricular_units_2nd_sem_approved'] / 
        max(x['curricular_units_2nd_sem_enrolled'], 1), 
        axis=1
    )
    
    df['index_ipk'] = df['ipk_semester2'] - df['ipk_semester1']
    df['kemajuan_sks'] = df['curricular_units_2nd_sem_approved'] - df['curricular_units_1st_sem_approved']
    
    # Pastikan nilai boolean
    df['tuition_fees_up_to_date'] = df['tuition_fees_up_to_date'].apply(lambda x: 1 if x == 1 else 0)
    df['debtor'] = df['debtor'].apply(lambda x: 1 if x == 1 else 0)
    
    df['status_pembayaran'] = df.apply(
        lambda x: (0 if x['tuition_fees_up_to_date'] == 1 else 1) + x['debtor'], 
        axis=1
    )
    
    return df

# Fungsi prediksi
def predict_dropout_risk(df):
    """Memprediksi risiko dropout untuk seluruh dataframe"""
    try:
        processed_df = add_derived_features(df.copy())
        
        # Fitur yang digunakan model
        features = [
            'curricular_units_1st_sem_enrolled', 'curricular_units_1st_sem_approved',
            'curricular_units_2nd_sem_enrolled', 'curricular_units_2nd_sem_approved', 'international',
            'mothers_occupation', 'ipk_semester1', 'ipk_semester2', 'age_at_enrollment', 'tuition_fees_up_to_date',
            'scholarship_holder', 'debtor', 'proporsi_sks_1', 'proporsi_sks_2',
            'index_ipk', 'kemajuan_sks', 'status_pembayaran'
        ]
        
        # Pastikan semua fitur ada
        for feature in features:
            if feature not in processed_df.columns:
                st.error(f"Fitur penting tidak ditemukan: {feature}")
                processed_df[feature] = 0
        
        X = processed_df[features]
        
        # Pastikan tidak ada nilai NaN
        if X.isnull().any().any():
            st.warning("Terdapat nilai NaN pada fitur, mengisi dengan 0")
            X.fillna(0, inplace=True)
        
        probabilities = model.predict_proba(X)[:, 1]  # Probabilitas dropout
        predictions = (probabilities > 0.7).astype(int)  # Threshold 70%
        
        results_df = df.copy()
        results_df['dropout_probability'] = probabilities
        results_df['risiko_tinggi'] = predictions
        results_df['rekomendasi'] = results_df.apply(generate_recommendation, axis=1)
        
        return results_df.sort_values('dropout_probability', ascending=False)
    
    except Exception as e:
        st.error(f"Terjadi kesalahan dalam prediksi: {str(e)}")
        st.error("Pastikan format data sesuai dengan contoh")
        return pd.DataFrame()

# Fungsi rekomendasi
def generate_recommendation(row):
    """Hasilkan rekomendasi berdasarkan fitur siswa"""
    recommendations = []
    
    try:
        # Rekomendasi akademik
        if row['proporsi_sks_1'] < 0.5:
            recommendations.append("📚 Program remedial mata kuliah semester 1")
        if row.get('ipk_semester1', 0) < 2.0:
            recommendations.append("🎯 Bimbingan belajar intensif 3x/minggu")
        
        # Rekomendasi keuangan
        if row.get('debtor', 0) == 1:
            recommendations.append("💳 Konsultasi restrukturisasi utang pendidikan")
        if row.get('tuition_fees_up_to_date', 0) == 0:
            recommendations.append("💰 Skema pembayaran fleksibel (cicilan 6x)")
        
        # Rekomendasi khusus
        if row.get('international', 0) == 1:
            recommendations.append("🌍 Konseling adaptasi budaya & bahasa")
        if row.get('age_at_enrollment', 0) > 25:
            recommendations.append("🕒 Kelas malam/jarak jauh")
    except:
        recommendations.append("Data tidak lengkap untuk rekomendasi")
    
    return " | ".join(recommendations) if recommendations else "Pantau berkala"

# Tampilan antarmuka
st.title("🎓 Sistem Deteksi Risiko Dropout - Jaya Jaya Institut")
st.markdown("""
**Identifikasi siswa berisiko tinggi dropout dan berikan intervensi tepat waktu**  
*Berdasarkan analisis kinerja akademik, status keuangan, dan profil demografi*
""")

# Menu sidebar
with st.sidebar:
    st.header("Pengaturan Sistem")
    risk_threshold = st.slider("Threshold Risiko Tinggi", 0.5, 1.0, 0.7, 0.05)
    st.divider()
    
    st.subheader("Panduan Penggunaan")
    st.markdown("""
    1. Upload file CSV data siswa
    2. Klik **Proses Data**
    3. Lihat hasil prediksi & rekomendasi
    4. Download hasil analisis
    """)
    
    st.divider()
    st.markdown("🛠️ **Tim Pengembang**: Divisi Data Science Institusi")
    st.markdown("📅 Versi: 1.0 | Juni 2024")

# Tab utama
tab1, tab2, tab3 = st.tabs(["Upload Data", "Analisis Siswa", "Dashboard Institusi"])

with tab1:
    st.subheader("Upload Data Siswa")
    
    # Tampilkan contoh data
    with st.expander("Contoh Format Data"):
        example_data = {
            'Student_ID': [101, 102],
            'Curricular_units_1st_sem_grade': [14.5, 9.8],
            'Curricular_units_2nd_sem_grade': [12.0, 8.5],
            'Curricular_units_1st_sem_approved': [5, 3],
            'Curricular_units_1st_sem_enrolled': [6, 6],
            'Curricular_units_2nd_sem_approved': [4, 2],
            'Curricular_units_2nd_sem_enrolled': [6, 6],
            'Age_at_enrollment': [19, 21],
            'Tuition_fees_up_to_date': [1, 0],
            'Scholarship_holder': [0, 1],
            'Debtor': [0, 1],
            'International': [0, 0],
            'Mothers_occupation': [5, 9],
            'Course': ['Computer Science', 'Business']
        }
        example_df = pd.DataFrame(example_data)
        st.dataframe(example_df)
        st.download_button(
            "📥 Download Contoh Data",
            example_df.to_csv(index=False).encode('utf-8'),
            "contoh_data_siswa.csv",
            "text/csv"
        )
    
    uploaded_file = st.file_uploader("Pilih file CSV", type="csv")
    
    if uploaded_file:
        try:
            # Coba baca dengan delimiter yang berbeda
            try:
                df = pd.read_csv(uploaded_file, delimiter=';')
            except:
                df = pd.read_csv(uploaded_file)
            
            # Proses data
            df = preprocess_dataframe(df)
            
            st.session_state['raw_data'] = df
            st.success(f"Data berhasil diunggah! {len(df)} rekaman ditemukan")
            
            with st.expander("Pratinjau Data (5 baris pertama)"):
                st.dataframe(df.head())
            
            if st.button("Proses Data", type="primary"):
                with st.spinner("Menganalisis risiko dropout..."):
                    results_df = predict_dropout_risk(df)
                    
                    if not results_df.empty:
                        st.session_state['results'] = results_df
                        st.success("Analisis selesai!")
                        
                        # Hitung statistik
                        high_risk_count = results_df['risiko_tinggi'].sum()
                        risk_percentage = (high_risk_count / len(results_df)) * 100
                        
                        st.metric("Siswa Berisiko Tinggi", 
                                 f"{high_risk_count} siswa ({risk_percentage:.1f}%)",
                                 delta_color="inverse")
                    else:
                        st.error("Tidak ada hasil yang bisa ditampilkan")
        except Exception as e:
            st.error(f"Gagal memproses file: {str(e)}")
            st.error("Pastikan file dalam format CSV yang benar")

with tab2:
    if 'results' in st.session_state and not st.session_state['results'].empty:
        results_df = st.session_state['results']
        
        # Filter siswa berisiko
        high_risk_df = results_df[results_df['risiko_tinggi'] == 1]
        
        col1, col2 = st.columns([3, 2])
        
        with col1:
            st.subheader("Siswa Berisiko Tinggi")
            
            # Pilih kolom untuk ditampilkan
            display_cols = ['student_id', 'dropout_probability', 'rekomendasi']
            
            # Tambahkan kolom tambahan jika ada
            for col in ['course', 'age_at_enrollment']:
                if col in results_df.columns:
                    display_cols.append(col)
            
            st.dataframe(
                high_risk_df[display_cols].head(10),
                height=400
            )
            
            # Download hasil
            csv = results_df.to_csv(index=False).encode('utf-8')
            st.download_button(
                "📥 Download Hasil Lengkap",
                csv,
                "hasil_deteksi_dropout.csv",
                "text/csv"
            )
        
        with col2:
            st.subheader("Distribusi Risiko")
            fig, ax = plt.subplots()
            sns.histplot(results_df['dropout_probability'], bins=20, kde=True, ax=ax)
            ax.axvline(x=risk_threshold, color='r', linestyle='--')
            ax.set_title("Distribusi Probabilitas Dropout")
            ax.set_xlabel("Probabilitas Dropout")
            ax.set_ylabel("Jumlah Siswa")
            st.pyplot(fig)
            
            # Tampilkan detail siswa
            if not high_risk_df.empty:
                selected_id = st.selectbox("Pilih ID Siswa", high_risk_df['student_id'].unique())
                student_data = high_risk_df[high_risk_df['student_id'] == selected_id].iloc[0]
                
                st.metric("Probabilitas Dropout", f"{student_data['dropout_probability']*100:.1f}%")
                st.write("**Faktor Risiko:**")
                
                if 'ipk_semester1' in student_data:
                    st.markdown(f"- IPK Semester 1: {student_data['ipk_semester1']:.2f}")
                
                if 'proporsi_sks_1' in student_data:
                    st.markdown(f"- Persentase SKS Lulus: {student_data['proporsi_sks_1']*100:.1f}%")
                
                if 'status_pembayaran' in student_data:
                    st.markdown(f"- Status Pembayaran: {'Bermasalah' if student_data['status_pembayaran'] > 0 else 'Lancar'}")
            else:
                st.info("Tidak ada siswa berisiko tinggi")

with tab3:
    if 'results' in st.session_state and not st.session_state['results'].empty:
        results_df = st.session_state['results']
        
        st.subheader("Dashboard Analisis Institusi")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("### Tren Risiko")
            # Asumsi ada kolom 'course' untuk program studi
            if 'course' in results_df:
                fig, ax = plt.subplots()
                risk_by_dept = results_df.groupby('course')['risiko_tinggi'].mean().sort_values()
                risk_by_dept.plot(kind='barh', ax=ax, color='salmon')
                ax.set_title("Risiko Dropout per Program Studi")
                ax.set_xlabel("Persentase Berisiko Tinggi")
                st.pyplot(fig)
            else:
                st.warning("Kolom 'course' tidak ditemukan untuk analisis program studi")
        
        with col2:
            st.markdown("### Faktor Penentu")
            # Ambil feature importance dari model
            if hasattr(model.named_steps['classifier'], 'feature_importances_'):
                importances = model.named_steps['classifier'].feature_importances_
                features = [
                    'SKS Semester 1', 'SKS Lulus S1', 'SKS Semester 2', 'SKS Lulus S2',
                    'Internasional', 'Pekerjaan Ibu', 'IPK S1', 'IPK S2', 'Usia',
                    'Pembayaran Tepat', 'Beasiswa', 'Debitur', '%SKS Lulus S1',
                    '%SKS Lulus S2', 'Perubahan IPK', 'Kemajuan SKS', 'Status Bayar'
                ]
                
                feat_imp = pd.DataFrame({'Feature': features, 'Importance': importances})
                feat_imp = feat_imp.sort_values('Importance', ascending=False).head(8)
                
                fig, ax = plt.subplots()
                sns.barplot(y='Feature', x='Importance', data=feat_imp, palette="viridis", ax=ax)
                ax.set_title("Faktor Penentu Risiko Dropout")
                st.pyplot(fig)
            else:
                st.info("Model saat ini tidak memiliki fitur importance (misal: SVM)")
        
        with col3:
            st.markdown("### Rekomendasi Institusi")
            
            # Hitung rekomendasi populer
            if 'rekomendasi' in results_df:
                all_recommendations = "|".join(results_df['rekomendasi'].dropna()).split("|")
                rec_counts = pd.Series(all_recommendations).value_counts().head(5)
                
                st.write("**Intervensi Paling Dibutuhkan:**")
                for rec, count in rec_counts.items():
                    st.markdown(f"- {rec} ({count} siswa)")
            else:
                st.warning("Kolom 'rekomendasi' tidak ditemukan")
            
            st.divider()
            
            # Statistik
            avg_risk = results_df['dropout_probability'].mean() * 100
            debt_risk = results_df[results_df['debtor'] == 1]['dropout_probability'].mean() * 100
            
            st.metric("Rata-rata Risiko Institusi", f"{avg_risk:.1f}%")
            st.metric("Risiko Siswa Berhutang", f"{debt_risk:.1f}%")

        # Visualisasi Interaktif
        with st.expander("Visualisasi Interaktif", expanded=True):
            # Pastikan ada kolom yang diperlukan
            if 'ipk_semester1' in results_df and 'proporsi_sks_1' in results_df and 'age_at_enrollment' in results_df:
                selected_feature = st.selectbox("Pilih fitur untuk analisis", 
                                              ['ipk_semester1', 'proporsi_sks_1', 'age_at_enrollment'])
                
                fig, ax = plt.subplots()
                sns.scatterplot(data=results_df, 
                               x=selected_feature, 
                               y='dropout_probability',
                               hue='risiko_tinggi',
                               palette={0: 'green', 1: 'red'},
                               ax=ax)
                ax.set_title(f"Risiko vs {selected_feature}")
                st.pyplot(fig)
            else:
                st.warning("Kolom yang diperlukan untuk visualisasi tidak tersedia")
        
        # Simulasi intervensi
        with st.expander("Simulasi Intervensi", expanded=True):
            st.write("Uji efektivitas strategi intervensi:")
            intervention = st.selectbox("Pilih intervensi", [
                "Program remedial akademik",
                "Bantuan keuangan darurat",
                "Konseling adaptasi",
                "Penyesuaian beban SKS"
            ])
            
            # Model efek intervensi (simplified)
            if st.button("Proyeksikan Hasil"):
                current_risk = results_df['risiko_tinggi'].mean()
                projected_risk = current_risk * 0.7  # Asumsi 30% penurunan
                
                col1, col2 = st.columns(2)
                col1.metric("Risiko Saat Ini", f"{current_risk*100:.1f}%")
                col2.metric("Proyeksi Setelah Intervensi", f"{projected_risk*100:.1f}%", 
                           delta=f"-{30}%", delta_color="inverse")
                
                st.info(f"Estimasi dampak {intervention}: Mencegah dropout ~{int(len(results_df)*0.3)} siswa")
    else:
        st.info("Silakan proses data di tab Upload Data terlebih dahulu")

# Mode input manual
st.sidebar.divider()
st.sidebar.subheader("Prediksi Individual")

with st.sidebar.form("manual_input"):
    st.write("Masukkan data siswa secara manual:")
    age = st.number_input("Usia", 17, 50, 20)
    grade_s1 = st.number_input("Nilai Rata-rata Semester 1", 0.0, 20.0, 12.5)
    approved_s1 = st.number_input("SKS Lulus Semester 1", 0, 30, 15)
    enrolled_s1 = st.number_input("SKS Diambil Semester 1", 0, 30, 20)
    tuition = st.selectbox("Pembayaran Tepat Waktu", [1, 0], format_func=lambda x: "Ya" if x == 1 else "Tidak")
    debtor = st.selectbox("Memiliki Hutang", [1, 0], format_func=lambda x: "Ya" if x == 1 else "Tidak")
    
    submit_manual = st.form_submit_button("Prediksi")
    if submit_manual:
        sample_data = {
            'student_id': ['MANUAL_001'],
            'age_at_enrollment': [age],
            'curricular_units_1st_sem_grade': [grade_s1],
            'curricular_units_1st_sem_approved': [approved_s1],
            'curricular_units_1st_sem_enrolled': [enrolled_s1],
            'tuition_fees_up_to_date': [tuition],
            'debtor': [debtor],
            'international': [0],
            'mothers_occupation': [9],
            'curricular_units_2nd_sem_grade': [12],
            'curricular_units_2nd_sem_approved': [12],
            'curricular_units_2nd_sem_enrolled': [18],
            'scholarship_holder': [0]
        }
        
        manual_df = pd.DataFrame(sample_data)
        result = predict_dropout_risk(manual_df)
        
        if not result.empty:
            proba = result['dropout_probability'].iloc[0]
            st.sidebar.success(f"Probabilitas Dropout: {proba*100:.1f}%")
            st.sidebar.write("**Rekomendasi:**")
            st.sidebar.info(result['rekomendasi'].iloc[0])
        else:
            st.sidebar.error("Gagal melakukan prediksi")

# Pesan jika belum ada data
if 'results' not in st.session_state:
    st.info("Silakan upload data siswa di tab Upload Data untuk memulai analisis")