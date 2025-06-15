# app.py
import streamlit as st
import pandas as pd
import joblib
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from io import BytesIO

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
    preprocessor = joblib.load('model/preprocessor.joblib')
    return model, preprocessor

model, preprocessor = load_components()

# Fungsi feature engineering
def add_derived_features(df):
    df['Ipk_semester1'] = (df['Curricular_units_1st_sem_grade'] / 20) * 4
    df['Ipk_semester2'] = (df['Curricular_units_2nd_sem_grade'] / 20) * 4
    
    df['proporsi_sks_1'] = df['Curricular_units_1st_sem_approved'] / df['Curricular_units_1st_sem_enrolled'].replace(0, 1)
    df['proporsi_sks_2'] = df['Curricular_units_2nd_sem_approved'] / df['Curricular_units_2nd_sem_enrolled'].replace(0, 1)
    
    df['index_ipk'] = df['Ipk_semester2'] - df['Ipk_semester1']
    df['kemajuan_sks'] = df['Curricular_units_2nd_sem_approved'] - df['Curricular_units_1st_sem_approved']
    
    df['status_pembayaran'] = df['Tuition_fees_up_to_date'].apply(lambda x: 0 if x == 1 else 1) + df['Debtor']
    
    return df

# Fungsi prediksi
def predict_dropout_risk(df):
    """Memprediksi risiko dropout untuk seluruh dataframe"""
    processed_df = add_derived_features(df.copy())
    
    # Fitur yang digunakan model
    features = [
        'Curricular_units_1st_sem_enrolled', 'Curricular_units_1st_sem_approved',
        'Curricular_units_2nd_sem_enrolled', 'Curricular_units_2nd_sem_approved', 'International',
        'Mothers_occupation', 'Ipk_semester1', 'Ipk_semester2', 'Age_at_enrollment', 'Tuition_fees_up_to_date',
        'Scholarship_holder', 'Debtor', 'proporsi_sks_1', 'proporsi_sks_2',
        'index_ipk', 'kemajuan_sks', 'status_pembayaran'
    ]
    
    X = processed_df[features]
    probabilities = model.predict_proba(X)[:, 1]  # Probabilitas dropout
    predictions = (probabilities > 0.7).astype(int)  # Threshold 70%
    
    results_df = df.copy()
    results_df['Dropout Probability'] = probabilities
    results_df['Risiko Tinggi'] = predictions
    results_df['Rekomendasi'] = results_df.apply(generate_recommendation, axis=1)
    
    return results_df.sort_values('Dropout Probability', ascending=False)

# Fungsi rekomendasi
def generate_recommendation(row):
    """Hasilkan rekomendasi berdasarkan fitur siswa"""
    recommendations = []
    
    # Rekomendasi akademik
    if row['proporsi_sks_1'] < 0.5:
        recommendations.append("📚 Program remedial mata kuliah semester 1")
    if row['Ipk_semester1'] < 2.0:
        recommendations.append("🎯 Bimbingan belajar intensif 3x/minggu")
    
    # Rekomendasi keuangan
    if row['Debtor'] == 1:
        recommendations.append("💳 Konsultasi restrukturisasi utang pendidikan")
    if row['Tuition_fees_up_to_date'] == 0:
        recommendations.append("💰 Skema pembayaran fleksibel (cicilan 6x)")
    
    # Rekomendasi khusus
    if row['International'] == 1:
        recommendations.append("🌍 Konseling adaptasi budaya & bahasa")
    if row['Age_at_enrollment'] > 25:
        recommendations.append("🕒 Kelas malam/jarak jauh")
    
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
    uploaded_file = st.file_uploader("Pilih file CSV", type="csv")
    
    if uploaded_file:
        df = pd.read_csv(uploaded_file, delimiter=';')
        st.session_state['raw_data'] = df
        st.success(f"Data berhasil diunggah! {len(df)} rekaman ditemukan")
        
        with st.expander("Pratinjau Data"):
            st.dataframe(df.head())
        
        if st.button("Proses Data", type="primary"):
            with st.spinner("Menganalisis risiko dropout..."):
                results_df = predict_dropout_risk(df)
                st.session_state['results'] = results_df
                st.success("Analisis selesai!")
                
                # Hitung statistik
                high_risk_count = results_df['Risiko Tinggi'].sum()
                risk_percentage = (high_risk_count / len(results_df)) * 100
                
                st.metric("Siswa Berisiko Tinggi", 
                         f"{high_risk_count} siswa ({risk_percentage:.1f}%)",
                         delta_color="inverse")

with tab2:
    if 'results' in st.session_state:
        results_df = st.session_state['results']
        
        # Filter siswa berisiko
        high_risk_df = results_df[results_df['Risiko Tinggi'] == 1]
        
        col1, col2 = st.columns([3, 2])
        
        with col1:
            st.subheader("Siswa Berisiko Tinggi")
            st.dataframe(
                high_risk_df[['Student_ID', 'Dropout Probability', 'Rekomendasi']].head(10),
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
            sns.histplot(results_df['Dropout Probability'], bins=20, kde=True, ax=ax)
            ax.axvline(x=risk_threshold, color='r', linestyle='--')
            ax.set_title("Distribusi Probabilitas Dropout")
            ax.set_xlabel("Probabilitas Dropout")
            ax.set_ylabel("Jumlah Siswa")
            st.pyplot(fig)
            
            # Tampilkan detail siswa
            selected_id = st.selectbox("Pilih ID Siswa", high_risk_df['Student_ID'].unique())
            student_data = high_risk_df[high_risk_df['Student_ID'] == selected_id].iloc[0]
            
            st.metric("Probabilitas Dropout", f"{student_data['Dropout Probability']*100:.1f}%")
            st.write("**Faktor Risiko:**")
            st.markdown(f"- IPK Semester 1: {student_data['Ipk_semester1']:.2f}")
            st.markdown(f"- Persentase SKS Lulus: {student_data['proporsi_sks_1']*100:.1f}%")
            st.markdown(f"- Status Pembayaran: {'Bermasalah' if student_data['status_pembayaran'] > 0 else 'Lancar'}")

with tab3:
    if 'results' in st.session_state:
        results_df = st.session_state['results']
        
        st.subheader("Dashboard Analisis Institusi")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("### Tren Risiko")
            # Asumsi ada kolom 'Course' untuk program studi
            if 'Course' in results_df:
                fig, ax = plt.subplots()
                risk_by_dept = results_df.groupby('Course')['Risiko Tinggi'].mean().sort_values()
                risk_by_dept.plot(kind='barh', ax=ax, color='salmon')
                ax.set_title("Risiko Dropout per Program Studi")
                ax.set_xlabel("Persentase Berisiko Tinggi")
                st.pyplot(fig)
            else:
                st.warning("Kolom 'Course' tidak ditemukan untuk analisis program studi")
        
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
            if 'Rekomendasi' in results_df:
                all_recommendations = "|".join(results_df['Rekomendasi'].dropna()).split("|")
                rec_counts = pd.Series(all_recommendations).value_counts().head(5)
                
                st.write("**Intervensi Paling Dibutuhkan:**")
                for rec, count in rec_counts.items():
                    st.markdown(f"- {rec} ({count} siswa)")
            else:
                st.warning("Kolom 'Rekomendasi' tidak ditemukan")
            
            st.divider()
            
            # Statistik
            avg_risk = results_df['Dropout Probability'].mean() * 100
            debt_risk = results_df[results_df['Debtor'] == 1]['Dropout Probability'].mean() * 100
            
            st.metric("Rata-rata Risiko Institusi", f"{avg_risk:.1f}%")
            st.metric("Risiko Siswa Berhutang", f"{debt_risk:.1f}%")

        # Visualisasi Interaktif
        with st.expander("Visualisasi Interaktif", expanded=True):
            # Pastikan ada kolom yang diperlukan
            if 'Ipk_semester1' in results_df and 'proporsi_sks_1' in results_df and 'Age_at_enrollment' in results_df:
                selected_feature = st.selectbox("Pilih fitur untuk analisis", 
                                              ['Ipk_semester1', 'proporsi_sks_1', 'Age_at_enrollment'])
                
                fig, ax = plt.subplots()
                sns.scatterplot(data=results_df, 
                               x=selected_feature, 
                               y='Dropout Probability',
                               hue='Risiko Tinggi',
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
                current_risk = results_df['Risiko Tinggi'].mean()
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
            'Student_ID': ['MANUAL_001'],
            'Age_at_enrollment': [age],
            'Curricular_units_1st_sem_grade': [grade_s1],
            'Curricular_units_1st_sem_approved': [approved_s1],
            'Curricular_units_1st_sem_enrolled': [enrolled_s1],
            'Tuition_fees_up_to_date': [tuition],
            'Debtor': [debtor],
            'International': [0],
            'Mothers_occupation': [9],
            'Curricular_units_2nd_sem_grade': [12],
            'Curricular_units_2nd_sem_approved': [12],
            'Curricular_units_2nd_sem_enrolled': [18],
            'Scholarship_holder': [0]
        }
        
        manual_df = pd.DataFrame(sample_data)
        result = predict_dropout_risk(manual_df)
        
        proba = result['Dropout Probability'].iloc[0]
        st.sidebar.success(f"Probabilitas Dropout: {proba*100:.1f}%")
        st.sidebar.write("**Rekomendasi:**")
        st.sidebar.info(result['Rekomendasi'].iloc[0])

# Pesan jika belum ada data
if 'results' not in st.session_state:
    st.info("Silakan upload data siswa di tab Upload Data untuk memulai analisis")