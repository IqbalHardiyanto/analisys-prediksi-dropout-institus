# Analisis Prediksi Dropout Mahasiswa Pada Institusi Jaya Jaya Maju

### **Business Understanding**

**a) Permasalahan Bisnis**
Jaya Jaya Institut mengalami **tingkat dropout siswa yang tinggi**, berdampak pada reputasi akademis dan keberlanjutan institusi. Deteksi dini siswa berisiko dropout diperlukan untuk memberikan intervensi tepat waktu, mempertahankan kualitas lulusan, dan meminimalkan kerugian operasional.

**b) Cakupan Proyek**Proyek ini berfokus pada:

1. **Analisis Data Historis**: Menyelidiki faktor-faktor yang memengaruhi status siswa (Dropout, Enrolled, Graduate).
2. **Prediksi Risiko Dropout**: Membangun model klasifikasi berbasis fitur akademik, demografi, dan sosio-ekonomi.
3. **Identifikasi Pola Kritis**: Menemukan indikator utama penyebab dropout (misal: kinerja akademik, status keuangan, atau latar belakang keluarga).

---

Sumber Dataset: [Predict Students&#39; Dropout and Academic Success](https://archive.ics.uci.edu/dataset/697/predict+students+dropout+and+academic+success)

---

#### Mengimplementasikan prototipe aplikasi Streamlit, harap ikuti prosedur berikut secara cermat:

1. Persiapan Lingkungan Python
   Pastikan sistem Anda telah terinstal Python (disarankan versi 3.9 atau lebih tinggi) dan pip (sebagai sistem manajemen paket Python).
2. Instalasi Pustaka yang Diperlukan
   Silakan buka antarmuka Terminal atau Command Prompt Anda, kemudian eksekusi perintah berikut untuk menginstal seluruh pustaka Python:

```
pip install streamlit pandas scikit-learn xgboost
```

3. Pengaturan Struktur Direktori Proyek
   Pastikan struktur direktori proyek Anda terorganisir dengan baik dan sesuai dengan ekspektasi aplikasi app.py:

```
your_project_folder/
├── app.py (File aplikasi Streamlit)
├── data.csv (File dataset Anda)
└── models/
├── best_model.joblib
└── preprocessor.joblib
```

Pastikan app.py, data.csv, dan direktori models/ berada dalam satu direktori yang sama.

4. Peluncuran Aplikasi Streamlit
   Setelah seluruh persiapan di atas rampung, buka kembali antarmuka Terminal atau Command Prompt Anda dan navigasikan ke direktori your_project_folder:

```
cd path/to/your_project_folder
```

(Harap ganti path/to/your_project_folder dengan jalur absolut ke direktori proyek Anda.)

Selanjutnya, eksekusi perintah berikut untuk meluncurkan aplikasi Streamlit:

```
streamlit run app.py
```

5. Akses Aplikasi melalui Peramban Web
   Segera setelah perintah di atas dieksekusi, Streamlit akan secara otomatis membuka aplikasi di peramban web default Anda (umumnya pada alamat http://localhost:8501).
6. Streamlit Community Cloud: [Prototipe](https://6xnygjgyqw5vjxvh4xd6w6.streamlit.app/)

---

### **Conclusion**

Berdasarkan analisis data:

1. **Faktor Penentu Dropout**:
   - **Kinerja Akademik Awal**: Siswa dengan nilai rendah di semester pertama (`Curricular_units_1st_sem_grade < 10`) dan persentase mata kuliah lulus rendah (`Curricular_units_1st_sem_approved / enrolled < 50%`) berisiko tinggi dropout.
   - **Status Keuangan**: Siswa dengan status _debtor_ (`Debtor = 1`) dan keterlambatan pembayaran (`Tuition_fees_up_to_date = 0`) cenderung dropout.
   - **Profil Demografi**: Siswa internasional (`International = 1`) dan usia di atas 30 tahun (`Age_at_enrollment > 30`) lebih rentan.
2. **Model Prediktif**:
   - Model berbasis **Random Forest/XGBoost** akurat memprediksi dropout (_accuracy > 85%_) dengan fitur kunci: `Admission_grade`, `Tuition_fees_up_to_date`, dan kinerja semester pertama.

---

### **Rekomendasi Action Items**

**a) Program Intervensi Akademik**

- **Target**: Siswa dengan nilai masuk rendah (`Admission_grade < 110`) dan kinerja semester pertama buruk.
- **Aksi**:
  - Berikan _remedial class_ gratis dan pendampingan mentor akademik.
  - Monitor perkembangan melalui _early warning system_ berbasis prediksi model.
- **Metrik Sukses**: Penurunan 30% dropout pada kelompok ini dalam 1 tahun.

**b) Dukungan Finansial & Sosial**

- **Target**: Siswa _debtor_ (`Debtor = 1`) dan status ekonomi rendah (indikator: `Mothers_occupation = 9/Unskilled Workers`).
- **Aksi**:
  - Tingkatkan akses beasiswa dan _flexible payment plan_.
  - Buka layanan konseling untuk masalah non-akademik (keuangan, keluarga).
- **Metrik Sukses**: Penurunan 40% siswa _debtor_ yang dropout dalam 2 tahun.

**c) Optimasi Rekrutmen & Retensi**

- **Target**: Siswa internasional (`International = 1`) dan usia dewasa (`Age_at_enrollment > 30`).
- **Aksi**:
  - Siapkan program orientasi khusus (bahasa, budaya, jaringan alumni).
  - Kembangkan kelas _evening attendance_ (`Daytime_evening_attendance = 0`) untuk fleksibilitas.
- **Metrik Sukses**: Peningkatan 25% retensi siswa internasional dalam 18 bulan.
