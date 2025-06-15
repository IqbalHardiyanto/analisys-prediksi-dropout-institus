# Analisis Prediksi Dropout Mahasiswa Pada Institusi Jaya Jaya Maju

### **Business Understanding**

**a) Permasalahan Bisnis**  
Jaya Jaya Institut mengalami **tingkat dropout siswa yang tinggi**, berdampak pada reputasi akademis dan keberlanjutan institusi. Deteksi dini siswa berisiko dropout diperlukan untuk memberikan intervensi tepat waktu, mempertahankan kualitas lulusan, dan meminimalkan kerugian operasional.

**b) Cakupan Proyek**  
Proyek ini berfokus pada:

1. **Analisis Data Historis**: Menyelidiki faktor-faktor yang memengaruhi status siswa (Dropout, Enrolled, Graduate).
2. **Prediksi Risiko Dropout**: Membangun model klasifikasi berbasis fitur akademik, demografi, dan sosio-ekonomi.
3. **Identifikasi Pola Kritis**: Menemukan indikator utama penyebab dropout (misal: kinerja akademik, status keuangan, atau latar belakang keluarga).

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
