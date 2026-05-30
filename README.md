# Proyek Klasifikasi Uang Kertas Rupiah (Banknote Classification) dengan ResNet18

## 1. Pendahuluan
Proyek ini bertujuan untuk membangun sistem klasifikasi visual nominal uang kertas (Rupiah) menggunakan pendekatan *Deep Learning*. Algoritma yang digunakan adalah arsitektur konvolusi **ResNet18** yang telah di-*fine-tuning* agar sesuai dengan kebutuhan pengenalan kelas-kelas nominal uang kertas pada dataset yang digunakan.

## 2. Arsitektur dan Pendekatan Model
Model ini menggunakan arsitektur bawaan **ResNet18** (Residual Network dengan 18 lapisan) yang terkenal efektif dalam menangani masalah klasifikasi citra. Proses pemodelan melibatkan:
- Pemrosesan gambar dan augmentasi data untuk meningkatkan ketahanan model.
- Penggunaan *pretrained weights* yang kemudian disesuaikan (*fine-tuning*) pada lapisan akhir (klasifikasi) sesuai jumlah kelas pecahan uang kertas yang ada.

## 3. Pelatihan dan Evaluasi (Training & Evaluation)
Selama fase pelatihan, model dipantau kinerjanya agar terhindar dari *overfitting* atau *underfitting*. Beberapa komponen evaluasi utama meliputi:
- **Kurva Loss & Akurasi**: Melacak perbedaan antara *training loss/accuracy* dan *validation loss/accuracy* setiap iterasi (epoch).
- **Confusion Matrix**: Matriks yang secara visual menjabarkan performa prediksi kelas per kelas. Matriks ini membantu melihat nominal uang apa saja yang sering tertukar atau salah diklasifikasikan oleh model.
- **Classification Report**: Laporan yang berisi ukuran metrik yang lebih rinci, meliputi:
  - *Precision*
  - *Recall*
  - *F1-Score*

## 4. Analisis Kesalahan (Error Analysis)
Alih-alih hanya melihat angka akurasi agregat, proyek ini menyertakan sesi khusus untuk menganalisis sampel atau gambar-gambar yang salah diprediksi (*misclassified*).
Sistem akan menyimpan dan menampilkan daftar gambar dengan "Label Asli" (True Label) versus "Prediksi Model" (Predicted Label). Evaluasi kritis ini berguna untuk mengidentifikasi apakah kesalahan terjadi karena:
- Kemiripan warna atau desain antar pecahan nominal tertentu.
- Kualitas gambar yang buruk, terlalu terang (*overexposed*), buram (*blur*), atau terpotong.
- Imbalans atau ketidakseimbangan jumlah sampel pada kelas tertentu.

## 5. Uji Coba dan Prediksi Nyata
Proyek ini juga mencakup blok untuk menampilkan prediksi visual nyata. Beberapa gambar dari *test set* diambil dan diprediksi menggunakan model, lengkap dengan keterangan apakah prediksinya benar (berwarna hijau) atau salah (berwarna merah).

## 6. Hasil Akhir dan Penyimpanan (Artifacts)
Setelah mencapai metrik yang optimal (*best epoch* dan *best validation loss*), model dan ringkasannya disimpan secara permanen untuk dapat digunakan sewaktu-waktu di tahap produksi (misalnya untuk deteksi melalui webcam):
- `resnet18_banknote_complete.pth`: File yang berisi *state dictionary* dari bobot model, arsitektur, resolusi gambar yang digunakan, serta metrik evaluasi model.
- `final_metrics_summary.csv` & `classification_report.csv`: Berkas log berbentuk tabular yang menyimpan statistik dan metrik performa akhir.

## 7. Menjalankan Deteksi Secara Real-Time (Webcam)
Anda dapat menguji coba model yang telah dilatih secara langsung (*real-time*) menggunakan kamera (webcam). Pastikan *environment* Conda Anda sudah aktif.

Jalankan perintah berikut pada terminal:
```bash
conda activate env_deeplearning
python webcam_test.py
```
Arahkan uang kertas Rupiah ke arah kamera untuk melihat hasil prediksinya. Untuk menghentikan program, cukup tekan tombol `q` pada *keyboard*.

## Kesimpulan
Secara keseluruhan, proyek ini bukan hanya melatih model *deep learning* biasa, namun telah mencakup alur standar *machine learning engineering* mulai dari persiapan data, pelatihan, evaluasi mendalam per kelas (*confusion matrix*), analisis kegagalan klasifikasi, hingga pengeksporan model siap pakai.
