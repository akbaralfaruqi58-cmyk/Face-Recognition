# 👤 Face Recognition dengan InsightFace & OpenCV

Repositori ini berisi implementasi sistem pengenalan wajah (*face recognition*) berbasis Python menggunakan pustaka **InsightFace** (RetinaFace untuk deteksi, ArcFace untuk ekstraksi *embedding*) dan **OpenCV**. Sistem ini dirancang untuk membaca referensi gambar, membangun database pengenalan wajah secara dinamis, dan mencocokkan wajah baik pada gambar statis maupun video.

Sistem ini dioptimalkan untuk dapat berjalan di lingkungan lokal.

## ✨ Fitur Utama

* **Deteksi & Ekstraksi Akurat:** Menggunakan model state-of-the-art `buffalo_l` (atau `buffalo_s` untuk performa lebih cepat) dari ekosistem InsightFace.
* **Auto-Build Database:** Sistem secara otomatis memindai folder `known_faces/`, menghitung *mean embedding* dari setiap identitas, dan menyimpannya ke dalam `face_db.pkl`.
* **Cosine Similarity:** Pencocokan wajah menggunakan metrik *cosine similarity* dengan *threshold* yang dapat disesuaikan.

## 🛠️ Prasyarat & Instalasi

Pastikan Anda menggunakan **Python 3.7+**. Instal semua pustaka yang dibutuhkan dengan menjalankan perintah berikut di cmd.

```bash
pip install insightface onnxruntime opencv-python numpy

```

## 📁 Struktur Direktori

Agar skrip berjalan dengan baik, susun direktori proyek Anda seperti berikut:

```text
📦 Face_Recognition_Project
 ┣ 📂 known_faces/          # Folder utama untuk referensi wajah
 ┃ ┣ 📂 Ben_Affleck/        # Nama subfolder = Nama Identitas
 ┃ ┃ ┣ 📜 gambar1.jpg
 ┃ ┃ ┗ 📜 gambar2.jpg
 ┃ ┣ 📂 Keanu_Reeves/
 ┃ ┃ ┗ 📜 foto1.jpg
 ┃ ┗ 📂 ...
 ┣ 📜 face_db.pkl           # Terbuat otomatis setelah skrip dijalankan
 ┣ 📜 main.py               # Skrip utama face recognition
 ┗ 📜 README.md             # Dokumentasi

```

## 🚀 Cara Penggunaan

1. **Siapkan Data Referensi:**
Buat folder `known_faces/` di dalam direktori proyek Anda. Buat subfolder untuk masing-masing orang yang ingin dikenali dan masukkan 1-5 foto wajah yang jelas ke dalamnya.
2. **Jalankan Skrip:**
Eksekusi skrip utama menggunakan Python.
```bash
python main.py

```


3. **Pilih Mode Evaluasi:**
Saat skrip berjalan, menu interaktif akan muncul:
* **Mode 1 (Gambar Tunggal):** Masukkan *path* lengkap menuju file gambar yang ingin Anda uji. Sistem akan menampilkan gambar beserta *bounding box* dan label nama.
* **Mode 2 (Video/Live):** Memproses *frame* secara beruntun.

```


3. **Kendala Mode Live (Webcam):**
Fungsi `cv2.VideoCapture(0)` tidak dapat mengakses webcam lokal Anda secara langsung melalui Colab. Jika ingin menggunakan Mode 2 di Colab, ganti angka `0` dengan *path* menuju file video uji (misal: `cv2.VideoCapture("/content/drive/MyDrive/video_uji.mp4")`).
