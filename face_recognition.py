import cv2
import numpy as np
import pickle
import os
from pathlib import Path
from insightface.app import FaceAnalysis

# ============================================================
# KONFIGURASI
# ============================================================
KNOWN_FACES_DIR = Path("known_faces") # satu subfolder per orang
DB_PATH = "face_db.pkl" # database embedding yang disimpan
THRESHOLD = 0.45 # ambang batas cosine similarity
                 # lebih rendah = pencocokan lebih ketat
MODEL_NAME = "buffalo_l" # buffalo_l: akurasi tinggi
                         # buffalo_s: lebih cepat, lebih ringan

# ============================================================
# 1. INISIALISASI MODEL
# InsightFace membundel: RetinaFace (detector) +
# ArcFace ResNet-50 (embedder) dalam satu paket
# ============================================================
app = FaceAnalysis(name=MODEL_NAME, providers=["CPUExecutionProvider"])
app.prepare(ctx_id=0, det_size=(640, 640))
# ctx_id=0 menggunakan GPU jika tersedia; -1 memaksa CPU

# ============================================================
# 2. MEMBANGUN DATABASE EMBEDDING DARI FOLDER known_faces/
# Untuk setiap orang: hitung embedding untuk setiap foto,
# lalu rata-ratakan menjadi satu vektor representatif.
# ============================================================
def build_database():
    """
    Scan known_faces/<NamaOrang>/*.jpg dan bangun
    dictionary: {nama: mean_embedding_vector}
    """
    db = {}
    if not KNOWN_FACES_DIR.exists():
        os.makedirs(KNOWN_FACES_DIR)
        print(f"Folder {KNOWN_FACES_DIR} dibuat. Silakan isi dengan subfolder nama orang.")
        return db

    for person_dir in sorted(KNOWN_FACES_DIR.iterdir()):
        if not person_dir.is_dir():
            continue
        person_name = person_dir.name
        embeddings = []

        for img_path in sorted(person_dir.glob("*.jpg")):
            img = cv2.imread(str(img_path))
            if img is None:
                print(f" [WARN] Tidak dapat membaca {img_path}, dilewati.")
                continue
            faces = app.get(img) # deteksi + embed semua wajah dalam gambar
            if not faces:
                print(f" [WARN] Tidak ada wajah ditemukan di {img_path.name}")
                continue
            # Ambil wajah terbesar yang terdeteksi (paling menonjol)
            face = max(faces, key=lambda f: f.bbox[2] * f.bbox[3])
            embeddings.append(face.embedding) # vektor ArcFace 512-d

        if embeddings:
            # Rata-ratakan semua embedding untuk orang ini
            mean_emb = np.stack(embeddings).mean(axis=0)
            # Normalisasi L2 agar cosine similarity = dot product
            db[person_name] = mean_emb / np.linalg.norm(mean_emb)
            print(f" Terdaftar: {person_name} ({len(embeddings)} foto)")
        else:
            print(f" [SKIP] Tidak ada wajah valid untuk {person_name}")

    with open(DB_PATH, "wb") as f:
        pickle.dump(db, f)
    print(f"\nDatabase disimpan ke {DB_PATH} ({len(db)} identitas)")
    return db

# ============================================================
# 3. MENGENALI GAMBAR TUNGGAL
# Mengembalikan gambar beranotasi + daftar tuple (nama, similarity)
# ============================================================
def recognize_image(img_path: str, db: dict):
    """
    Deteksi semua wajah di img_path, cocokkan masing-masing dengan database,
    dan gambar bounding boxes + label identitas pada gambar.
    """
    img = cv2.imread(img_path)
    if img is None:
        print(f"Tidak dapat membaca {img_path}")
        return None, []

    faces = app.get(img)
    results = []

    for face in faces:
        # Ekstrak bounding box
        x1, y1, x2, y2 = [int(v) for v in face.bbox]

        # Normalisasi query embedding
        query_emb = face.embedding / np.linalg.norm(face.embedding)

        # Hitung cosine similarity terhadap semua identitas yang dikenal
        best_name, best_sim = "Unknown", -1.0
        for name, ref_emb in db.items():
            sim = float(np.dot(query_emb, ref_emb)) # cosine similarity
            if sim > best_sim:
                best_sim, best_name = sim, name

        # Terapkan ambang batas: di bawah threshold = Unknown
        if best_sim < THRESHOLD:
            label = f"Unknown ({best_sim:.2f})"
            color = (0, 0, 220) # merah untuk unknown
        else:
            label = f"{best_name} ({best_sim:.2f})"
            color = (0, 200, 0) # hijau untuk dikenali

        # Gambar bounding box dan label pada gambar
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        cv2.putText(img, label, (x1, y1 - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)
        results.append((best_name, best_sim))

    return img, results

# ============================================================
# 4. PENGENALAN LIVE WEBCAM
# Tekan 'q' untuk keluar, 'r' untuk membangun ulang database
# ============================================================
def live_recognition(db: dict):
    """
    Buka webcam dan jalankan face recognition pada setiap frame.
    """
    cap = cv2.VideoCapture(0)
    print("Webcam dimulai. Tekan 'q' untuk keluar, 'r' untuk rebuild DB.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        faces = app.get(frame)
        for face in faces:
            x1, y1, x2, y2 = [int(v) for v in face.bbox]
            query_emb = face.embedding / np.linalg.norm(face.embedding)

            best_name, best_sim = "Unknown", -1.0
            for name, ref_emb in db.items():
                sim = float(np.dot(query_emb, ref_emb))
                if sim > best_sim:
                    best_sim, best_name = sim, name

            if best_sim < THRESHOLD:
                label = f"Unknown ({best_sim:.2f})"
                color = (0, 0, 220)
            else:
                label = f"{best_name} ({best_sim:.2f})"
                color = (0, 200, 0)

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, label, (x1, y1 - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)

        cv2.imshow("Face Recognition (tekan q untuk keluar)", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        elif key == ord("r"):
            print("Membangun ulang database...")
            db = build_database()

    cap.release()
    cv2.destroyAllWindows()

# ============================================================
# 5. MAIN
# ============================================================
if __name__ == "__main__":
    # Langkah 1: Bangun atau muat database
    if os.path.exists(DB_PATH):
        print(f"Memuat database yang ada dari {DB_PATH}")
        with open(DB_PATH, "rb") as f:
            db = pickle.load(f)
        print(f"Dimuat {len(db)} identitas: {list(db.keys())}")
    else:
        print("Membangun database baru dari folder known_faces/...")
        db = build_database()

    # Langkah 2: Pilih mode
    print("\nPilih mode:")
    print(" 1 - Uji pada file gambar tunggal")
    print(" 2 - Pengenalan live webcam")
    mode = input("Masukkan 1 atau 2: ").strip()

    if mode == "1":
        img_path = input("Masukkan path gambar: ").strip()
        result_img, results = recognize_image(img_path, db)
        if result_img is not None:
            print("Hasil:", results)
            cv2.imshow("Result", result_img)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
    elif mode == "2":
        live_recognition(db)
    else:
        print("Pilihan tidak valid.")