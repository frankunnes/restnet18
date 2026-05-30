import cv2
import torch
import torchvision.transforms as transforms
import torchvision.models as models
import torch.nn as nn
from PIL import Image
import time

# ==========================================
# 1. KONFIGURASI
# ==========================================
MODEL_PATH = './hasil_resnet18_project/resnet18_banknote_complete.pth'
CLASSES = ['1000', '10000', '100000', '2000', '20000', '5000', '50000']
IMG_SIZE = 224

# ==========================================
# 2. SETUP MODEL & DEVICE
# ==========================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Menggunakan device: {device}")

print(f"Memuat model dari {MODEL_PATH}...")
try:
    # Memuat arsitektur ResNet-18 dan bobot yang sudah disave (state_dict)
    model = models.resnet18(weights=None)
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, len(CLASSES))
    
    checkpoint = torch.load(MODEL_PATH, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval() # Set mode evaluasi (penting agar dropout/batchnorm statis)
    print("Model berhasil dimuat!\n")
except Exception as e:
    print(f"Gagal memuat model! Pastikan path benar. Error: {e}")
    exit()

# Transformasi gambar (Harus SAMA PERSIS dengan saat training)
transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# ==========================================
# 3. WEBCAM REAL-TIME INFERENCE
# ==========================================
cap = cv2.VideoCapture(0) # 0 adalah ID kamera utama (webcam laptop)

if not cap.isOpened():
    print("Tidak dapat membuka webcam. Pastikan tidak sedang digunakan aplikasi lain.")
    exit()

print("="*50)
print("KAMERA AKTIF! Arahkan uang kertas Rupiah ke kamera.")
print("Tekan tombol 'q' pada keyboard untuk keluar.")
print("="*50)

prev_time = 0

while True:
    ret, frame = cap.read()
    if not ret:
        print("Gagal menangkap frame dari webcam.")
        break
        
    # Hitung FPS (Frame per Second)
    current_time = time.time()
    fps = 1 / (current_time - prev_time) if prev_time > 0 else 0
    prev_time = current_time

    # -------------------------------------
    # Preprocessing Frame
    # -------------------------------------
    # BGR (OpenCV) -> RGB (PyTorch/PIL)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(rgb_frame)
    
    # Apply transform & tambah batch dimension (C,H,W) -> (1,C,H,W)
    input_tensor = transform(pil_img).unsqueeze(0).to(device)
    
    # -------------------------------------
    # Prediksi
    # -------------------------------------
    with torch.no_grad():
        outputs = model(input_tensor)
        # Ubah output raw (logits) jadi probabilitas (persentase)
        probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
        
        # Ambil tebakan paling meyakinkan
        max_prob, predicted_idx = torch.max(probabilities, 0)
        
        confidence = max_prob.item() * 100
        predicted_class = CLASSES[predicted_idx.item()]

    # -------------------------------------
    # Tampilkan Hasil di Layar
    # -------------------------------------
    # Warna Hijau jika yakin > 85%, jika tidak Merah
    color = (0, 255, 0) if confidence > 85 else (0, 0, 255)
    
    text_label = f"Uang: Rp {predicted_class}"
    text_conf = f"Kepercayaan: {confidence:.2f}%"
    text_fps = f"FPS: {int(fps)}"
    
    # Gambar background hitam transparan agar teks mudah dibaca
    cv2.rectangle(frame, (10, 10), (350, 110), (0, 0, 0), -1)
    
    # Tulis Teks ke Layar
    cv2.putText(frame, text_label, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
    cv2.putText(frame, text_conf, (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    cv2.putText(frame, text_fps, (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    # Munculkan Window GUI
    cv2.imshow('Uji Coba ResNet-18 Realtime', frame)

    # Cek input keyboard (tekan 'q' untuk quit)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Tutup kamera dan window
cap.release()
cv2.destroyAllWindows()
