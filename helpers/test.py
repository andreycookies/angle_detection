import cv2, torch, time
from ultralytics import YOLO

device = "cuda"
model = YOLO(r"C:\Users\Администратор\Desktop\shittens\models\raw_model.pt").to(device)

cap = cv2.VideoCapture(r"C:\Users\Администратор\Desktop\record_090 - Trimss.mp4")

frames = []
for _ in range(100):
    ret, f = cap.read()
    if not ret:
        break
    frames.append(f)

cap.release()

t0 = time.time()
print(t0)
with torch.no_grad(), torch.amp.autocast("cuda"):
    results = model(
        frames,
        imgsz=960,        # 🔥 КРИТИЧНО
        device=device,
        stream=False,
        verbose=False
    )

print(time.time())
dt = time.time() - t0
print("FPS:", len(frames) / dt)
