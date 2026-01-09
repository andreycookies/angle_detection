import torch
import cv2
import numpy as np
from ultralytics import YOLO
import time
import json

def profile_detailed():
    """Детальное профилирование inference"""
    
    print("=" * 80)
    print("🔬 DETAILED INFERENCE PROFILING")
    print("=" * 80)
    
    device = torch.device('cuda')
    
    # Загрузка моделей
    print("\n📦 Loading models...")
    handled_model = YOLO('./models/handled_model.pt')
    raw_model = YOLO('./models/raw_model.pt')
    
    handled_model.to(device)
    raw_model.to(device)
    
    # Информация о моделях
    print(f"\n📊 Model info:")
    print(f"   Handled model parameters: {sum(p.numel() for p in handled_model.model.parameters()) / 1e6:.2f}M")
    print(f"   Raw model parameters: {sum(p.numel() for p in raw_model.model.parameters()) / 1e6:.2f}M")
    
    # Создаем тестовое изображение (реалистичный размер)
    test_image = np.random.randint(0, 255, (1080, 1920, 3), dtype=np.uint8)
    
    # Загружаем homography данные
    with open("../frames_homo.json", "r") as f:
        data = json.load(f)
    
    K = np.array(data["camera_matrix"], dtype=np.float64)
    H = np.array(data["homography_matrix"], dtype=np.float64)
    k1, k2 = 0.014, -0.037
    dist = np.array([k1, k2, 0, 0, 0], dtype=np.float64)
    
    print("\n" + "=" * 80)
    print("🧪 TEST 1: Full pipeline with homography (как в приложении)")
    print("=" * 80)
    
    for i in range(3):
        total_start = time.time()
        
        # CV операции
        cv_start = time.time()
        undistorted = cv2.undistort(test_image, K, dist)
        h, w = undistorted.shape[:2]
        
        corners = np.array([[0, 0], [w, 0], [w, h], [0, h]], dtype=np.float32).reshape(-1, 1, 2)
        transformed_corners = cv2.perspectiveTransform(corners, H)
        [xmin, ymin] = np.floor(transformed_corners.min(axis=0).ravel()).astype(int)
        [xmax, ymax] = np.ceil(transformed_corners.max(axis=0).ravel()).astype(int)
        
        tx, ty = -xmin if xmin < 0 else 0, -ymin if ymin < 0 else 0
        T = np.array([[1, 0, tx], [0, 1, ty], [0, 0, 1]], dtype=np.float64)
        H_inv_shifted = T @ H
        
        new_w = int(xmax - xmin)
        new_h = int(ymax - ymin)
        reprojected = cv2.warpPerspective(undistorted, H_inv_shifted, (new_w, new_h))
        cv_time = time.time() - cv_start
        
        print(f"\n🔄 Iteration {i+1}:")
        print(f"   CV ops: {cv_time*1000:.1f} ms")
        print(f"   Warped image size: {reprojected.shape}")
        
        # GPU Transfer + Preprocessing
        transfer_start = time.time()
        frame_tensor = torch.from_numpy(reprojected).to(device, non_blocking=True).float().div_(255.0)
        frame_tensor = frame_tensor.permute(2, 0, 1).unsqueeze(0).contiguous()
        
        h, w = frame_tensor.shape[2], frame_tensor.shape[3]
        new_h = (h + 31) // 32 * 32
        new_w = (w + 31) // 32 * 32
        
        frame_tensor = torch.nn.functional.interpolate(
            frame_tensor, size=(new_h, new_w), mode='bilinear', align_corners=False
        )
        torch.cuda.synchronize()
        transfer_time = time.time() - transfer_start
        
        print(f"   Transfer+Resize: {transfer_time*1000:.1f} ms")
        print(f"   Tensor size: {frame_tensor.shape}")
        
        # Handled model inference
        inference1_start = time.time()
        with torch.no_grad(), torch.amp.autocast('cuda'):
            results1 = handled_model(frame_tensor, verbose=False)
        torch.cuda.synchronize()
        inference1_time = time.time() - inference1_start
        
        print(f"   🔥 Handled model: {inference1_time*1000:.1f} ms")
        
        # Raw model на оригинальном изображении
        transfer2_start = time.time()
        image_tensor = torch.from_numpy(test_image).to(device, non_blocking=True).float().div_(255.0)
        image_tensor = image_tensor.permute(2, 0, 1).unsqueeze(0).contiguous()
        
        h, w = image_tensor.shape[2], image_tensor.shape[3]
        new_h = (h + 31) // 32 * 32
        new_w = (w + 31) // 32 * 32
        
        image_tensor = torch.nn.functional.interpolate(
            image_tensor, size=(new_h, new_w), mode='bilinear', align_corners=False
        )
        torch.cuda.synchronize()
        transfer2_time = time.time() - transfer2_start
        
        print(f"   Transfer+Resize #2: {transfer2_time*1000:.1f} ms")
        
        # Raw model inference
        inference2_start = time.time()
        with torch.no_grad(), torch.amp.autocast('cuda'):
            results2 = raw_model(image_tensor, verbose=False)
        torch.cuda.synchronize()
        inference2_time = time.time() - inference2_start
        
        print(f"   🔥 Raw model: {inference2_time*1000:.1f} ms")
        
        total_time = time.time() - total_start
        print(f"   ⏱️  TOTAL: {total_time*1000:.1f} ms ({1/total_time:.1f} FPS)")
    
    print("\n" + "=" * 80)
    print("🧪 TEST 2: Только inference без homography")
    print("=" * 80)
    
    # Простой тест на стандартном размере
    simple_tensor = torch.randn(1, 3, 640, 640, device=device)
    
    for i in range(5):
        start = time.time()
        with torch.no_grad(), torch.amp.autocast('cuda'):
            _ = handled_model(simple_tensor, verbose=False)
        torch.cuda.synchronize()
        t = time.time() - start
        print(f"   Run {i+1}: {t*1000:.1f} ms ({1/t:.1f} FPS)")
    
    print("\n" + "=" * 80)
    print("🧪 TEST 3: Без mixed precision")
    print("=" * 80)
    
    for i in range(3):
        start = time.time()
        with torch.no_grad():
            _ = handled_model(simple_tensor, verbose=False)
        torch.cuda.synchronize()
        t = time.time() - start
        print(f"   Run {i+1}: {t*1000:.1f} ms ({1/t:.1f} FPS)")
    
    print("\n" + "=" * 80)
    print("💾 GPU Memory Usage")
    print("=" * 80)
    print(f"   Allocated: {torch.cuda.memory_allocated(0) / 1e9:.2f} GB")
    print(f"   Reserved:  {torch.cuda.memory_reserved(0) / 1e9:.2f} GB")
    print(f"   Max allocated: {torch.cuda.max_memory_allocated(0) / 1e9:.2f} GB")
    
    print("\n" + "=" * 80)
    print("💡 RECOMMENDATIONS")
    print("=" * 80)
    
    # Анализируем результаты
    if inference1_time > 0.5:  # >500ms
        print("⚠️  Handled model inference is VERY SLOW (>500ms)")
        print("   Possible causes:")
        print("   - Model is too large (check model size)")
        print("   - Input resolution too high (try reducing)")
        print("   - Model not properly compiled for GPU")
        print("\n   💡 Try:")
        print("   - Use smaller YOLO model (yolov8n.pt instead of yolov8x.pt)")
        print("   - Reduce input resolution before inference")
        print("   - Check if model is quantized or has issues")
    
    if transfer_time > 0.05:  # >50ms
        print("⚠️  CPU->GPU transfer is slow")
        print("   💡 This is normal for large images, but can be optimized")

if __name__ == "__main__":
    profile_detailed()