import torch
import cv2
import numpy as np
from ultralytics import YOLO
import time


def test_gpu_performance():
    """Тестирование производительности GPU vs CPU"""
    
    print("=" * 60)
    print("🔍 GPU DIAGNOSTICS")
    print("=" * 60)
    
    # 1. Базовая информация
    print(f"\n✅ PyTorch version: {torch.__version__}")
    print(f"✅ CUDA available: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        print(f"✅ CUDA version: {torch.version.cuda}")
        print(f"✅ GPU device: {torch.cuda.get_device_name(0)}")
        print(f"✅ GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    else:
        print("❌ CUDA not available!")
        return
    
    # 2. Проверка загрузки модели на GPU
    print("\n" + "=" * 60)
    print("📦 LOADING YOLO MODELS")
    print("=" * 60)
    
    try:
        handled_model = YOLO('./models/handled_model.pt')
        raw_model = YOLO('./models/raw_model.pt')
        
        # Явно переносим на GPU
        device = torch.device('cuda')
        handled_model.to(device)
        raw_model.to(device)
        
        print("✅ Models loaded successfully")
        
        # Проверяем, на каком устройстве параметры модели
        first_param = next(handled_model.model.parameters())
        print(f"✅ Model device: {first_param.device}")
        
    except Exception as e:
        print(f"❌ Failed to load models: {e}")
        return
    
    # 3. Создаем тестовое изображение
    print("\n" + "=" * 60)
    print("🎬 PERFORMANCE TEST")
    print("=" * 60)
    
    # Генерируем случайное изображение
    test_image = np.random.randint(0, 255, (1080, 1920, 3), dtype=np.uint8)
    
    # 4. Тест CPU inference
    print("\n🐌 Testing CPU inference...")
    handled_model.to('cpu')
    
    start = time.time()
    for i in range(3):
        frame_tensor = torch.from_numpy(test_image).float() / 255.0
        frame_tensor = frame_tensor.permute(2, 0, 1).unsqueeze(0)  # CPU tensor
        
        with torch.no_grad():
            _ = handled_model(frame_tensor, verbose=False, device='cpu')
    
    cpu_time = (time.time() - start) / 3
    print(f"   Average CPU time: {cpu_time*1000:.1f} ms ({1/cpu_time:.1f} FPS)")
    
    # 5. Тест GPU inference (НЕПРАВИЛЬНЫЙ способ)
    print("\n⚠️  Testing GPU inference (bad way - creating tensor on CPU)...")
    handled_model.to(device)
    
    start = time.time()
    for i in range(10):
        frame_tensor = torch.from_numpy(test_image).float() / 255.0
        frame_tensor = frame_tensor.permute(2, 0, 1).unsqueeze(0)
        frame_tensor = frame_tensor.to(device)  # ❌ CPU -> GPU копирование!
        
        with torch.no_grad(), torch.amp.autocast('cuda'):
            _ = handled_model(frame_tensor, verbose=False)
    
    gpu_time_bad = (time.time() - start) / 10
    print(f"   Average GPU time: {gpu_time_bad*1000:.1f} ms ({1/gpu_time_bad:.1f} FPS)")
    print(f"   ⚠️  This includes CPU->GPU transfer overhead!")
    
    # 6. Тест GPU inference (ПРАВИЛЬНЫЙ способ)
    print("\n🚀 Testing GPU inference (good way - creating tensor on GPU)...")
    
    start = time.time()
    for i in range(10):
        # Создаем тензор сразу на GPU
        frame_tensor = torch.from_numpy(test_image).to(device, non_blocking=True).float().div_(255.0)
        frame_tensor = frame_tensor.permute(2, 0, 1).unsqueeze(0).contiguous()
        
        with torch.no_grad(), torch.amp.autocast('cuda'):
            _ = handled_model(frame_tensor, verbose=False)
    
    gpu_time_good = (time.time() - start) / 10
    print(f"   Average GPU time: {gpu_time_good*1000:.1f} ms ({1/gpu_time_good:.1f} FPS)")
    
    # 7. Сравнение
    print("\n" + "=" * 60)
    print("📊 RESULTS SUMMARY")
    print("=" * 60)
    print(f"CPU:              {cpu_time*1000:6.1f} ms ({1/cpu_time:5.1f} FPS)")
    print(f"GPU (bad way):    {gpu_time_bad*1000:6.1f} ms ({1/gpu_time_bad:5.1f} FPS) - Speedup: {cpu_time/gpu_time_bad:.1f}x")
    print(f"GPU (good way):   {gpu_time_good*1000:6.1f} ms ({1/gpu_time_good:5.1f} FPS) - Speedup: {cpu_time/gpu_time_good:.1f}x")
    print(f"\n💡 Optimization gain: {gpu_time_bad/gpu_time_good:.1f}x faster with proper GPU usage!")
    
    # 8. Проверка памяти GPU
    print("\n" + "=" * 60)
    print("💾 GPU MEMORY USAGE")
    print("=" * 60)
    print(f"Allocated: {torch.cuda.memory_allocated(0) / 1e9:.2f} GB")
    print(f"Reserved:  {torch.cuda.memory_reserved(0) / 1e9:.2f} GB")
    
    # 9. Рекомендации
    print("\n" + "=" * 60)
    print("💡 RECOMMENDATIONS")
    print("=" * 60)
    
    if gpu_time_good < cpu_time * 0.5:
        print("✅ GPU acceleration is working correctly!")
    else:
        print("⚠️  GPU is not providing expected speedup. Possible issues:")
        print("   - GPU drivers may need update")
        print("   - CUDA toolkit version mismatch")
        print("   - Model size too small to benefit from GPU")
    
    if gpu_time_bad > gpu_time_good * 1.2:
        print("⚠️  CPU->GPU transfer is a bottleneck!")
        print("   👉 Make sure to create tensors directly on GPU:")
        print("      torch.from_numpy(data).to(device).float()")
        print("   👉 NOT: torch.from_numpy(data).float().to(device)")

if __name__ == "__main__":
    test_gpu_performance()