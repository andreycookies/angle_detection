import datetime
import json
from collections import deque
import torch
import cv2
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from PIL import Image, ImageTk
import threading
import queue
import time
from ultralytics import YOLO
import numpy as np
import math
from homo import mask_preprocessing, has_no_ones, detect_sheet_angle_no_homography


class VideoProcessorApp:
    def __init__(self, root):
        self.root = root
        self.root.title("YOLO Video Processor")
        self.root.geometry("1200x800")

        self.video_source = None
        self.video_path = None
        self.rtsp_url = None
        self.cap = None
        self.is_playing = False
        self.current_frame = None
        self.handled_model = None
        self.raw_model = None
        self.processing_queue = queue.Queue(maxsize=1)
        self.display_queue = queue.Queue(maxsize=1)
        self.new_video = []
        self.display_fps_5s = 0.0
        self.inference_times = deque(maxlen=30)  # Для отслеживания времени inference
        
        # 🔥 GPU оптимизации
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"🚀 Using device: {self.device}")
        
        if torch.cuda.is_available():
            print(f"   GPU: {torch.cuda.get_device_name(0)}")
            print(f"   Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
            torch.backends.cudnn.benchmark = True  # Автооптимизация cuDNN
            torch.cuda.empty_cache()
        else:
            print("⚠️  CUDA not available, using CPU")
        
        json_path = r".\video_homo.json"
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            H = np.array(data["homography_matrix"], dtype=np.float64)
        
        self.H = H
        
        # Загрузка модели YOLO
        self.load_yolo_model()

        # Создание интерфейса
        self.create_widgets()

    def connect_rtsp_camera(self):
        self.ask_rtsp_url()

    def disconnect_source(self):
        """Отключение текущего источника"""
        self.stop_playback()

        if self.cap:
            self.cap.release()
            self.cap = None

        self.video_source = None
        self.video_path = None
        self.rtsp_url = None
        self.source_info_label.config(text="No video source connected")
        self.play_button.config(state=tk.DISABLED)
        self.fps_label.config(text="FPS: 0")

        self.video_canvas.delete("all")
        self.stats_text.delete(1.0, tk.END)

    def stop_camera(self):
        """Полная остановка камеры"""
        self.stop_processing()
        if self.cap:
            self.cap.release()
            self.cap = None
        self.play_button.config(state=tk.DISABLED)

    def show_settings(self):
        """Показать настройки камеры"""
        settings_window = tk.Toplevel(self.root)
        settings_window.title("Camera Settings")
        settings_window.geometry("400x300")

        ttk.Label(settings_window, text="RTSP URL:").pack(pady=(10, 5))
        url_entry = ttk.Entry(settings_window, width=50)
        url_entry.pack(pady=5)
        if self.rtsp_url:
            url_entry.insert(0, self.rtsp_url)

        def save_settings():
            new_url = url_entry.get()
            if new_url and new_url != self.rtsp_url:
                self.rtsp_url = new_url
                if self.is_playing:
                    self.stop_processing()
                if self.cap:
                    self.cap.release()
                self.connect_camera()
            settings_window.destroy()

        ttk.Button(settings_window, text="Save", command=save_settings).pack(pady=10)

    def connect_camera(self):
        """Подключение к камере"""
        if not self.rtsp_url:
            return

        try:
            self.cap = cv2.VideoCapture(self.rtsp_url)
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            self.cap.set(cv2.CAP_PROP_FPS, 30)
            self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'H264'))

            if self.cap.isOpened():
                ret, frame = self.cap.read()
                if ret:
                    self.play_button.config(state=tk.NORMAL)
                    self.current_frame = frame
                    self.display_frame(frame)
                    self.start_processing()
                else:
                    messagebox.showerror("Error", "Connected but cannot receive frames")
                    self.cap.release()
                    self.cap = None
            else:
                messagebox.showerror("Error", f"Cannot connect to camera:\n{self.rtsp_url}")

        except Exception as e:
            messagebox.showerror("Error", f"Connection failed: {str(e)}")

    def ask_rtsp_url(self):
        """Запрос RTSP URL у пользователя"""
        self.rtsp_url = tk.simpledialog.askstring(
            "RTSP Connection",
            "Enter RTSP URL for Hiwatch camera:\n\n" +
            "Common formats:\n" +
            "rtsp://username:password@IP:554/stream\n" +
            "rtsp://IP:554/user=username_password=password_channel=1_stream=0.sdp\n\n" +
            "Example: rtsp://admin:12345@192.168.1.108:554/Streaming/Channels/101",
            initialvalue="rtsp://admin:12345@192.168.1.108:554/Streaming/Channels/101"
        )

        if self.rtsp_url:
            self.connect_camera()
        else:
            messagebox.showwarning("Warning", "No RTSP URL provided.")

    def load_yolo_model(self):
        """Загрузка модели YOLO с оптимизацией"""
        try:
            print("📦 Loading YOLO models...")
            
            # Загружаем модели
            self.handled_model = YOLO('./models/handled_model.pt')
            self.raw_model = YOLO('./models/raw_model.pt')
            
            # Переносим на GPU
            self.handled_model.to(self.device)
            self.raw_model.to(self.device)
            
            # Проверяем, что модели действительно на GPU
            if self.device.type == 'cuda':
                first_param = next(self.handled_model.model.parameters())
                print(f"✅ Models loaded on: {first_param.device}")
            
            # 🔥 КРИТИЧНО: Прогрев моделей (первый inference всегда медленный)
            print("🔥 Warming up models (compiling CUDA kernels)...")
            # Используем размеры кратные 32
            dummy_input = torch.randn(1, 3, 640, 640, device=self.device)
            
            with torch.no_grad():
                _ = self.handled_model.model(dummy_input)
                _ = self.raw_model.model(dummy_input)
            
            # Очистка кэша после прогрева
            if self.device.type == 'cuda':
                torch.cuda.synchronize()  # Ждем завершения всех операций
                torch.cuda.empty_cache()
                print(f"✅ GPU memory allocated: {torch.cuda.memory_allocated(0) / 1e9:.2f} GB")
            
            print("✅ Models ready for inference")
            
        except Exception as e:
            print(f"❌ Failed to load YOLO models: {e}")
            import traceback
            traceback.print_exc()
            messagebox.showerror("Error", f"Failed to load YOLO model: {str(e)}")

    def create_widgets(self):
        """Создание элементов интерфейса"""
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(1, weight=1)
        main_frame.rowconfigure(1, weight=1)

        source_frame = ttk.LabelFrame(main_frame, text="Video Source", padding="10")
        source_frame.grid(row=0, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(0, 10))

        source_buttons_frame = ttk.Frame(source_frame)
        source_buttons_frame.pack(fill=tk.X)

        ttk.Button(source_buttons_frame, text="Open Video File",
                   command=self.open_video_file, width=15).pack(side=tk.LEFT, padx=(0, 10))
        ttk.Button(source_buttons_frame, text="Connect RTSP Camera",
                   command=self.connect_rtsp_camera, width=15).pack(side=tk.LEFT, padx=(0, 10))
        ttk.Button(source_buttons_frame, text="Disconnect",
                   command=self.disconnect_source, width=10).pack(side=tk.LEFT)

        self.source_info_label = ttk.Label(source_frame, text="No video source connected")
        self.source_info_label.pack(fill=tk.X, pady=(5, 0))

        control_frame = ttk.LabelFrame(main_frame, text="Playback Control", padding="10")
        control_frame.grid(row=1, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(0, 10))

        control_buttons_frame = ttk.Frame(control_frame)
        control_buttons_frame.pack(fill=tk.X)

        self.play_button = ttk.Button(source_buttons_frame, text="Start",
                                      command=self.toggle_play, width=10)
        self.play_button.pack(side=tk.LEFT, padx=(0, 10))

        ttk.Button(source_buttons_frame, text="Stop",
                   command=self.stop_playback, width=10).pack(side=tk.LEFT, padx=(0, 10))

        self.progress_frame = ttk.Frame(control_buttons_frame)
        self.progress_frame.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(20, 0))

        self.progress_var = tk.DoubleVar()
        self.progress_bar = ttk.Progressbar(self.progress_frame, variable=self.progress_var, maximum=100)
        self.progress_bar.pack(side=tk.LEFT, fill=tk.X, expand=True)

        self.progress_label = ttk.Label(self.progress_frame, text="0%", width=5)
        self.progress_label.pack(side=tk.RIGHT, padx=(5, 0))

        stats_frame = ttk.Frame(control_buttons_frame)
        stats_frame.pack(side=tk.RIGHT)

        self.fps_label = ttk.Label(stats_frame, text="FPS: 0")
        self.fps_label.pack(side=tk.RIGHT, padx=(10, 0))

        settings_frame = ttk.Frame(control_buttons_frame)
        settings_frame.pack(side=tk.RIGHT, padx=(20, 0))

        video_frame = ttk.LabelFrame(main_frame, text="Video Output", padding="5")
        video_frame.grid(row=3, column=0, columnspan=2, sticky=(tk.W, tk.E, tk.N, tk.S), pady=(0, 10))
        video_frame.columnconfigure(0, weight=1)
        video_frame.rowconfigure(0, weight=1)

        self.video_canvas = tk.Canvas(video_frame, bg='black', width=800, height=600)
        self.video_canvas.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

        stats_frame = ttk.LabelFrame(main_frame, text="Detection Statistics", padding="5")
        stats_frame.grid(row=2, column=0, columnspan=2, sticky=(tk.W, tk.E))

        self.stats_text = tk.Text(stats_frame, height=6, width=80)
        scrollbar = ttk.Scrollbar(stats_frame, orient="vertical", command=self.stats_text.yview)
        self.stats_text.configure(yscrollcommand=scrollbar.set)
        self.stats_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        self.processing_thread = None
        self.display_thread = None

    def open_video_file(self):
        """Открытие видеофайла"""
        file_path = filedialog.askopenfilename(
            title="Select Video File",
            filetypes=[
                ("Video files", "*.mp4 *.avi *.mov *.mkv *.wmv *.flv"),
                ("All files", "*.*")
            ]
        )

        if file_path:
            self.disconnect_source()
            self.video_source = 'file'
            self.video_path = file_path

            try:
                self.cap = cv2.VideoCapture(file_path)

                if self.cap.isOpened():
                    fps = self.cap.get(cv2.CAP_PROP_FPS)
                    frame_count = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
                    duration = frame_count / fps if fps > 0 else 0
                    width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                    height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

                    info_text = (f"File: {file_path.split('/')[-1]} | "
                                 f"Size: {width}x{height} | FPS: {fps:.1f} | "
                                 f"Frames: {frame_count} | Duration: {duration:.1f}s")
                    self.source_info_label.config(text=info_text)

                    self.play_button.config(state=tk.NORMAL)
                    self.progress_var.set(0)
                    self.show_first_frame()
                else:
                    messagebox.showerror("Error", "Could not open video file")

            except Exception as e:
                messagebox.showerror("Error", f"Error opening video file: {str(e)}")

    def show_first_frame(self):
        """Показать первый кадр видео"""
        if self.cap:
            ret, frame = self.cap.read()
            if ret:
                self.current_frame = frame
                self.display_frame(frame)
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    def toggle_play(self):
        """Запуск/пауза воспроизведения"""
        if not self.is_playing:
            self.start_processing()
        else:
            self.stop_processing()

    def start_processing(self):
        if not self.cap:
            return

        self.is_playing = True
        self.play_button.config(text="Pause")

        self.processing_thread = threading.Thread(target=self.process_video, daemon=True)
        self.yolo_thread = threading.Thread(target=self.update_processing, daemon=True)
        self.display_thread = threading.Thread(target=self.update_display, daemon=True)

        self.processing_thread.start()
        self.yolo_thread.start()
        self.display_thread.start()

    def stop_processing(self):
        """Остановка обработки видео"""
        self.is_playing = False
        self.play_button.config(text="Play")

    def stop_video(self):
        """Полная остановка видео"""
        self.stop_processing()
        if self.cap:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            self.show_first_frame()

    def stop_playback(self):
        """Полная остановка воспроизведения"""
        self.stop_processing()
        if self.cap and self.video_source == 'file':
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            self.show_first_frame()
        self.progress_var.set(0)
        self.progress_label.config(text="0%")

    def process_video(self):
        """Обработка видео в отдельном потоке"""
        fps = self.cap.get(cv2.CAP_PROP_FPS)
        frame_delay = 1.0 / fps if fps > 0 else 0.033

        frame_times = deque()
        window_sec = 5.0

        while self.is_playing and self.cap:
            loop_start = time.time()

            ret, frame = self.cap.read()
            if not ret:
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue

            now = time.time()
            frame_times.append(now)

            while frame_times and (now - frame_times[0]) > window_sec:
                frame_times.popleft()

            # Выбрасываем старые кадры из очереди
            while self.processing_queue.full():
                try:
                    self.processing_queue.get_nowait()
                except queue.Empty:
                    break

            try:
                self.processing_queue.put_nowait(frame)
            except queue.Full:
                pass

            processing_time = time.time() - loop_start
            sleep_time = max(0, frame_delay - processing_time)
            time.sleep(sleep_time)

    def process_frame_with_yolo(self, frame):
        if self.handled_model is None or self.raw_model is None:
            return frame

        try:
            # ИСХОДНЫЙ КАДР
            image = frame.copy()
            #cv2.imwrite("img/image.png", image)

            warped_image = self.warp_with_homography_keep_size(image, self.H)
            cv2.imwrite("img/warped_image.png", warped_image)

            with torch.no_grad(), torch.amp.autocast("cuda"):
                results = self.handled_model(
                    warped_image,
                    imgsz=960,
                    device=self.device,
                    verbose=False
                )

            if not results or results[0].masks is None:
                return image

            masks = results[0].masks.data.detach().cpu().numpy()

            if masks.shape[0] != 1:
                return image

            # ИСХОДНАЯ МАСКА
            mask = masks[0]
            binary_mask = (mask > 0).astype(np.uint8)
            cv2.imwrite("img/mask.png", binary_mask * 255)


            # ОБРАБОТАННАЯ МАСКА
            m = mask_preprocessing(binary_mask)
            #cv2.imwrite("img/processed_mask.png", m * 255)

            if has_no_ones(m):
                return image

            # ИЗМЕНЕННАЯ МАСКА
            h, w = image.shape[:2]
            resized_mask = cv2.resize(m, (w, h), interpolation=cv2.INTER_NEAREST)

            
            angle = detect_sheet_angle_no_homography(resized_mask)

            self.update_statistics(angle, self.display_fps_5s)

            # ===== RAW MODEL =====
            with torch.no_grad(), torch.amp.autocast("cuda"):
                raw_results = self.raw_model(
                    image,
                    imgsz=960,
                    device=self.device,
                    verbose=False
                )

            if not raw_results or raw_results[0].boxes is None:
                return image

            # ❗ ВАЖНО: clone / copy
            boxes = raw_results[0].boxes.xyxy.detach().cpu().numpy().copy()

            for box in boxes:
                x1, y1, x2, y2 = map(int, box)
                cv2.rectangle(image, (x1, y1), (x2, y2), (0, 0, 255), 5)

            # BOXED КАДР
            #cv2.imwrite("img/boxed_image.png", image)
            return image

        except Exception as e:
            print(f"❌ process_frame_with_yolo error: {e}")
            return frame

    # def update_statistics(self, angle, fps, inference_time, cv_time=0, gpu1_time=0, gpu2_time=0):
    #     text = (f"Display FPS: {fps:.1f}\n"
    #             f"Inference: {inference_time*1000:.1f} ms ({1/inference_time:.1f} FPS)\n"
    #             f"  CV ops: {cv_time*1000:.1f} ms\n"
    #             f"  GPU #1: {gpu1_time*1000:.1f} ms\n"
    #             f"  GPU #2: {gpu2_time*1000:.1f} ms\n"
    #             f"Угол отклонения: {int(angle)}°")
    #     self.root.after(0, lambda: self._update_stats_display(text))

    def update_statistics(self, angle, fps):
        text = (f"Display FPS: {fps:.1f}\n"
                f"Угол отклонения: {int(angle)}°")
        self.root.after(0, lambda: self._update_stats_display(text))

    def _update_stats_display(self, text):
        """Обновление отображения статистики"""
        self.stats_text.delete(1.0, tk.END)
        self.stats_text.insert(1.0, text)

    def update_display(self):
        """Обновление отображения в отдельном потоке"""
        while self.is_playing:
            try:
                frame = self.display_queue.get(timeout=0.1)
                self.display_frame(frame)
            except queue.Empty:
                continue

    def display_frame(self, frame):
        """Отображение кадра на Canvas"""
        try:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            h, w = frame_rgb.shape[:2]
            canvas_width = self.video_canvas.winfo_width()
            canvas_height = self.video_canvas.winfo_height()

            if canvas_width <= 1:
                canvas_width = 800
            if canvas_height <= 1:
                canvas_height = 600

            scale_x = canvas_width / w
            scale_y = canvas_height / h
            scale = min(scale_x, scale_y, 1.0)

            new_w = int(w * scale)
            new_h = int(h * scale)

            if new_w != w or new_h != h:
                frame_resized = cv2.resize(frame_rgb, (new_w, new_h))
            else:
                frame_resized = frame_rgb

            pil_image = Image.fromarray(frame_resized)
            tk_image = ImageTk.PhotoImage(pil_image)

            self.video_canvas.delete("all")
            self.video_canvas.create_image(
                canvas_width // 2,
                canvas_height // 2,
                image=tk_image,
                anchor=tk.CENTER
            )

            self.video_canvas.tk_image = tk_image

        except Exception as e:
            print(f"Error displaying frame: {str(e)}")

    def update_processing(self):
        frame_times = deque()
        window = 5.0

        while self.is_playing:
            try:
                frame = self.processing_queue.get(timeout=0.1)
                # Берём только последний кадр
                while not self.processing_queue.empty():
                    frame = self.processing_queue.get_nowait()

            except queue.Empty:
                continue

            processed = self.process_frame_with_yolo(frame)

            now = time.time()
            frame_times.append(now)

            while frame_times and now - frame_times[0] > window:
                frame_times.popleft()

            self.display_fps_5s = len(frame_times) / window

            # Добавляем в очередь отображения
            if not self.display_queue.full():
                self.display_queue.put(processed)
            else:
                try:
                    self.display_queue.get_nowait()
                except:
                    pass
                self.display_queue.put(processed)

    def warp_with_homography_keep_size(
            self,
            image: np.ndarray,
            H: np.ndarray,
            interpolation=cv2.INTER_LINEAR
    ) -> np.ndarray:
        """
        Применяет гомографию H так, чтобы:
        - размер изображения остался прежним
        - всё преобразованное содержимое было видно
        - допускается масштабирование (потеря качества)

        image: входное изображение (H, W, C)
        H: 3x3 матрица гомографии
        """

        h, w = image.shape[:2]

        # 1. Углы исходного изображения
        corners = np.array([
            [0, 0],
            [w - 1, 0],
            [w - 1, h - 1],
            [0, h - 1]
        ], dtype=np.float32)

        corners = corners.reshape(-1, 1, 2)

        # 2. Применяем гомографию к углам
        warped_corners = cv2.perspectiveTransform(corners, H)

        xs = warped_corners[:, 0, 0]
        ys = warped_corners[:, 0, 1]

        min_x, max_x = xs.min(), xs.max()
        min_y, max_y = ys.min(), ys.max()

        warped_width = max_x - min_x
        warped_height = max_y - min_y

        # 3. Масштаб, чтобы всё влезло в исходный размер
        scale = min(w / warped_width, h / warped_height)

        # 4. Центрирование
        tx = -min_x * scale + (w - warped_width * scale) / 2
        ty = -min_y * scale + (h - warped_height * scale) / 2

        # 5. Матрица "вписывания"
        fit_matrix = np.array([
            [scale, 0, tx],
            [0, scale, ty],
            [0, 0, 1]
        ], dtype=np.float64)

        # 6. Итоговая гомография
        H_final = fit_matrix @ H

        # 7. Warp с сохранением размера
        warped = cv2.warpPerspective(
            image,
            H_final,
            (w, h),
            flags=interpolation,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0
        )

        return warped

    def __del__(self):
        """Освобождение ресурсов"""
        if self.cap:
            self.cap.release()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()