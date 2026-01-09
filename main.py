import tkinter as tk
from video_processor_app import VideoProcessorApp


def main():
    root = tk.Tk()
    app = VideoProcessorApp(root)

    # Обработка закрытия окна
    def on_closing():
        app.is_playing = False
        if app.cap:
            app.cap.release()
        root.destroy()

    root.protocol("WM_DELETE_WINDOW", on_closing)
    root.mainloop()


if __name__ == "__main__":
    main()