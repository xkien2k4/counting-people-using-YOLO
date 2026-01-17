import cv2
import numpy as np
from ultralytics import YOLO
from sort import Sort
import tkinter as tk
from tkinter import filedialog, messagebox
from threading import Thread

class PeopleCounterApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Hệ thống đếm người ra vào")
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)

        self.in_count = 0
        self.out_count = 0
        self.memory = {}
        self.running = False
        self.video_path = None
        self.model = None
        self.tracker = Sort(max_age=15, min_hits=2, iou_threshold=0.3)

        tk.Button(root, text="Chọn video", command=self.choose_video).pack(pady=5)
        tk.Button(root, text="Bắt đầu", command=self.start).pack(pady=5)
        tk.Button(root, text="Dừng", command=self.stop).pack(pady=5)

        self.status = tk.Label(root, text="Không video nào được chọn", font=("Arial", 12))
        self.status.pack(pady=5)

        self.label = tk.Label(root, text="IN: 0 | OUT: 0", font=("Arial", 16))
        self.label.pack(pady=10)

    def choose_video(self):
        self.video_path = filedialog.askopenfilename(
            title="Select Video File",
            filetypes=[("Video Files", "*.mp4 *.avi *.mov")]
        )
        if self.video_path:
            self.status.config(text=f"Đã chọn: {self.video_path}")
            self.in_count = 0
            self.out_count = 0
            self.memory.clear()
            self.label.config(text="IN: 0 | OUT: 0")

    def load_model(self):
        if self.model is None:
            self.model = YOLO("yolov8s.pt")

    def start(self):
        if self.video_path and not self.running:
            self.running = True
            Thread(target=self.run_detection, daemon=True).start()
        elif not self.video_path:
            messagebox.showwarning("Cảnh báo", "Vui lòng chọn video trước!")

    def stop(self):
        self.running = False

    def run_detection(self):
        self.load_model()
        cap = cv2.VideoCapture(self.video_path)

        if not cap.isOpened():
            messagebox.showerror("Lỗi", "Không thể mở video!")
            return

        line_y = 300
        active_ids = {}
        crossed_ids = set()

        while self.running:
            ret, frame = cap.read()
            if not ret:
                break

            results = self.model(frame, verbose=False)[0]
            detections = []

            for box in results.boxes:
                if int(box.cls[0]) == 0:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    conf = float(box.conf[0])
                    detections.append([x1, y1, x2, y2, conf])

            tracked = self.tracker.update(np.array(detections))

            cv2.line(frame, (0, line_y), (frame.shape[1], line_y), (0, 255, 0), 2)

            current_ids = []

            for x1, y1, x2, y2, obj_id in tracked:
                obj_id = int(obj_id)
                current_ids.append(obj_id)
                cx = int((x1 + x2) / 2)
                cy = int((y1 + y2) / 2)

                if obj_id not in active_ids:
                    active_ids[obj_id] = cy

                if obj_id not in crossed_ids:
                    if active_ids[obj_id] < line_y <= cy:
                        self.in_count += 1
                        crossed_ids.add(obj_id)

                    elif active_ids[obj_id] > line_y >= cy:
                        self.out_count += 1
                        crossed_ids.add(obj_id)

                active_ids[obj_id] = cy

                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                cv2.putText(frame, f"ID:{obj_id}", (int(x1), int(y1) - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

            for tid in list(active_ids.keys()):
                if tid not in current_ids:
                    del active_ids[tid]

            cv2.putText(frame, f"IN: {self.in_count}", (10, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)
            cv2.putText(frame, f"OUT: {self.out_count}", (10, 80),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)

            self.label.config(text=f"IN: {self.in_count} | OUT: {self.out_count}")
            cv2.imshow("People Counter Video", frame)

            if cv2.waitKey(1) & 0xFF == 27:
                break

        cap.release()
        cv2.destroyAllWindows()
        self.running = False

    def on_close(self):
        self.running = False
        cv2.destroyAllWindows()
        self.root.destroy()


if __name__ == "__main__":
    root = tk.Tk()
    app = PeopleCounterApp(root)
    root.mainloop()
