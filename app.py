import torch
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

import sys
import cv2
import numpy as np
import albumentations as A
import time

from huggingface_hub import hf_hub_download
from ultralytics import YOLO
from PyQt5.QtWidgets import QApplication, QWidget, QLabel, QPushButton, QVBoxLayout, QHBoxLayout, QCheckBox, QFrame, QSizePolicy
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtCore import QThread, pyqtSignal, Qt
from albumentations.pytorch import ToTensorV2

from src.models.simple_resnet import FaceLandmarkModel
from src.kalman_filter import KalmanFilter
from src.utils.overlay_transparent import overlay_transparent

_old_load = torch.load
def unsafe_load(*args, **kwargs):
    kwargs["weights_only"] = False
    return _old_load(*args, **kwargs)
torch.load = unsafe_load

def rotate_image(image, angle):
        h, w = image.shape[:2]
        cX, cY = (w // 2, h // 2)
        
        M = cv2.getRotationMatrix2D((cX, cY), angle, 1.0)
        
        cos = np.abs(M[0, 0])
        sin = np.abs(M[0, 1])
        nW = int((h * sin) + (w * cos))
        nH = int((h * cos) + (w * sin))
        
        M[0, 2] += (nW / 2) - cX
        M[1, 2] += (nH / 2) - cY
        
        return cv2.warpAffine(image, M, (nW, nH), flags=cv2.INTER_LINEAR)

class VideoThread(QThread):
    change_pixmap_signal = pyqtSignal(np.ndarray)

    def __init__(self, landmark_model_wp, device):
        super().__init__()
        self._run_flag = True
        self.device = device
        
        self.landmark_model = FaceLandmarkModel()
        self.landmark_model.load_state_dict(torch.load(landmark_model_wp, map_location=device, weights_only=True))
        self.landmark_model.to(device)
        self.landmark_model.eval()

        self.yolo_model = YOLO(hf_hub_download(repo_id="arnabdhar/YOLOv8-Face-Detection", filename="model.pt"))
        self.yolo_model.to(device)

        self.transform = A.Compose([
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2()
        ])

        self.filter_glasses = cv2.imread("assets/glasses.png", cv2.IMREAD_UNCHANGED)
        self.filter_moustache = cv2.imread("assets/moustache.png", cv2.IMREAD_UNCHANGED)
        self.show_glasses = False
        self.show_moustache = False

    def run(self):
        cap = cv2.VideoCapture(0)
        face_filters = {}
        MAX_IDLE_TIME = 2.0
        prev_time = time.time()
        fps = 0.0

        lk_params = dict(winSize=(15, 15),
                         maxLevel=2,
                         criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
        prev_gray = None

        while self._run_flag:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame = cv2.flip(frame, 1)
            h, w = frame.shape[:2]
            
            t_now = time.time() 

            current_fps = 1.0 / (t_now - prev_time + 1e-5)
            fps = (fps * 0.9) + (current_fps * 0.1)
            prev_time = t_now

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            results = self.yolo_model.track(frame, persist=True, verbose=False) 

            if results[0].boxes is not None and results[0].boxes.id is not None:
                boxes = results[0].boxes.xyxy.cpu().numpy()
                ids = results[0].boxes.id.cpu().numpy().astype(int)

                for box, face_id in zip(boxes, ids):
                    bx1, by1, bx2, by2 = map(int, box)
                    orig_w, orig_h = bx2 - bx1, by2 - by1
                    x1, y1 = max(0, int(bx1 - 0.2 * orig_w)), max(0, int(by1 - 0.2 * orig_h))
                    x2, y2 = min(w, int(bx2 + 0.2 * orig_w)), min(h, int(by2 + 0.2 * orig_h))
                    box_w, box_h = x2 - x1, y2 - y1

                    if box_w <= 20 or box_h <= 20: continue

                    crop_bgr = frame[y1:y2, x1:x2]
                    
                    crop_rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
                    model_image = cv2.resize(crop_rgb, (256, 256))
                    input_tensor = self.transform(image=model_image)["image"].unsqueeze(0).to(self.device)

                    with torch.no_grad():
                        outputs = self.landmark_model(input_tensor)
                    landmarks = outputs.cpu().numpy().squeeze().reshape(-1, 2)

                    if face_id not in face_filters:
                        face_filters[face_id] = {
                            "x": [KalmanFilter(q=5e-1, r=1e-1) for _ in range(len(landmarks))],
                            "y": [KalmanFilter(q=5e-1, r=1e-1) for _ in range(len(landmarks))],
                            "prev_points": None,
                            "last_seen": t_now
                        }
                    else:
                        face_filters[face_id]["last_seen"] = t_now
                    
                    current_measurements = np.zeros((len(landmarks), 2), dtype=np.float32)
                    for i, (lx, ly) in enumerate(landmarks):
                        current_measurements[i] = [lx * box_w + x1, ly * box_h + y1]

                    predictions = np.copy(current_measurements) # Mặc định nếu OF tịt thì lấy luôn ResNet
            
                    if prev_gray is not None and face_filters[face_id]["prev_points"] is not None:
                        p0 = face_filters[face_id]["prev_points"]
                        # Chạy Lucas-Kanade
                        p1, st, err = cv2.calcOpticalFlowPyrLK(prev_gray, gray, p0, None, **lk_params)
                        
                        # Lọc các điểm mà OF theo dõi thành công (status == 1)
                        for i in range(len(landmarks)):
                            if st[i][0] == 1:
                                predictions[i] = p1[i]

                    smoothed_points = np.zeros((98, 2), dtype=int)
                    new_prev_points = np.zeros((len(landmarks), 2), dtype=np.float32)

                    for i in range(len(landmarks)):
                        pred_x, pred_y = predictions[i]
                        meas_x, meas_y = current_measurements[i]

                        # kalman
                        smooth_x = face_filters[face_id]["x"][i].smooth(pred_x, meas_x)
                        smooth_y = face_filters[face_id]["y"][i].smooth(pred_y, meas_y)

                        smoothed_points[i] = [int(smooth_x), int(smooth_y)]
                        new_prev_points[i] = [smooth_x, smooth_y] 

                        cv2.circle(frame, (int(smooth_x), int(smooth_y)), 2, (180, 105, 255), -1)
                    
                    face_filters[face_id]["prev_points"] = new_prev_points
                    left_eye = smoothed_points[60]     
                    right_eye = smoothed_points[72] 
                    
                    dy = right_eye[1] - left_eye[1]
                    dx = right_eye[0] - left_eye[0]
                    head_angle = -np.degrees(np.arctan2(dy, dx)) 

                    if self.show_glasses and self.filter_glasses is not None:
                        nose_bridge = smoothed_points[51]  
                        
                        eye_width = np.linalg.norm(left_eye - right_eye)
                        glasses_width = int(eye_width * 2.0)
                        
                        aspect_ratio = self.filter_glasses.shape[0] / self.filter_glasses.shape[1]
                        glasses_height = int(glasses_width * aspect_ratio)
                        
                        if glasses_width > 0 and glasses_height > 0:
                            resized_glasses = cv2.resize(self.filter_glasses, (glasses_width, glasses_height))
                            
                            rotated_glasses = rotate_image(resized_glasses, head_angle)
                            rot_h, rot_w = rotated_glasses.shape[:2] 
                            gx = int(nose_bridge[0] - rot_w / 2)
                            gy = int(nose_bridge[1] - rot_h / 2)
                            
                            frame = overlay_transparent(frame, rotated_glasses, gx, gy)

                    if self.show_moustache and self.filter_moustache is not None:
                        moustache_left = smoothed_points[55]  
                        moustache_right = smoothed_points[59]  
                        moustache_mid = smoothed_points[57]  
                        
                        moustache_width = np.linalg.norm(moustache_left - moustache_right)
                        moustache_width = int(moustache_width * 3)
                        
                        aspect_ratio = self.filter_moustache.shape[0] / self.filter_moustache.shape[1]
                        moustache_height = int(moustache_width * aspect_ratio)
                        
                        if moustache_width > 0 and moustache_height > 0:
                            resized_moustache = cv2.resize(self.filter_moustache, (moustache_width, moustache_height))
                            
                            base_mx = int(moustache_mid[0] - moustache_width / 2)
                            base_my = int(moustache_mid[1] - moustache_height / 2)

                            shift_my = int(moustache_height * 0.15)
                            
                            mx = int(base_mx)
                            my = int(base_my - shift_my)
                            
                            frame = overlay_transparent(frame, resized_moustache, mx, my)

                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(frame, f"ID: {face_id}", (x1, y1 - 10), 0, 0.6, (0, 255, 0), 2)

            expired_ids = [fid for fid, data in face_filters.items() if t_now - data["last_seen"] > MAX_IDLE_TIME]
            for fid in expired_ids:
                del face_filters[fid]

            cv2.putText(frame, f"FPS: {int(fps)}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            self.change_pixmap_signal.emit(frame)

            prev_gray = gray.copy()
            
        cap.release()

    def stop(self):
        self._run_flag = False
        self.wait()
        
class FaceLandmarkApp(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("YORHA Face Landmark AI 🌸")
        self.resize(1000, 750)
        
        self.setStyleSheet("""
            QWidget {
                background-color: #FFFFFF; /* White background */
                color: #555555; /* Dark grey text for readability */
                font-family: 'Segoe UI', 'Helvetica Neue', sans-serif;
                font-size: 14px;
            }
            QLabel {
                color: #555555;
            }
            QLabel#VideoDisplay {
                background-color: #F0F8FF; /* AliceBlue - super light blue */
                border: 4px dashed #87CEFA; /* Dashed LightSkyBlue border */
                border-radius: 30px; /* Highly rounded corners */
                qproperty-alignment: AlignCenter;
                color: #87CEFA;
                font-weight: bold;
                font-size: 16px;
            }
            QLabel#TitleLabel {
                font-size: 24px;
                font-weight: bold;
                color: #FFB6C1; /* LightPink title */
                margin-bottom: 10px;
            }
            QPushButton {
                background-color: #87CEFA; /* LightSkyBlue buttons */
                color: white;
                border-radius: 20px; /* Rounded buttons */
                padding: 12px 24px;
                font-weight: bold;
                border: none;
            }
            QPushButton:hover {
                background-color: #00BFFF; /* DeepSkyBlue on hover */
            }
            QPushButton:pressed {
                background-color: #4682B4; /* SteelBlue when pressed */
            }
            QPushButton#StopBtn {
                background-color: #FFB6C1; /* LightPink for Stop button */
            }
            QPushButton#StopBtn:hover {
                background-color: #FF69B4; /* HotPink on hover */
            }
            QCheckBox {
                spacing: 10px;
                color: #555555;
            }
            QCheckBox::indicator {
                width: 22px;
                height: 22px;
                border-radius: 11px; /* Round indicators */
                border: 2px solid #87CEFA;
                background-color: white;
            }
            QCheckBox::indicator:checked {
                background-color: #87CEFA;
                image: url(assets/check.png); /* Need a small check icon or use border/color */
            }
            QCheckBox::indicator:unchecked:hover {
                border: 2px solid #00BFFF;
            }
            QFrame#ControlPanel {
                background-color: #FFF0F5; /* LavenderBlush - very light pink background */
                border-radius: 25px;
                padding: 20px;
                border: 2px solid #FFB6C1;
            }
        """)

        # --- MAIN LAYOUT ---
        main_layout = QVBoxLayout()
        main_layout.setContentsMargins(25, 25, 25, 25)
        main_layout.setSpacing(20)

        # Title
        self.title_label = QLabel("YORHA Face Landmark AI 🌸", self)
        self.title_label.setObjectName("TitleLabel")
        self.title_label.setAlignment(Qt.AlignCenter)
        main_layout.addWidget(self.title_label)

        # Video and Controls area
        content_layout = QHBoxLayout()
        content_layout.setSpacing(25)

        # --- LEFT SIDE: VIDEO DISPLAY ---
        self.image_label = QLabel(self)
        self.image_label.setObjectName("VideoDisplay")
        self.image_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.image_label.setText("Bấm 'Start AI' để bắt đầu nha senpai! (◡‿◡✿)")
        self.image_label.setMinimumSize(640, 480)

        self.control_panel = QFrame(self)
        self.control_panel.setObjectName("ControlPanel")
        control_layout = QVBoxLayout(self.control_panel)
        control_layout.setContentsMargins(15, 15, 15, 15)
        control_layout.setSpacing(15)

        panel_title = QLabel("Filter ✨")
        panel_title.setStyleSheet("font-size: 18px; font-weight: bold; color: #555;")
        panel_title.setAlignment(Qt.AlignCenter)
        control_layout.addWidget(panel_title)
        
        control_layout.addStretch(1)

        self.cb_glasses = QCheckBox("😎 Kính Thug Life", self)
        self.cb_moustache = QCheckBox("🥸 Moustache họa sĩ người áo", self)
        control_layout.addWidget(self.cb_glasses)
        control_layout.addWidget(self.cb_moustache)

        control_layout.addStretch(1)

        self.btn_start = QPushButton("▶ Start AI", self)
        self.btn_stop = QPushButton("🛑 Stop", self)
        self.btn_stop.setObjectName("StopBtn")
        control_layout.addWidget(self.btn_start)
        control_layout.addWidget(self.btn_stop)
        
        control_layout.addStretch(2) # Khoảng trống dưới

        # Hợp nhất Left và Right
        content_layout.addWidget(self.image_label, stretch=3)
        content_layout.addWidget(self.control_panel, stretch=1)

        main_layout.addLayout(content_layout)
        
        self.setLayout(main_layout)

        # Connect signals
        self.btn_start.clicked.connect(self.start_video)
        self.btn_stop.clicked.connect(self.stop_video)
        self.cb_glasses.stateChanged.connect(self.toggle_filters)
        self.cb_moustache.stateChanged.connect(self.toggle_filters)
        
        self.thread = None

    def toggle_filters(self):
        if self.thread and self.thread.isRunning():
            self.thread.show_glasses = self.cb_glasses.isChecked()
            self.thread.show_moustache = self.cb_moustache.isChecked()

    def start_video(self):
        if self.thread is None or not self.thread.isRunning():
            landmark_model_wp = "src/models/resnet18_v2.pt" 
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            
            self.thread = VideoThread(landmark_model_wp, device)
            
            self.thread.show_glasses = self.cb_glasses.isChecked()
            self.thread.show_moustache = self.cb_moustache.isChecked()
            
            self.thread.change_pixmap_signal.connect(self.update_image)
            self.thread.start()

    def stop_video(self):
        if self.thread and self.thread.isRunning():
            self.thread.stop()
            self.image_label.clear()
            self.image_label.setText("Đã tắt Camera. Chờ lệnh senpai! (๑˃ᴗ˂)ﻭ")

    def update_image(self, cv_img):
        qt_img = self.convert_cv_qt(cv_img)
        self.image_label.setPixmap(qt_img)

    def convert_cv_qt(self, cv_img):
        rgb_image = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        convert_to_Qt_format = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)

        target_w = self.image_label.width()
        target_h = self.image_label.height()

        p = convert_to_Qt_format.scaled(target_w, target_h, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        return QPixmap.fromImage(p)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = FaceLandmarkApp()
    window.show()
    sys.exit(app.exec_())