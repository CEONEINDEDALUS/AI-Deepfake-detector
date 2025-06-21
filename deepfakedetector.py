import sys
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import mediapipe as mp
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
                            QPushButton, QLabel, QFileDialog, QSlider, QComboBox, QMessageBox)
from PyQt5.QtCore import Qt, QTimer, QThread, pyqtSignal
from PyQt5.QtGui import QImage, QPixmap
import uuid
import time
import requests
import os
import sqlite3
from pathlib import Path

# Dummy CNN model for deepfake detection (will load pretrained weights)
class DeepfakeCNN(nn.Module):
    def __init__(self):
        super(DeepfakeCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(32 * 56 * 56, 128)
        self.fc2 = nn.Linear(128, 1)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = x.view(-1, 32 * 56 * 56)
        x = self.relu(self.fc1(x))
        x = self.sigmoid(self.fc2(x))
        return x

# Dataset for fine-tuning
class FeedbackDataset(Dataset):
    def __init__(self, db_path):
        self.conn = sqlite3.connect(db_path)
        self.cursor = self.conn.cursor()
        self.cursor.execute("SELECT image, label FROM feedback")
        self.data = [(np.frombuffer(img, dtype=np.uint8), label) for img, label in self.cursor.fetchall()]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_data, label = self.data[idx]
        img = cv2.imdecode(img_data, cv2.IMREAD_COLOR)
        img = cv2.resize(img, (224, 224)).transpose((2, 0, 1)) / 255.0
        img = torch.tensor(img, dtype=torch.float32)
        label = torch.tensor(label, dtype=torch.float32)
        return img, label

    def __del__(self):
        self.conn.close()

# Worker thread for video processing
class VideoProcessor(QThread):
    frame_ready = pyqtSignal(np.ndarray, list)
    status_update = pyqtSignal(str)
    time_update = pyqtSignal(str, str)

    def __init__(self, video_path, mode, threshold, db_path):
        super().__init__()
        self.video_path = video_path
        self.mode = mode
        self.threshold = threshold
        self.db_path = db_path
        self.running = False
        self.paused = False
        self.cap = None
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(max_num_faces=3)
        self.model = DeepfakeCNN().eval()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.avg_fake_score = []
        self.last_eye_state = {}
        self.blink_threshold = 0.2
        self.frame_count = 0
        self.face_crops = []
        self.face_coords = []
        # Add blob detector parameters
        self.params = cv2.SimpleBlobDetector_Params()
        self.params.minThreshold = 10
        self.params.maxThreshold = 200
        self.params.filterByArea = True
        self.params.minArea = 1500
        self.blob_detector = cv2.SimpleBlobDetector_create(self.params)
        self.download_model()

    def download_model(self):
        model_path = Path("deepfake_model.pth")
        if not model_path.exists():
            self.status_update.emit("Downloading pretrained model...")
            try:
                # Placeholder URL (replace with actual pretrained model URL, e.g., from Hugging Face)
                url = "https://example.com/deepfake_model.pth"
                response = requests.get(url, stream=True)
                if response.status_code == 200:
                    with open(model_path, "wb") as f:
                        f.write(response.content)
                    self.status_update.emit("Model downloaded successfully.")
                else:
                    self.status_update.emit("Failed to download model. Using default model.")
            except Exception as e:
                self.status_update.emit(f"Error downloading model: {str(e)}")
        try:
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
            self.status_update.emit("Pretrained model loaded.")
        except Exception as e:
            self.status_update.emit(f"Error loading model: {str(e)}. Using default model.")

    def fine_tune_model(self):
        self.status_update.emit("Fine-tuning model...")
        dataset = FeedbackDataset(self.db_path)
        if len(dataset) == 0:
            self.status_update.emit("No feedback data for fine-tuning.")
            return
        dataloader = DataLoader(dataset, batch_size=4, shuffle=True)
        optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        criterion = nn.BCELoss()
        self.model.train()
        for epoch in range(3):  # Few epochs for quick fine-tuning
            for images, labels in dataloader:
                images, labels = images.to(self.device), labels.to(self.device).view(-1, 1)
                optimizer.zero_grad()
                outputs = self.model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
        self.model.eval()
        torch.save(self.model.state_dict(), "deepfake_model.pth")
        self.status_update.emit("Model fine-tuned and saved.")

    def preprocess_face(self, face_crop):
        face_crop = cv2.resize(face_crop, (224, 224))
        face_crop = face_crop.transpose((2, 0, 1)) / 255.0
        face_crop = torch.tensor(face_crop, dtype=torch.float32).unsqueeze(0).to(self.device)
        return face_crop

    def detect_blink(self, landmarks, prev_state, face_id):
        left_eye = [33, 133]
        right_eye = [362, 263]
        def eye_aspect_ratio(eye_indices):
            p1, p2 = landmarks[eye_indices[0]], landmarks[eye_indices[1]]
            dist = np.sqrt((p2.x - p1.x)**2 + (p2.y - p1.y)**2)
            return dist
        ear_left = eye_aspect_ratio(left_eye)
        ear_right = eye_aspect_ratio(right_eye)
        ear = (ear_left + ear_right) / 2.0
        is_closed = ear < self.blink_threshold
        blink_detected = prev_state.get(face_id, False) and not is_closed
        self.last_eye_state[face_id] = is_closed
        return blink_detected, ear

    def analyze_frame(self, face_crop, score):
        # Detect potential artificial artifacts using blob detection
        gray = cv2.cvtColor(face_crop, cv2.COLOR_BGR2GRAY)
        keypoints = self.blob_detector.detect(gray)
        
        # More blobs may indicate artificial artifacts
        artifact_score = len(keypoints) * 0.1
        return min(score + artifact_score, 1.0), keypoints

    def run(self):
        self.cap = cv2.VideoCapture(self.video_path)
        if not self.cap.isOpened():
            self.status_update.emit("Error: Could not open video file.")
            return
        total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = self.cap.get(cv2.CAP_PROP_FPS)
        duration = total_frames / fps if fps > 0 else 0
        self.running = True
        current_frame = 0

        while self.running and self.cap.isOpened():
            if not self.paused:
                ret, frame = self.cap.read()
                if not ret:
                    break
                current_time = current_frame / fps if fps > 0 else 0
                time_str = f"{int(current_time//60):02d}:{int(current_time%60):02d}"
                duration_str = f"{int(duration//60):02d}:{int(duration%60):02d}"
                self.time_update.emit(time_str, duration_str)

                self.face_crops = []
                self.face_coords = []
                if self.mode == "Deepfake Finder":
                    self.status_update.emit("Analyzing frame...")
                    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    results = self.face_mesh.process(rgb_frame)
                    if results.multi_face_landmarks:
                        for idx, face_landmarks in enumerate(results.multi_face_landmarks):
                            face_id = str(uuid.uuid4())[:8]
                            h, w = frame.shape[:2]
                            x_min, y_min, x_max, y_max = w, h, 0, 0
                            for lm in face_landmarks.landmark:
                                x, y = int(lm.x * w), int(lm.y * h)
                                x_min, y_min = min(x_min, x), min(y_min, y)
                                x_max, y_max = max(x_max, x), max(y_max, y)
                            if x_max > x_min and y_max > y_min:
                                face_crop = frame[y_min:y_max, x_min:x_max]
                                if face_crop.size > 0:
                                    face_tensor = self.preprocess_face(face_crop)
                                    with torch.no_grad():
                                        score = self.model(face_tensor).item()
                                    self.avg_fake_score.append(score)
                                    self.face_crops.append(face_crop)
                                    self.face_coords.append((x_min, y_min, x_max, y_max, score, face_id))
                                    if score > self.threshold:
                                        fake_percent = int(score * 100)
                                        real_percent = 100 - fake_percent
                                        
                                        # Apply blob analysis
                                        adjusted_score, keypoints = self.analyze_frame(face_crop, score)
                                        adjusted_fake = int(adjusted_score * 100)
                                        
                                        # Draw detection box and labels
                                        cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 0, 255), 2)
                                        
                                        # Add labels for both fake and real percentages
                                        cv2.putText(frame, f"FAKE: {adjusted_fake}%", 
                                                    (x_min, y_min-25), cv2.FONT_HERSHEY_SIMPLEX, 
                                                    0.7, (0, 0, 255), 2)
                                        cv2.putText(frame, f"REAL: {100-adjusted_fake}%", 
                                                    (x_min, y_min-5), cv2.FONT_HERSHEY_SIMPLEX, 
                                                    0.7, (0, 255, 0), 2)
                                        
                                        # Draw detected blobs
                                        frame = cv2.drawKeypoints(frame, keypoints, np.array([]), 
                                                                (0, 255, 255),
                                                                cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
                                        
                                        self.status_update.emit(
                                            f"Fake detected! Score: {adjusted_fake}% (Raw: {fake_percent}%)")
                                    blink, ear = self.detect_blink(face_landmarks.landmark, self.last_eye_state, face_id)
                                    if blink:
                                        self.status_update.emit("Blink detected (auxiliary signal)")

                self.frame_ready.emit(frame, self.face_crops)
                current_frame += 1
                self.frame_count = current_frame
                time.sleep(1.0 / fps if fps > 0 else 0.033)
            else:
                time.sleep(0.1)

        if self.avg_fake_score:
            avg_score = np.mean(self.avg_fake_score)
            self.status_update.emit(f"Analysis complete. Average fake score: {int(avg_score*100)}%")
        self.cap.release()
        self.running = False

    def stop(self):
        self.running = False
        if self.cap:
            self.cap.release()

    def pause(self):
        self.paused = True

    def resume(self):
        self.paused = False

class DeepfakeDetectorApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Deepfake Detector")
        self.setStyleSheet("""
            QMainWindow { background-color: #2b2b2b; }
            QPushButton { background-color: #3c3f41; color: #ffffff; border: 1px solid #555555; padding: 5px; }
            QPushButton:hover { background-color: #4a4a4a; }
            QLabel { color: #ffffff; }
            QComboBox { background-color: #3c3f41; color: #ffffff; border: 1px solid #555555; }
            QSlider { background-color: #3c3f41; }
        """)
        self.db_path = "feedback.db"
        self.init_db()
        self.init_ui()
        self.video_processor = None
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)
        self.video_path = ""
        self.current_face_crops = []

    def init_db(self):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS feedback (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                image BLOB,
                label INTEGER
            )
        """)
        conn.commit()
        conn.close()

    def init_ui(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout = QVBoxLayout(central_widget)

        # Control panel
        control_layout = QHBoxLayout()
        self.load_btn = QPushButton("Load Video")
        self.load_btn.clicked.connect(self.load_video)
        control_layout.addWidget(self.load_btn)

        self.mode_combo = QComboBox()
        self.mode_combo.addItems(["Simple Player", "Deepfake Finder"])
        control_layout.addWidget(self.mode_combo)

        self.play_btn = QPushButton("Play")
        self.play_btn.clicked.connect(self.play_video)
        control_layout.addWidget(self.play_btn)

        self.pause_btn = QPushButton("Pause")
        self.pause_btn.clicked.connect(self.pause_video)
        control_layout.addWidget(self.pause_btn)

        self.stop_btn = QPushButton("Stop")
        self.stop_btn.clicked.connect(self.stop_video)
        control_layout.addWidget(self.stop_btn)

        self.finetune_btn = QPushButton("Fine-Tune Model")
        self.finetune_btn.clicked.connect(self.fine_tune)
        control_layout.addWidget(self.finetune_btn)

        layout.addLayout(control_layout)

        # Video display
        self.video_label = QLabel("No video loaded")
        self.video_label.setAlignment(Qt.AlignCenter)
        self.video_label.setMinimumSize(640, 480)
        layout.addWidget(self.video_label)

        # Feedback buttons
        feedback_layout = QHBoxLayout()
        self.real_btn = QPushButton("Label as Real")
        self.real_btn.clicked.connect(lambda: self.label_feedback(0))
        self.fake_btn = QPushButton("Label as Fake")
        self.fake_btn.clicked.connect(lambda: self.label_feedback(1))
        feedback_layout.addWidget(self.real_btn)
        feedback_layout.addWidget(self.fake_btn)
        layout.addLayout(feedback_layout)

        # Status and time
        self.status_label = QLabel("Status: Idle")
        layout.addWidget(self.status_label)

        self.time_label = QLabel("Current Time: 00:00 / 00:00")
        layout.addWidget(self.time_label)

        # Detection threshold slider
        threshold_layout = QHBoxLayout()
        threshold_label = QLabel("Detection Threshold: 50%")
        self.threshold_slider = QSlider(Qt.Horizontal)
        self.threshold_slider.setMinimum(0)
        self.threshold_slider.setMaximum(100)
        self.threshold_slider.setValue(50)
        self.threshold_slider.valueChanged.connect(
            lambda: threshold_label.setText(f"Detection Threshold: {self.threshold_slider.value()}%"))
        threshold_layout.addWidget(threshold_label)
        threshold_layout.addWidget(self.threshold_slider)
        layout.addLayout(threshold_layout)

    def load_video(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Select Video File", "",
                                                  "Video Files (*.mp4 *.avi *.mov)")
        if file_path:
            self.video_path = file_path
            self.status_label.setText(f"Loaded: {file_path.split('/')[-1]}")
            self.video_label.setText("Video loaded. Press Play to start.")

    def play_video(self):
        if not self.video_path:
            QMessageBox.warning(self, "Error", "Please load a video file first.")
            return
        if self.video_processor and self.video_processor.isRunning():
            self.video_processor.resume()
            self.status_label.setText("Playing...")
        else:
            self.video_processor = VideoProcessor(
                self.video_path, self.mode_combo.currentText(), self.threshold_slider.value() / 100.0, self.db_path)
            self.video_processor.frame_ready.connect(self.display_frame)
            self.video_processor.status_update.connect(self.status_label.setText)
            self.video_processor.time_update.connect(
                lambda current, total: self.time_label.setText(f"Current Time: {current} / {total}"))
            self.video_processor.start()
            self.timer.start(33)
            self.status_label.setText("Playing...")

    def pause_video(self):
        if self.video_processor and self.video_processor.isRunning():
            self.video_processor.pause()
            self.status_label.setText("Paused")

    def stop_video(self):
        if self.video_processor:
            self.video_processor.stop()
            self.timer.stop()
            self.status_label.setText("Stopped")
            self.video_label.setText("No video loaded")
            self.video_processor = None
            self.current_face_crops = []

    def display_frame(self, frame, face_crops):
        self.current_face_crops = face_crops
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_frame.shape
        bytes_per_line = ch * w
        q_image = QImage(rgb_frame.data, w, h, bytes_per_line, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(q_image)
        scaled_pixmap = pixmap.scaled(self.video_label.size(), Qt.KeepAspectRatio)
        self.video_label.setPixmap(scaled_pixmap)

    def label_feedback(self, label):
        if not self.current_face_crops:
            QMessageBox.warning(self, "Error", "No faces detected to label.")
            return
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        for face_crop in self.current_face_crops:
            _, buffer = cv2.imencode(".png", face_crop)
            cursor.execute("INSERT INTO feedback (image, label) VALUES (?, ?)", (buffer.tobytes(), label))
        conn.commit()
        conn.close()
        self.status_label.setText(f"Labeled {len(self.current_face_crops)} face(s) as {'Real' if label == 0 else 'Fake'}")

    def fine_tune(self):
        if self.video_processor:
            self.video_processor.fine_tune_model()
        else:
            self.status_label.setText("No video processor active. Load and play a video first.")

    def update_frame(self):
        pass

    def closeEvent(self, event):
        if self.video_processor:
            self.video_processor.stop()
        event.accept()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = DeepfakeDetectorApp()
    window.resize(800, 600)
    window.show()
    sys.exit(app.exec_())