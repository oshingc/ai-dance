import cv2
import mediapipe as mp
import numpy as np
from datetime import datetime
import json
import time

class PoseDetector:
    def __init__(self):
        self.mp_pose = mp.solutions.pose
        self.mp_drawing = mp.solutions.drawing_utils
        self.pose = self.mp_pose.Pose(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.current_similarity = 0
        self.previous_score = 0  # Guardamos el score anterior
        self.is_active = False
        print("PoseDetector inicializado")

    def calculate_similarity(self, landmarks):
        if not landmarks:
            return self.previous_score  # Mantener score anterior si no hay landmarks
        
        # Solo muñecas
        left_wrist = landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_WRIST]
        right_wrist = landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_WRIST]
        
        target_score = 0
        
        # Calculamos el score objetivo
        if left_wrist.visibility > 0.5:
            if left_wrist.y < 0.5:
                target_score += 30
        
        if right_wrist.visibility > 0.5:
            if right_wrist.y < 0.5:
                target_score += 30
        
        # Bonus por ambos brazos
        if left_wrist.y < 0.5 and right_wrist.y < 0.5:
            target_score += 10
        
        # Suavizar la transición
        if target_score > self.previous_score:
            # Subir gradualmente (más rápido)
            new_score = min(target_score, self.previous_score + 5)
        else:
            # Bajar gradualmente (más lento)
            new_score = max(target_score, self.previous_score - 3)
        
        self.previous_score = new_score
        return min(100, new_score)

    def process_frame(self, frame):
        if frame is None:
            return None
        
        frame = cv2.flip(frame, 1)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.pose.process(frame_rgb)

        if results.pose_landmarks:
            self.mp_drawing.draw_landmarks(
                frame,
                results.pose_landmarks,
                self.mp_pose.POSE_CONNECTIONS,
                self.mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
                self.mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2)
            )
            
            score = self.calculate_similarity(results.pose_landmarks)
            self.current_similarity = score
            
            cv2.putText(frame, 
                       f"{int(score)}%",  # Convertir a entero para evitar decimales
                       (50, 70),
                       cv2.FONT_HERSHEY_SIMPLEX, 
                       2,
                       (0, 255, 0), 
                       3)
        
        return frame

    def get_similarity(self):
        print(f"Retornando similitud actual: {self.current_similarity}")
        return self.current_similarity

    def get_recent_similarity_data(self):
        print(f"Retornando datos recientes. Similitud: {self.current_similarity}")
        return [{
            "timestamp": datetime.now().isoformat(),
            "similarity": self.current_similarity
        }]

    def start(self):
        print("¡Detector iniciado!")
        self.is_active = True
        self.current_similarity = 0
        self.previous_score = 0  # Reiniciar score anterior

    def stop(self):
        print("Detector detenido")
        self.is_active = False
        self.current_similarity = 0
        self.previous_score = 0  # Reiniciar score anterior

    def __del__(self):
        self.pose.close()