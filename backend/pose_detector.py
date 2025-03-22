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
        self.previous_score = 0
        self.is_active = False
        
        # Lista de keypoints importantes para el baile
        self.IMPORTANT_KEYPOINTS = [
            'NOSE', 'LEFT_SHOULDER', 'RIGHT_SHOULDER',
            'LEFT_ELBOW', 'RIGHT_ELBOW', 'LEFT_WRIST', 'RIGHT_WRIST',
            'LEFT_HIP', 'RIGHT_HIP', 'LEFT_KNEE', 'RIGHT_KNEE',
            'LEFT_ANKLE', 'RIGHT_ANKLE'
        ]

        # Cargar poses de referencia
        try:
            with open('backend/reference_keypoints_armagedon.json', 'r') as file:
                self.reference_keypoints = json.load(file)
                print(f"Cargados {len(self.reference_keypoints)} frames de referencia")
        except Exception as e:
            print(f"Error cargando keypoints: {e}")
            self.reference_keypoints = []

    def calculate_pose_similarity(self, current_pose, reference_pose):
        total_similarity = 0
        valid_points = 0
        
        # Pesos para diferentes partes del cuerpo
        weights = {
            'NOSE': 0.5,
            'LEFT_SHOULDER': 1.5,
            'RIGHT_SHOULDER': 1.5,
            'LEFT_ELBOW': 2.0,
            'RIGHT_ELBOW': 2.0,
            'LEFT_WRIST': 2.5,    # Mayor peso a las manos
            'RIGHT_WRIST': 2.5,
            'LEFT_HIP': 1.5,
            'RIGHT_HIP': 1.5,
            'LEFT_KNEE': 1.0,
            'RIGHT_KNEE': 1.0,
            'LEFT_ANKLE': 0.5,
            'RIGHT_ANKLE': 0.5
        }
        
        for keypoint in self.IMPORTANT_KEYPOINTS:
            current = current_pose.landmark[getattr(self.mp_pose.PoseLandmark, keypoint)]
            reference = reference_pose.landmark[getattr(self.mp_pose.PoseLandmark, keypoint)]
            
            if current.visibility > 0.5 and reference.visibility > 0.5:
                # Calcular distancia euclidiana 2D (ignorando z para mayor tolerancia)
                distance = np.sqrt(
                    (current.x - reference.x)**2 + 
                    (current.y - reference.y)**2
                )
                
                # Hacer la comparación más tolerante (0.5)
                point_similarity = max(0, 100 * (1 - distance/0.5))
                
                # Aplicar peso específico para cada parte del cuerpo
                weight = weights.get(keypoint, 1.0)
                total_similarity += point_similarity * weight
                valid_points += weight
        
        # Ajustar el puntaje final
        final_score = (total_similarity / valid_points if valid_points > 0 else 0)
        return min(100, final_score * 1.2)

    def calculate_similarity(self, landmarks):
        if not landmarks:
            return self.previous_score
        
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
            new_score = min(target_score, self.previous_score + 2)
        else:
            new_score = max(target_score, self.previous_score - 1)
        
        self.previous_score = new_score
        return min(100, new_score)

    def process_frame(self, frame):
        if frame is None:
            return None
        

        # Procesar el frame recortado
        frame = cv2.flip(frame, 1)
        
        # Redimensionar al formato vertical de TikTok (9:16)
        height, width = frame.shape[:2]
        target_width = int((height * 9) / 16)  # Mantener proporción 9:16
        
        # Calcular los puntos de recorte para centrar
        start_x = (width - target_width) // 2
        end_x = start_x + target_width
        
        # Recortar al formato vertical
        frame = frame[:, start_x:end_x]
        
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
        
        return frame

    def get_similarity(self):
        return self.current_similarity

    def get_recent_similarity_data(self):
        return [{
            "timestamp": datetime.now().isoformat(),
            "similarity": self.current_similarity
        }]

    def start(self):
        self.is_active = True
        self.current_similarity = 0
        self.previous_score = 0

    def stop(self):
        self.is_active = False
        self.current_similarity = 0
        self.previous_score = 0

    def __del__(self):
        self.pose.close()