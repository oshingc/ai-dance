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

        # Cargar poses de referencia desde el archivo JSON
        self.reference_keypoints_armagedon = self.load_reference_poses()

    def load_reference_poses(self):
        try:
            # Cargar el archivo JSON con las poses de Armagedón
            with open('reference_poses/armagedon_poses.json', 'r') as file:
                poses_data = json.load(file)
                return poses_data['poses']  # Asumiendo que el JSON tiene una estructura con 'poses'
        except Exception as e:
            print(f"Error al cargar poses de referencia: {e}")
            return None

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
        if not landmarks or self.reference_keypoints_armagedon is None:
            return self.previous_score
        
        # Uso de las poses de referencia
        reference_poses = self.reference_keypoints_armagedon
        
        # Calcular similitud con cada pose de referencia
        similarities = [self.calculate_pose_similarity(landmarks, ref_pose) 
                       for ref_pose in reference_poses]
        
        # Tomar la mejor similitud
        target_score = max(similarities)
        
        # Suavizar la transición
        if target_score > self.previous_score:
            new_score = min(target_score, self.previous_score + 4)
        else:
            new_score = max(target_score, self.previous_score - 2)
        
        self.previous_score = new_score
        return min(100, new_score)

    def process_frame(self, frame):
        if frame is None:
            return None
        
        frame = cv2.flip(frame, 1)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.pose.process(frame_rgb)

        if results.pose_landmarks:
            # Si no tenemos poses de referencia, usar la primera pose detectada
            if not self.reference_keypoints_armagedon:
                self.reference_keypoints_armagedon = [results.pose_landmarks]
                print("Primera pose guardada como referencia")

            self.mp_drawing.draw_landmarks(
                frame,
                results.pose_landmarks,
                self.mp_pose.POSE_CONNECTIONS,
                self.mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
                self.mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2)
            )
            
            score = self.calculate_similarity(results.pose_landmarks)
            self.current_similarity = int(score)
            
            cv2.putText(frame, 
                       f"{self.current_similarity}%",
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
        print("Iniciando detector - reseteando scores")
        self.is_active = True
        self.current_similarity = 0
        self.previous_score = 0

    def stop(self):
        print("Deteniendo detector")
        self.is_active = False
        self.current_similarity = 0
        self.previous_score = 0

    def __del__(self):
        self.pose.close()