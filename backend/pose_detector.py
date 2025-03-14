import cv2
import mediapipe as mp
import numpy as np
from datetime import datetime
import json
import time
import os

class PoseDetector:
    def __init__(self):
        self.mp_pose = mp.solutions.pose
        self.mp_drawing = mp.solutions.drawing_utils
        self.pose = self.mp_pose.Pose(
            min_detection_confidence=0.7,  # Aumentamos la confianza mínima
            min_tracking_confidence=0.7,   # Aumentamos el tracking
            model_complexity=2  # Usamos el modelo más preciso
        )
        
        # Estados
        self.countdown_duration = 5
        self.countdown_start = None
        self.dance_start = None
        self.is_countdown = False
        self.is_dancing = False
        self.current_similarity = 0
        self.frame_index = 0
        
        # Cargar keypoints de referencia
        try:
            keypoints_path = os.path.join(os.path.dirname(__file__), "reference_keypoints_misamo.json")
            print(f"Intentando cargar keypoints desde: {keypoints_path}")
            with open(keypoints_path, "r") as f:
                self.reference_keypoints = json.load(f)
                print(f"Keypoints cargados exitosamente: {len(self.reference_keypoints)} frames")
        except Exception as e:
            print(f"Error cargando keypoints: {str(e)}")
            self.reference_keypoints = []

        self.required_landmarks = [
            self.mp_pose.PoseLandmark.NOSE,
            self.mp_pose.PoseLandmark.LEFT_SHOULDER,
            self.mp_pose.PoseLandmark.RIGHT_SHOULDER,
            self.mp_pose.PoseLandmark.LEFT_ELBOW,
            self.mp_pose.PoseLandmark.RIGHT_ELBOW,
            self.mp_pose.PoseLandmark.LEFT_WRIST,
            self.mp_pose.PoseLandmark.RIGHT_WRIST,
            self.mp_pose.PoseLandmark.LEFT_HIP,
            self.mp_pose.PoseLandmark.RIGHT_HIP,
            self.mp_pose.PoseLandmark.LEFT_KNEE,
            self.mp_pose.PoseLandmark.RIGHT_KNEE,
            self.mp_pose.PoseLandmark.LEFT_ANKLE,
            self.mp_pose.PoseLandmark.RIGHT_ANKLE
        ]

    def calculate_similarity(self, landmarks):
        # Verificar si todos los puntos clave están visibles
        if not landmarks:
            return 0
            
        visible_landmarks = 0
        required_landmarks = len(self.required_landmarks)
        
        for landmark in self.required_landmarks:
            if (landmarks.landmark[landmark].visibility > 0.7):  # Aumentamos el umbral de visibilidad
                visible_landmarks += 1
        
        # Calcular porcentaje de visibilidad
        visibility_percentage = (visible_landmarks / required_landmarks) * 100
        
        # Si no se ve el cuerpo completo, penalizar la similitud
        if visibility_percentage < 80:  # Necesitamos ver al menos 80% del cuerpo
            return int(visibility_percentage / 2)  # Penalización fuerte
            
        return int(visibility_percentage)

    def process_frame(self, frame):
        if frame is None:
            return None

        # Primero hacemos el flip para efecto espejo
        frame = cv2.flip(frame, 1)
        
        # Convertir a RGB para MediaPipe
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.pose.process(frame_rgb)

        if results.pose_landmarks:
            # Calcular similitud basada en visibilidad
            self.current_similarity = self.calculate_similarity(results.pose_landmarks)
            
            # Dibujar landmarks
            self.mp_drawing.draw_landmarks(
                frame,
                results.pose_landmarks,
                self.mp_pose.POSE_CONNECTIONS,
                self.mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
                self.mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2)
            )
            
            # Mostrar mensaje si no se ve el cuerpo completo
            if self.current_similarity < 80:
                cv2.putText(frame, "Alejate de la camara!", 
                           (int(frame.shape[1]/2) - 100, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        else:
            self.current_similarity = 0
            cv2.putText(frame, "No se detecta persona", 
                       (int(frame.shape[1]/2) - 100, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        if self.is_countdown:
            elapsed = time.time() - self.countdown_start
            remaining = self.countdown_duration - elapsed

            if remaining <= 0:
                self.is_countdown = False
                self.is_dancing = True
                self.dance_start = time.time()
                self.frame_index = 0
            else:
                # Crear una copia del frame para el texto
                text_overlay = frame.copy()
                
                # Dibujar texto en la capa de overlay
                count = int(remaining) + 1
                cv2.putText(
                    text_overlay,
                    str(count),
                    (frame.shape[1]//2 - 50, frame.shape[0]//2),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    4,
                    (255, 255, 255),
                    6
                )
                cv2.putText(
                    text_overlay,
                    "LISTOS!",
                    (frame.shape[1]//2 - 80, frame.shape[0]//2 - 100),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.5,
                    (255, 255, 255),
                    2
                )
                
                # Voltear solo la capa de texto
                text_overlay = cv2.flip(text_overlay, 1)
                
                # Combinar las capas
                frame = cv2.addWeighted(frame, 0.7, text_overlay, 0.3, 0)

        # Timer y similitud
        if self.is_dancing:
            elapsed_time = int(time.time() - self.dance_start)
            minutes = elapsed_time // 60
            seconds = elapsed_time % 60
            
            # Crear capa para texto
            text_overlay = frame.copy()
            
            # Dibujar texto
            cv2.putText(
                text_overlay,
                f"{minutes:02d}:{seconds:02d}",
                (frame.shape[1]//2 - 70, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                2,
                (255, 255, 255),
                3
            )
            
            # Voltear solo el texto
            text_overlay = cv2.flip(text_overlay, 1)
            
            # Combinar capas
            frame = cv2.addWeighted(frame, 0.7, text_overlay, 0.3, 0)

        return frame

    def get_recent_similarity_data(self):
        if not self.is_dancing:
            return []
            
        return [{
            "timestamp": datetime.now().isoformat(),
            "similarity": self.current_similarity
        }]

    def get_similarity(self):
        return self.current_similarity

    def __del__(self):
        self.pose.close()