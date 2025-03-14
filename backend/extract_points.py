import cv2
import mediapipe as mp
import numpy as np
import json

# Configuración de Mediapipe
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()
mp_draw = mp.solutions.drawing_utils

# Extraer keypoints de la coreografía de referencia
video_path = "./data/misamo_dance.mp4"
cap = cv2.VideoCapture(video_path)

reference_keypoints = []

def normalize_keypoints(landmarks):
    """ Normaliza los keypoints en relación con la distancia entre las caderas. """
    left_hip = landmarks[mp_pose.PoseLandmark.LEFT_HIP.value]
    right_hip = landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value]
    hip_distance = np.linalg.norm(np.array([left_hip.x, left_hip.y]) - np.array([right_hip.x, right_hip.y]))
    
    if hip_distance == 0:
        return None  # Evita divisiones por cero
    
    keypoints = [[lm.x / hip_distance, lm.y / hip_distance, lm.z / hip_distance] for lm in landmarks]
    return keypoints

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(img_rgb)
    
    if results.pose_landmarks:
        keypoints = normalize_keypoints(results.pose_landmarks.landmark)
        if keypoints:
            reference_keypoints.append(keypoints)

cap.release()

# Guardar los keypoints normalizados en un archivo JSON
with open("reference_keypoints_misamo.json", "w") as f:
    json.dump(reference_keypoints, f, indent=4)

print("Reference keypoints saved!")