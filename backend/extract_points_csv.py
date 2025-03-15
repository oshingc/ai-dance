import cv2
import mediapipe as mp
import numpy as np
import csv
import os

# Configuración de Mediapipe
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()
mp_draw = mp.solutions.drawing_utils

# Lista de videos de coreografía
video_paths = [
    "./data/misamo_dance1.mp4",
    "./data/misamo_dance2.mp4",
    # Agrega aquí las rutas de tus otros videos
]

all_keypoints = []

def normalize_keypoints(landmarks):
    """ Normaliza los keypoints en relación con la distancia entre las caderas. """
    left_hip = landmarks[mp_pose.PoseLandmark.LEFT_HIP.value]
    right_hip = landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value]
    hip_distance = np.linalg.norm(np.array([left_hip.x, left_hip.y]) - np.array([right_hip.x, right_hip.y]))
    
    if hip_distance == 0:
        return None  # Evita divisiones por cero
    
    keypoints = [[lm.x / hip_distance, lm.y / hip_distance, lm.z / hip_distance] for lm in landmarks]
    return keypoints

for video_path in video_paths:
    cap = cv2.VideoCapture(video_path)
    video_name = os.path.basename(video_path)  # Obtener el nombre del archivo de video
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(img_rgb)
        
        if results.pose_landmarks:
            keypoints = normalize_keypoints(results.pose_landmarks.landmark)
            if keypoints:
                all_keypoints.append([video_name] + keypoints)  # Agregar nombre del video

    cap.release()

# Guardar los keypoints normalizados en un archivo CSV
with open("all_reference_keypoints.csv", "w", newline='') as csvfile:
    writer = csv.writer(csvfile)
    # Escribir encabezado
    header = ["video_name"] + [f"landmark_{i}_{coord}" for i in range(len(all_keypoints[0]) - 1) for coord in ['x', 'y', 'z']] # Se resta 1 por la columna video_name
    writer.writerow(header)
    
    # Escribir datos
    for row in all_keypoints:
        flat_row = [row[0]] + [coord for point in row[1:] for coord in point] # Se agrega el nombre del video al principio
        writer.writerow(flat_row)

print("All reference keypoints saved!")