from flask import Flask, Response, jsonify
from flask_cors import CORS
import cv2
import logging
import sys
import os
import time
from pose_detector import PoseDetector

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)

# Inicializar la cámara y el detector
camera = cv2.VideoCapture(0)
pose_detector = PoseDetector()

def generate_frames():
    while True:
        success, frame = camera.read()
        if not success:
            break
        else:
            # Solo voltear horizontalmente
            frame = cv2.flip(frame, 1)
            frame = pose_detector.process_frame(frame)
            
            if frame is not None:
                ret, buffer = cv2.imencode('.jpg', frame)
                if ret:
                    frame = buffer.tobytes()
                    yield (b'--frame\r\n'
                           b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/start_camera')
def start_camera():
    pose_detector.start()
    return jsonify({"status": "success"})

@app.route('/stop_camera')
def stop_camera():
    pose_detector.stop()
    return jsonify({"status": "success"})

@app.route('/get_similarity')
def get_similarity():
    return jsonify({
        "similarity": pose_detector.get_similarity()
    })

if __name__ == '__main__':
    app.run(debug=True, port=5000)

