
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

# Agregar el directorio que contiene pose_detector.py al path
#sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'mediapipe'))

app = Flask(__name__)
CORS(app)

class VideoStream:
    def __init__(self):
        self.cap = None
        self.is_active = False
        self.pose_detector = PoseDetector()

    def start(self):
        if not self.is_active:
            self.cap = cv2.VideoCapture(0)
            # Configurar la cámara para mejor calidad
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
            self.cap.set(cv2.CAP_PROP_FPS, 30)
            self.is_active = True
        return self.is_active

    def stop(self):
        if self.is_active:
            self.cap.release()
            self.is_active = False

    def get_frame(self):
        if not self.is_active:
            return None
        ret, frame = self.cap.read()
        if not ret:
            return None
        # Aplicar flip horizontal aquí
        frame = cv2.flip(frame, 1)
        return frame

video_stream = VideoStream()

@app.route('/start_camera')
def start_camera():
    success = video_stream.start()
    return jsonify({"status": "success" if success else "error"})

@app.route('/stop_camera')
def stop_camera():
    video_stream.stop()
    return jsonify({"status": "success"})

def gen_frames():
    while True:
        if video_stream.is_active:
            frame = video_stream.get_frame()
            if frame is not None:
                # Procesar frame con pose detector
                processed_frame = video_stream.pose_detector.process_frame(frame)
                if processed_frame is not None:
                    ret, buffer = cv2.imencode('.jpg', processed_frame, [cv2.IMWRITE_JPEG_QUALITY, 90])
                    if ret:
                        yield (b'--frame\r\n'
                               b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
            else:
                time.sleep(0.01)  # Reducido el tiempo de espera
        else:
            time.sleep(0.01)  # Reducido el tiempo de espera

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(),
                   mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/get_similarity_data')
def get_similarity_data():
    return jsonify(video_stream.pose_detector.get_recent_similarity_data())

@app.route('/get_similarity')
def get_similarity():
    if video_stream.is_active:
        similarity = video_stream.pose_detector.get_similarity()
        return jsonify({"similarity": similarity})
    return jsonify({"similarity": 0})

if __name__ == '__main__':
    try:
        app.run(host='0.0.0.0', port=5000, threaded=True)
    finally:
        video_stream.stop()

