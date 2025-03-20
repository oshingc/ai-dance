from flask import jsonify
from pose_detector import PoseDetector  # Importar la clase

pose_detector = PoseDetector()  # Crear instancia global

@app.route('/get_similarity')
def get_similarity():
    return jsonify({
        "similarity": pose_detector.get_similarity()
    }) 