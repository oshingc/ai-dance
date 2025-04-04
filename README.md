# AI DANCE

## Requisitos Previos

- Python 3.8 o superior
- Node.js y npm
- Webcam funcional

## Instalación

### Backend (Python)

1. Instalar dependencias del backend:
```bash
pip install flask flask-cors opencv-python mediapipe numpy
```

### Frontend (React)

1. Instalar dependencias del frontend:
```bash
npm install
```

## Ejecutar la Aplicación

1. Iniciar el backend (en una terminal):
```bash
cd backend
python3 server.py
```

2. Iniciar el frontend (en otra terminal):
```bash
npm start
```

La aplicación se abrirá automáticamente en tu navegador predeterminado en `http://localhost:3000`
Recomendación de compatibilidad: Usar Chrome


## Notas Importantes

- Asegúrate de que tu cámara no esté siendo utilizada por otra aplicación
- El backend corre en el puerto 5000
- El frontend corre en el puerto 3000
- Permite el acceso a la cámara cuando el navegador lo solicite

## Solución de Problemas

Si encuentras una pantalla negra o la cámara no funciona:
1. Verifica que tu cámara funcione en otras aplicaciones
2. Reinicia el servidor backend
3. Limpia el caché del navegador
4. Asegúrate de que no haya otras aplicaciones usando la cámara


https://github.com/user-attachments/assets/ed72fcb3-1ca9-4d5e-ad04-5bfa1c5314d3


