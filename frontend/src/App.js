import React, { useState, useRef, useCallback, useEffect } from 'react';
import './App.css';

const BACKEND_CONFIG = {
  BASE_URL: 'http://localhost:5000',
  ENDPOINTS: {
    VIDEO_FEED: '/video_feed',
    START_CAMERA: '/start_camera',
    STOP_CAMERA: '/stop_camera'
  }
};

function App() {
  const [isStarted, setIsStarted] = useState(false);
  const [countdown, setCountdown] = useState(null);
  const [similarity, setSimilarity] = useState(0);
  const [similarityHistory, setSimilarityHistory] = useState([]);
  const [showResults, setShowResults] = useState(false);
  const [averageSimilarity, setAverageSimilarity] = useState(0);
  const videoRef = useRef(null);

  const handleStart = async () => {
    try {
      // Primero iniciamos la cámara y esperamos respuesta
      const response = await fetch(`${BACKEND_CONFIG.BASE_URL}${BACKEND_CONFIG.ENDPOINTS.START_CAMERA}`);
      if (!response.ok) throw new Error('Error al iniciar cámara');

      // Activamos la cámara inmediatamente
      setIsStarted(true);

      // Esperamos un momento para asegurar que la cámara esté activa
      await new Promise(resolve => setTimeout(resolve, 100));

      // Iniciamos el countdown
      let count = 5;
      setCountdown(count);
      
      const countdownInterval = setInterval(() => {
        count--;
        if (count < 0) {
          clearInterval(countdownInterval);
          setCountdown(null);
          if (videoRef.current) {
            videoRef.current.currentTime = 0;
            videoRef.current.play();
          }
        } else {
          setCountdown(count);
        }
      }, 1000);

    } catch (error) {
      console.error('Error:', error);
      setIsStarted(false);
    }
  };

  const handleStop = async () => {
    try {
      await fetch(`${BACKEND_CONFIG.BASE_URL}${BACKEND_CONFIG.ENDPOINTS.STOP_CAMERA}`);
      setIsStarted(false);
      setCountdown(null);
      if (videoRef.current) {
        videoRef.current.pause();
      }
      
      // Filtrar valores no válidos y ceros
      const validScores = similarityHistory.filter(score => 
        Number.isFinite(score) && score > 0 && score <= 100
      );
      
      // Si no hay scores válidos, mostrar 0
      const avg = validScores.length 
        ? Math.round(validScores.reduce((a, b) => a + b, 0) / validScores.length)
        : 0;
      
      // Si no hay scores válidos, mostrar 0
      const bestScore = validScores.length 
        ? Math.max(...validScores)
        : 0;
      
      setAverageSimilarity(avg);
      setShowResults(true);
    } catch (error) {
      console.error('Error al detener:', error);
    }
  };

  // Función para obtener la similitud
  const fetchSimilarity = useCallback(async () => {
    if (!isStarted) return;
    try {
      const response = await fetch(`${BACKEND_CONFIG.BASE_URL}/get_similarity`);
      const data = await response.json();
      // Asegurar que similarity sea un número entero válido
      const validSimilarity = Number.isFinite(data.similarity) ? 
        Math.round(Math.max(0, Math.min(100, data.similarity))) : 50;
      setSimilarity(validSimilarity);
    } catch (error) {
      console.error('Error fetching similarity:', error);
      setSimilarity(50);
    }
  }, [isStarted]);

  // Actualizar similitud cada 100ms cuando está activo
  useEffect(() => {
    if (!isStarted) return;
    const intervalId = setInterval(fetchSimilarity, 100);
    return () => clearInterval(intervalId);
  }, [isStarted, fetchSimilarity]);

  // Actualizar historial de similitud - solo guardar valores válidos y reales
  useEffect(() => {
    if (isStarted && Number.isFinite(similarity) && similarity > 0 && similarity <= 100) {
      // Solo guardar si el valor es significativamente diferente del anterior
      setSimilarityHistory(prev => {
        const lastValue = prev[prev.length - 1];
        // Si es un valor nuevo y diferente, lo guardamos
        if (!lastValue || Math.abs(lastValue - similarity) > 1) {
          return [...prev, similarity];
        }
        return prev;
      });
    }
  }, [similarity, isStarted]);

  return (
    <div className="app">
      <div className="controls">
        <button 
          onClick={isStarted ? handleStop : handleStart}
          className={`control-button ${isStarted ? 'stop' : 'start'}`}
        >
          {isStarted ? 'DETENER' : 'INICIAR'}
        </button>
      </div>

      <div className="similarity-panel">
        {isStarted && (
          <div className="similarity-display">
            Precisión: {Number.isFinite(similarity) ? Math.round(similarity) : 50}%
          </div>
        )}
      </div>

      {showResults && !isStarted && (
        <div className="results-overlay">
          <div className="results-card">
            <h2>Resultados del Baile</h2>
            <div className="score">
              <span className="score-label">Precisión Promedio:</span>
              <span className="score-value">{Math.round(averageSimilarity)}%</span>
            </div>
            <div className="score">
              <span className="score-label">Mejor Momento:</span>
              <span className="score-value">
                {similarityHistory.length > 0 
                  ? Math.round(Math.max(...similarityHistory.filter(s => 
                      s > 0 && s < 100 && Number.isFinite(s)
                    ))) || 0
                  : 0}%
              </span>
            </div>
            <button 
              className="control-button start"
              onClick={() => {
                setShowResults(false);
                setSimilarityHistory([]);
              }}
            >
              Intentar de Nuevo
            </button>
          </div>
        </div>
      )}

      <div className="main-content">
        <div className="webcam-container">
          {isStarted && (
            <>
              <img 
                key={Date.now()}
                src={`${BACKEND_CONFIG.BASE_URL}${BACKEND_CONFIG.ENDPOINTS.VIDEO_FEED}`}
                alt="Webcam"
                className="webcam-feed"
                style={{ 
                  width: '100%',
                  height: '100%',
                  objectFit: 'contain'
                }}
              />
              <div className="overlay-stats">
                <div className="similarity-value">
                  {Math.round(similarity)}%
                </div>
              </div>
            </>
          )}
          {countdown && (
            <div className="countdown-overlay">
              {countdown}
            </div>
          )}
        </div>

        <div className="video-container">
          <video 
            ref={videoRef}
            src="/videos/armagedon_dance.mp4"
            className="reference-video"
            controls
          />
        </div>
      </div>
    </div>
  );
}

export default App;