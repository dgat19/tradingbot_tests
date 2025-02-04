// src/components/Prediction.js
import React, { useState } from 'react';
import { predict } from '../services/api';

const Prediction = () => {
  const [symbol, setSymbol] = useState('');
  const [prediction, setPrediction] = useState(null);
  const [error, setError] = useState(null);

  const handlePredict = async () => {
    try {
      const data = await predict(symbol);
      setPrediction(data);
      setError(null);
    } catch (err) {
      setError('Failed to fetch prediction.');
      setPrediction(null);
    }
  };

  return (
    <div>
      <h2>Predict Stock Movement</h2>
      <input
        type="text"
        placeholder="Enter Stock Symbol"
        value={symbol}
        onChange={(e) => setSymbol(e.target.value)}
      />
      <button onClick={handlePredict}>Predict</button>

      {prediction && (
        <div>
          <h3>Prediction for {prediction.symbol}</h3>
          <p>Prediction: {prediction.prediction === 1 ? 'Buy' : 'Sell'}</p>
          <p>Confidence: {(prediction.confidence * 100).toFixed(2)}%</p>
        </div>
      )}

      {error && <p style={{ color: 'red' }}>{error}</p>}
    </div>
  );
};

export default Prediction;
