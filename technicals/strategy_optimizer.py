import logging
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from typing import Dict, Tuple, Optional, Any

# Configure basic logging for this module
logger = logging.getLogger(__name__)

class StrategyOptimizer:
    """
    A class for training machine learning models to optimize trading strategy parameters.
    Specifically:
      - A RandomForestRegressor to predict potential profit percentage.
      - A RandomForestClassifier to predict probability of a winning trade.
    """

    def __init__(self):
        """
        Initialize the optimizer with default models and a scaler for feature normalization.
        """
        self.scaler = StandardScaler()

        # Models for predicting profit and win/loss
        self.profit_model = RandomForestRegressor(n_estimators=100, random_state=42)
        self.win_model = RandomForestClassifier(n_estimators=100, random_state=42)

        # Bookkeeping
        self.best_parameters: Dict[str, Any] = {}
        self.performance_history: list = []
        self.is_trained: bool = False

    def prepare_training_data(self, technical_data: Dict[str, Dict[str, Any]]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Prepare training data from a dictionary of technical data. Each symbol’s data is a dict containing
        technical indicators plus an optional 'profit_pct' used as a direct training target.

        :param technical_data: 
            A dictionary keyed by symbol. 
            Example:
            {
                "AAPL": {
                    "RSI": 45.0,
                    "ADX": 25.5,
                    "MACD": { "hist": 0.05, "signal": 0.01, "macd": 0.04 },
                    "BB": { "upper": 150.0, "lower": 130.0, "middle": 140.0 },
                    "MA_DATA": { "SMA_50": 135.0, "SMA_200": 130.0 },
                    "OBV": 123456789,
                    "volume": 50000000,
                    "avg_volume": 48000000,
                    "profit_pct": 5.0       # optional if you have actual trade data
                },
                ...
            }
        :return: (features, targets_profit, targets_win)
            - features is an NxM NumPy array (N samples, M features).
            - targets_profit is length N array of float profit values.
            - targets_win is length N array of 0/1 indicating loss/win.
        """
        features = []
        targets_profit = []
        targets_win = []

        for symbol, data in technical_data.items():
            try:
                # Extract necessary feature values
                rsi_value = float(data.get("RSI", 50))
                adx_value = float(data.get("ADX", 25))

                macd_dict = data.get("MACD", {})
                macd_hist = float(macd_dict.get("hist", 0.0))

                bb_dict = data.get("BB", {})
                bb_upper = float(bb_dict.get("upper", 0.0))
                bb_lower = float(bb_dict.get("lower", 0.0))
                bb_width = bb_upper - bb_lower

                ma_data = data.get("MA_DATA", {})
                sma_50 = float(ma_data.get("SMA_50", 0.0))

                obv_value = float(data.get("OBV", 0.0))
                volume_value = float(data.get("volume", 0.0))

                # Consolidate features
                row_features = [
                    rsi_value,
                    adx_value,
                    macd_hist,
                    bb_width,
                    sma_50,
                    obv_value,
                    volume_value
                ]

                # Check for NaNs
                if any(np.isnan(x) for x in row_features):
                    logger.warning(f"Skipping {symbol} due to NaN in features.")
                    continue

                # Append features
                features.append(row_features)

                # If actual profit is provided, use it; otherwise default to 0
                profit = float(data.get("profit_pct", 0.0))
                # Mark as win if profit > 0
                is_win = 1 if profit > 0 else 0

                targets_profit.append(profit)
                targets_win.append(is_win)

            except (ValueError, TypeError) as e:
                logger.warning(f"Skipping {symbol} due to invalid data: {e}")
                continue

        if not features:
            raise ValueError("No valid training data was found to train the model.")

        return np.array(features), np.array(targets_profit), np.array(targets_win)

    def train_model(self, technical_data: Dict[str, Dict[str, Any]]) -> bool:
        """
        Train the profit regression and win-probability classification models
        on the provided technical data.

        :param technical_data: Dictionary as described in prepare_training_data().
        :return: True if the training was successful, False otherwise.
        """
        try:
            X, y_profit, y_win = self.prepare_training_data(technical_data)

            if len(X) < 2:
                logger.warning("Not enough samples to train models.")
                return False

            # Check that there's at least one winning and one losing trade
            unique_labels = np.unique(y_win)
            if len(unique_labels) < 2:
                logger.warning(
                    f"Only one unique class ({unique_labels}) in y_win. "
                    "Binary classification requires both 0 and 1. Skipping training."
                )
                return False

            # Scale features
            X_scaled = self.scaler.fit_transform(X)

            # Train the models
            self.profit_model.fit(X_scaled, y_profit)
            self.win_model.fit(X_scaled, y_win)

            # Get feature importances and report them
            profit_importances = self.profit_model.feature_importances_
            win_importances = self.win_model.feature_importances_
            self._report_feature_importances(profit_importances, win_importances)

            # Suggest which indicators might need improvement
            self.suggest_indicator_improvements(profit_importances, win_importances, threshold=0.05)

            self.is_trained = True
            logger.info("Model training completed successfully.")
            return True

        except ValueError as e:
            logger.error(f"Error in training model: {e}", exc_info=True)
            return False
        except Exception as e:
            logger.error(f"Unexpected error in training model: {e}", exc_info=True)
            return False

    def _report_feature_importances(self, profit_importances: np.ndarray, win_importances: np.ndarray) -> None:
        """
        Report the feature importances from both models in descending order.

        :param profit_importances: Feature importances for the regression model.
        :param win_importances: Feature importances for the classification model.
        """
        features = ["RSI", "ADX", "MACD_Hist", "BB_Width", "SMA_50", "OBV", "Volume"]

        logger.info("=== Profit Model Feature Importances ===")
        sorted_idx_profit = np.argsort(profit_importances)[::-1]
        for idx in sorted_idx_profit:
            logger.info(f"{features[idx]}: {profit_importances[idx]:.4f}")

        logger.info("=== Win Model Feature Importances ===")
        sorted_idx_win = np.argsort(win_importances)[::-1]
        for idx in sorted_idx_win:
            logger.info(f"{features[idx]}: {win_importances[idx]:.4f}")

    def suggest_indicator_improvements(self,
                                       profit_importances: np.ndarray,
                                       win_importances: np.ndarray,
                                       threshold: float = 0.05) -> None:
        """
        Suggest improvements based on feature importance. 
        If an indicator's combined importance is below threshold, we warn that it may be unhelpful.

        :param profit_importances: Feature importances array from the regressor.
        :param win_importances: Feature importances array from the classifier.
        :param threshold: Cutoff for "low importance."
        """
        features = ["RSI", "ADX", "MACD_Hist", "BB_Width", "SMA_50", "OBV", "Volume"]
        # Average the two importance arrays, so we get a single measure
        combined_importances = (profit_importances + win_importances) / 2.0

        suggestions = []
        for i, imp in enumerate(combined_importances):
            if imp < threshold:
                suggestions.append(
                    f"{features[i]} is below importance threshold ({imp:.4f}). "
                    "Consider refining or removing."
                )

        if suggestions:
            logger.info("=== Indicator Improvement Suggestions ===")
            for s in suggestions:
                logger.info(s)
        else:
            logger.info("All indicators are above the importance threshold. No immediate improvements suggested.")

    def optimize_strategy_parameters(self, technical_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Given a single dictionary of technical features (e.g., for a recent data point),
        predict profit and win probability. Then derive 'optimized' strategy parameters
        such as confidence threshold, position size, stop_loss, etc.

        :param technical_data: Dictionary of a single data point's indicators.
        :return: Dictionary with optimized parameters or None if there's an error.
        """
        if not self.is_trained:
            logger.warning("Attempting to optimize before training the models.")
            return None

        try:
            # Extract features from the single dictionary
            features = self._prepare_single_feature_vector(technical_data)
            if features is None:
                logger.warning("Feature preparation returned None. Skipping optimization.")
                return None

            # Scale features
            X_scaled = self.scaler.transform([features])

            # Predictions
            profit_pred = float(self.profit_model.predict(X_scaled)[0])

            # Safely retrieve predicted probability of a win
            pred_probs = self.win_model.predict_proba(X_scaled)
            if pred_probs.shape[1] == 2:
                win_prob = float(pred_probs[0][1])
            else:
                # single-class fallback
                single_class = self.win_model.classes_[0]
                if single_class == 1:
                    win_prob = 1.0
                else:
                    win_prob = 0.0

            optimized_params = {
                'confidence_threshold': self._optimize_confidence_threshold(win_prob),
                'position_size': self._optimize_position_size(profit_pred, win_prob),
                'stop_loss': self._optimize_stop_loss(profit_pred),
                'take_profit': self._optimize_take_profit(profit_pred),
                'predicted_profit': profit_pred,
                'win_probability': win_prob
            }

            return optimized_params

        except Exception as e:
            logger.error(f"Error optimizing strategy: {e}", exc_info=True)
            return None

    def _prepare_single_feature_vector(self, data: Dict[str, Any]) -> Optional[list]:
        """
        Turn a single technical_data dict into a list of numeric features, matching the model's feature order.

        :param data: e.g. {
            "RSI": 45.0,
            "ADX": 25.5,
            "MACD": { "hist": 0.05, ...},
            "BB": {"upper": 150, "lower": 130, ...},
            "MA_DATA": {"SMA_50": 135.0, ...},
            "OBV": 123456,
            "volume": 50000000
        }
        :return: List of numeric features or None if data is invalid.
        """
        try:
            rsi = float(data.get("RSI", 50.0))
            adx = float(data.get("ADX", 25.0))

            macd_dict = data.get("MACD", {})
            macd_hist = float(macd_dict.get("hist", 0.0))

            bb_dict = data.get("BB", {})
            bb_upper = float(bb_dict.get("upper", 0.0))
            bb_lower = float(bb_dict.get("lower", 0.0))
            bb_width = bb_upper - bb_lower

            ma_data = data.get("MA_DATA", {})
            sma_50 = float(ma_data.get("SMA_50", 0.0))

            obv = float(data.get("OBV", 0.0))
            volume = float(data.get("volume", 0.0))

            features = [
                rsi,
                adx,
                macd_hist,
                bb_width,
                sma_50,
                obv,
                volume
            ]
            if any(np.isnan(x) for x in features):
                logger.warning("Encountered NaN in single feature vector.")
                return None
            return features

        except Exception as e:
            logger.error(f"Failed to build single feature vector: {e}", exc_info=True)
            return None

    def _optimize_confidence_threshold(self, win_probability: float) -> float:
        """
        Example function to set confidence threshold based on model's predicted win probability.
        
        :param win_probability: Probability of winning trade from classifier.
        :return: A float in [0.5, 0.9] range as confidence threshold.
        """
        base_threshold = 0.7
        adjustment = (win_probability - 0.5) * 0.2
        return max(0.5, min(0.9, base_threshold - adjustment))

    def _optimize_position_size(self, predicted_profit: float, win_probability: float) -> float:
        """
        Adjust position size in proportion to predicted profit and win probability.
        
        :param predicted_profit: Predicted profit from regressor (in %).
        :param win_probability: Probability of winning the trade.
        :return: Position size as fraction of capital, e.g., 0.02 meaning 2%.
        """
        base_size = 0.02
        profit_factor = min(2.0, max(0.5, abs(predicted_profit) / 5.0))
        probability_factor = min(2.0, max(0.5, win_probability * 2.0))
        return base_size * profit_factor * probability_factor

    def _optimize_stop_loss(self, predicted_profit: float) -> float:
        """
        Adjust stop_loss level in response to expected volatility or profit potential.

        :param predicted_profit: Predicted profit in %.
        :return: A float in [0.05, 0.25] as the stop loss fraction.
        """
        base_stop = 0.15
        volatility_adjustment = abs(predicted_profit) / 10.0
        return max(0.05, min(0.25, base_stop + volatility_adjustment))

    def _optimize_take_profit(self, predicted_profit: float) -> float:
        """
        Adjust take_profit level in response to predicted profit.

        :param predicted_profit: Predicted profit in %.
        :return: A float in [0.15, 0.5] as the take profit fraction.
        """
        base_profit = 0.25
        profit_adjustment = predicted_profit / 5.0
        return max(0.15, min(0.50, base_profit + profit_adjustment))

    def generate_strategy_recommendations(self, technical_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        High-level wrapper that calls optimize_strategy_parameters and returns a
        dictionary containing recommended parameters and suggested strategy adjustments.

        :param technical_data: Single data point (dict) for which we want a recommendation.
        :return: Dict with 'parameters' key containing the optimized params, or None if error.
        """
        optimized_params = self.optimize_strategy_parameters(technical_data)
        if not optimized_params:
            return None

        recommendations = {
            'parameters': optimized_params,
            'strategy_adjustments': []
        }

        # Example logic: If win_probability > 0.6, pick a bullish or bearish spread
        # based on predicted profit sign. Add more elaborate logic as needed.
        if optimized_params['win_probability'] > 0.6:
            if optimized_params['predicted_profit'] > 0:
                recommendations['strategy_adjustments'].append({
                    'type': 'BULL_CALL_SPREAD',
                    'confidence_threshold': optimized_params['confidence_threshold'],
                    'position_size': optimized_params['position_size'],
                    'stop_loss': optimized_params['stop_loss'],
                    'take_profit': optimized_params['take_profit'],
                    'rationale': 'High probability bullish setup detected'
                })
            else:
                recommendations['strategy_adjustments'].append({
                    'type': 'BEAR_PUT_SPREAD',
                    'confidence_threshold': optimized_params['confidence_threshold'],
                    'position_size': optimized_params['position_size'],
                    'stop_loss': optimized_params['stop_loss'],
                    'take_profit': optimized_params['take_profit'],
                    'rationale': 'High probability bearish setup detected'
                })

        return recommendations
