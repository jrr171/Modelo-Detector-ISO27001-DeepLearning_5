"""
Autoencoder para Detección de Anomalías en Logs
================================================
Arquitectura:
  Entrada (63) → Encoder [63→32→16→8] → Bottleneck (8) → Decoder [8→16→32→63]

Principio:
  Entrenado solo con eventos NORMALES.
  Error de reconstrucción alto → evento ANÓMALO → incidente de seguridad.

Métricas exportadas:
  - anomaly_score (MSE) por evento
  - threshold óptimo (percentil 95 del train set)
  - tasa de anomalías detectadas
"""

import os, warnings
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
warnings.filterwarnings("ignore")

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, regularizers
from typing import List, Tuple, Dict, Optional

from analyzer.log_parser import LogEntry
from ml.feature_extractor import LogFeatureExtractor, N_TOTAL

# Reproducibilidad
tf.random.set_seed(42)
np.random.seed(42)


class LogAutoencoder:
    """
    Autoencoder Variacional (VAE-lite) para detección de anomalías en logs.

    Arquitectura:
      Encoder: Dense(63→32, relu) → BatchNorm → Dense(32→16, relu) → Dense(16→8)
      Decoder: Dense(8→16, relu) → BatchNorm → Dense(16→32, relu) → Dense(32→63, sigmoid)
    """

    INPUT_DIM   = N_TOTAL   # 63
    BOTTLENECK  = 8
    HIDDEN      = [32, 16]
    THRESHOLD_PERCENTILE = 95

    def __init__(self):
        self.extractor   = LogFeatureExtractor()
        self.model_: Optional[keras.Model] = None
        self.threshold_  = 0.05
        self.train_losses_: List[float] = []
        self.val_losses_:   List[float] = []
        self._fitted = False

    # ── Build ─────────────────────────────────────────────────────────────────

    def _build_model(self) -> keras.Model:
        inp = keras.Input(shape=(self.INPUT_DIM,), name="log_input")

        # Encoder
        x = layers.Dense(32, activation="relu",
                          kernel_regularizer=regularizers.l2(1e-4), name="enc_1")(inp)
        x = layers.BatchNormalization(name="bn_enc")(x)
        x = layers.Dense(16, activation="relu", name="enc_2")(x)
        bottleneck = layers.Dense(self.BOTTLENECK, activation="relu",
                                   name="bottleneck")(x)

        # Decoder
        x = layers.Dense(16, activation="relu", name="dec_1")(bottleneck)
        x = layers.BatchNormalization(name="bn_dec")(x)
        x = layers.Dense(32, activation="relu", name="dec_2")(x)
        out = layers.Dense(self.INPUT_DIM, activation="sigmoid", name="reconstruction")(x)

        model = keras.Model(inputs=inp, outputs=out, name="LogAutoencoder")
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=1e-3),
            loss="mse",
            metrics=["mae"],
        )
        return model

    # ── Training ──────────────────────────────────────────────────────────────

    def fit(
        self,
        normal_entries: List[LogEntry],
        epochs: int = 30,
        batch_size: int = 64,
        validation_split: float = 0.15,
        verbose: int = 0,
    ) -> "LogAutoencoder":
        """Entrenar con eventos normales."""
        X = self.extractor.fit_transform(normal_entries)

        # Normalizar a [0,1] (necesario para activación sigmoid en salida)
        self._min = X.min(axis=0)
        self._max = X.max(axis=0) + 1e-9
        X_norm = (X - self._min) / (self._max - self._min)

        self.model_ = self._build_model()
        history = self.model_.fit(
            X_norm, X_norm,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=validation_split,
            shuffle=True,
            verbose=verbose,
            callbacks=[
                keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True, monitor="val_loss"),
                keras.callbacks.ReduceLROnPlateau(patience=3, factor=0.5, verbose=0),
            ],
        )
        self.train_losses_ = history.history["loss"]
        self.val_losses_   = history.history.get("val_loss", [])

        # Calcular threshold con percentil sobre set de entrenamiento
        recon = self.model_.predict(X_norm, verbose=0)
        mse_train = np.mean((X_norm - recon) ** 2, axis=1)
        self.threshold_ = float(np.percentile(mse_train, self.THRESHOLD_PERCENTILE))
        self._fitted = True
        return self

    # ── Inference ─────────────────────────────────────────────────────────────

    def _preprocess(self, entries: List[LogEntry]) -> np.ndarray:
        X = self.extractor.transform(entries)
        return (X - self._min) / (self._max - self._min)

    def reconstruction_errors(self, entries: List[LogEntry]) -> np.ndarray:
        """MSE de reconstrucción por evento (shape: N,)."""
        X_norm = self._preprocess(entries)
        recon  = self.model_.predict(X_norm, verbose=0)
        return np.mean((X_norm - recon) ** 2, axis=1)

    def anomaly_scores(self, entries: List[LogEntry]) -> np.ndarray:
        """Score normalizado 0–100 (100 = máxima anomalía)."""
        errors = self.reconstruction_errors(entries)
        # Normalizar respecto al threshold
        scores = np.clip(errors / (self.threshold_ * 2), 0, 1) * 100
        return scores

    def predict_anomalies(self, entries: List[LogEntry]) -> np.ndarray:
        """Bool array: True si el evento es anómalo."""
        return self.reconstruction_errors(entries) > self.threshold_

    def anomaly_rate(self, entries: List[LogEntry]) -> float:
        return float(self.predict_anomalies(entries).mean())

    def summary(self) -> Dict:
        if not self._fitted:
            return {"fitted": False}
        return {
            "fitted": True,
            "architecture": "63 → 32 → 16 → 8 → 16 → 32 → 63",
            "parameters": self.model_.count_params(),
            "threshold": round(self.threshold_, 6),
            "epochs_trained": len(self.train_losses_),
            "final_train_loss": round(self.train_losses_[-1], 6) if self.train_losses_ else None,
            "final_val_loss":   round(self.val_losses_[-1], 6)   if self.val_losses_   else None,
        }
