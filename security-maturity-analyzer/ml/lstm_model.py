"""
LSTM para Detección de Amenazas en Secuencias Temporales
=========================================================
Arquitectura:
  Entrada: ventana de 20 eventos × 13 features numéricas
  LSTM(32) → Dropout(0.3) → LSTM(16) → Dense(8, relu) → Dense(1, sigmoid)

Detecta patrones temporales maliciosos:
  - Fuerza bruta: muchos fallos → éxito repentino
  - Reconocimiento: accesos a múltiples recursos en poco tiempo
  - Exfiltración: transferencias fuera de horario
  - Escalada de privilegios: cambios rápidos de usuario/nivel
"""

import os, warnings
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
warnings.filterwarnings("ignore")

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from typing import List, Tuple, Dict, Optional

from analyzer.log_parser import LogEntry
from ml.feature_extractor import LogFeatureExtractor, N_NUMERIC

tf.random.set_seed(42)
np.random.seed(42)

SEQ_LEN   = 20    # eventos por ventana
N_FEATURES = N_NUMERIC  # 13 features numéricas


class LSTMThreatDetector:
    """
    Detector de amenazas basado en LSTM bidireccional.

    Entrada: secuencias de 20 eventos (cada uno = 13 features numéricas).
    Salida: probabilidad de amenaza [0, 1].
    """

    def __init__(self):
        self.extractor = LogFeatureExtractor()
        self.model_: Optional[keras.Model] = None
        self.threat_threshold_ = 0.5
        self.train_losses_: List[float] = []
        self.val_losses_:   List[float] = []
        self.train_accs_:   List[float] = []
        self.val_accs_:     List[float] = []
        self._fitted = False

    # ── Model ─────────────────────────────────────────────────────────────────

    def _build_model(self) -> keras.Model:
        inp = keras.Input(shape=(SEQ_LEN, N_FEATURES), name="sequence_input")

        # Bidirectional LSTM captura patrones hacia adelante y atrás
        x = layers.Bidirectional(
            layers.LSTM(32, return_sequences=True, name="lstm_1"),
            name="bilstm_1"
        )(inp)
        x = layers.Dropout(0.3, name="dropout_1")(x)
        x = layers.LSTM(16, name="lstm_2")(x)
        x = layers.Dropout(0.2, name="dropout_2")(x)
        x = layers.Dense(8, activation="relu", name="dense_1")(x)
        out = layers.Dense(1, activation="sigmoid", name="threat_prob")(x)

        model = keras.Model(inputs=inp, outputs=out, name="LSTMThreatDetector")
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=5e-4),
            loss="binary_crossentropy",
            metrics=["accuracy", keras.metrics.AUC(name="auc")],
        )
        return model

    # ── Data helpers ──────────────────────────────────────────────────────────

    def _make_sequences(
        self, entries: List[LogEntry]
    ) -> np.ndarray:
        """Convierte lista de entries a matriz de secuencias (N, SEQ_LEN, N_FEATURES)."""
        features = self.extractor.transform_numeric_only(entries)  # (N, 13)
        seqs = []
        for i in range(len(features) - SEQ_LEN + 1):
            seqs.append(features[i : i + SEQ_LEN])
        if not seqs:
            # Padding si hay menos eventos que SEQ_LEN
            pad = np.zeros((SEQ_LEN - len(features), N_FEATURES), dtype=np.float32)
            seqs.append(np.vstack([features, pad]))
        return np.array(seqs, dtype=np.float32)

    # ── Training ──────────────────────────────────────────────────────────────

    def fit(
        self,
        normal_entries: List[LogEntry],
        attack_entries: List[LogEntry],
        epochs: int = 25,
        batch_size: int = 32,
        verbose: int = 0,
    ) -> "LSTMThreatDetector":
        """Entrenamiento supervisado: normal=0, ataque=1."""
        # Asegurar extractor ajustado
        if not self.extractor._fitted:
            all_entries = normal_entries + attack_entries
            self.extractor.fit(all_entries)

        X_normal = self._make_sequences(normal_entries)
        X_attack = self._make_sequences(attack_entries)

        y_normal = np.zeros(len(X_normal), dtype=np.float32)
        y_attack = np.ones (len(X_attack), dtype=np.float32)

        X = np.concatenate([X_normal, X_attack])
        y = np.concatenate([y_normal, y_attack])

        # Shuffle
        idx = np.random.permutation(len(X))
        X, y = X[idx], y[idx]

        self.model_ = self._build_model()
        history = self.model_.fit(
            X, y,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=0.2,
            shuffle=True,
            verbose=verbose,
            callbacks=[
                keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True),
                keras.callbacks.ReduceLROnPlateau(patience=3, factor=0.5, verbose=0),
            ],
            class_weight={0: 1.0, 1: 2.0},  # Penalizar más los falsos negativos
        )
        self.train_losses_ = history.history["loss"]
        self.val_losses_   = history.history.get("val_loss", [])
        self.train_accs_   = history.history.get("accuracy", [])
        self.val_accs_     = history.history.get("val_accuracy", [])
        self._fitted = True
        return self

    # ── Inference ─────────────────────────────────────────────────────────────

    def predict_threat_probs(self, entries: List[LogEntry]) -> np.ndarray:
        """Probabilidad de amenaza por ventana de 20 eventos (shape: N_seq,)."""
        seqs = self._make_sequences(entries)
        probs = self.model_.predict(seqs, verbose=0).flatten()
        return probs

    def overall_threat_level(self, entries: List[LogEntry]) -> Dict:
        """Resumen: nivel global de amenaza detectado en el log completo."""
        probs = self.predict_threat_probs(entries)
        high   = float((probs >= 0.75).mean() * 100)
        medium = float(((probs >= 0.50) & (probs < 0.75)).mean() * 100)
        low    = float((probs < 0.50).mean() * 100)
        return {
            "mean_threat_prob": float(probs.mean()),
            "max_threat_prob":  float(probs.max()),
            "pct_high_threat":  round(high, 1),
            "pct_medium_threat":round(medium, 1),
            "pct_low_threat":   round(low, 1),
            "total_sequences":  len(probs),
        }

    def summary(self) -> Dict:
        if not self._fitted:
            return {"fitted": False}
        final_acc = self.val_accs_[-1] if self.val_accs_ else None
        return {
            "fitted": True,
            "architecture": f"({SEQ_LEN}, {N_FEATURES}) → BiLSTM(32) → LSTM(16) → Dense(8) → Dense(1)",
            "parameters": self.model_.count_params(),
            "epochs_trained": len(self.train_losses_),
            "final_train_loss": round(self.train_losses_[-1], 4) if self.train_losses_ else None,
            "final_val_loss":   round(self.val_losses_[-1], 4)   if self.val_losses_   else None,
            "final_val_accuracy": round(final_acc, 4) if final_acc else None,
        }
