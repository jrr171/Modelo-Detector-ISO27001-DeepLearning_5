"""
Clasificador MLP de Nivel de Madurez ISO 27001
===============================================
Arquitectura:
  Entrada (24 features de dominio) → Dense(64, relu) → Dropout(0.3)
  → Dense(32, relu) → Dense(16, relu) → Dense(6, softmax)

Predice el nivel de madurez COBIT (0–5) directamente desde
estadísticas agregadas por dominio ISO 27001, sin necesidad de
aplicar reglas manuales.

Features de entrada (24):
  Por cada uno de los 6 dominios ISO 27001 (4 features cada uno):
    - score normalizado [0,1]
    - tasa de riesgo [0,1]
    - log(total_eventos + 1) normalizado
    - cobertura IPs+usuarios normalizada
"""

import os, warnings
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
warnings.filterwarnings("ignore")

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, regularizers
from typing import List, Dict, Optional, Tuple

from analyzer.event_classifier import DomainStats
from rules.iso27001_controls   import ISO27001_DOMAINS, get_maturity_level

tf.random.set_seed(42)
np.random.seed(42)

N_DOMAINS   = 6
N_DOM_FEATS = 4
N_INPUT     = N_DOMAINS * N_DOM_FEATS   # 24
N_CLASSES   = 6                          # niveles 0–5


class MaturityClassifier:
    """
    Red neuronal MLP para clasificación de nivel de madurez.

    Comparar con el método basado en reglas para validación cruzada
    y análisis de concordancia — sección clave para la tesis.
    """

    def __init__(self):
        self.model_: Optional[keras.Model] = None
        self.train_losses_:    List[float] = []
        self.val_losses_:      List[float] = []
        self.train_accs_:      List[float] = []
        self.val_accs_:        List[float] = []
        self._fitted = False

    # ── Feature engineering ──────────────────────────────────────────────────

    @staticmethod
    def domain_stats_to_vector(stats: Dict[str, DomainStats]) -> np.ndarray:
        """
        Convierte las estadísticas de dominios a vector de 24 features.
        Orden de dominios: access_control, operations, communications,
                           incident_management, cryptography, physical_security
        """
        ordered_keys = list(ISO27001_DOMAINS.keys())
        feats = []
        max_events = max((s.total_events for s in stats.values()), default=1) or 1

        for key in ordered_keys:
            ds = stats.get(key)
            if ds is None or ds.total_events == 0:
                feats.extend([0.0, 1.0, 0.0, 0.0])
                continue
            score_norm  = min(ds.total_events / max(max_events, 1), 1.0)
            risk_rate   = min(ds.risk_rate, 1.0)
            log_events  = min(np.log1p(ds.total_events) / 8.0, 1.0)  # log(3000)/8 ≈ 1
            coverage    = min((len(ds.unique_ips) + len(ds.unique_users)) / 40.0, 1.0)
            feats.extend([score_norm, risk_rate, log_events, coverage])

        return np.array(feats, dtype=np.float32)

    # ── Model ─────────────────────────────────────────────────────────────────

    def _build_model(self) -> keras.Model:
        inp = keras.Input(shape=(N_INPUT,), name="domain_features")

        x = layers.Dense(64, activation="relu",
                          kernel_regularizer=regularizers.l2(1e-4), name="h1")(inp)
        x = layers.BatchNormalization(name="bn1")(x)
        x = layers.Dropout(0.3, name="drop1")(x)
        x = layers.Dense(32, activation="relu", name="h2")(x)
        x = layers.BatchNormalization(name="bn2")(x)
        x = layers.Dropout(0.2, name="drop2")(x)
        x = layers.Dense(16, activation="relu", name="h3")(x)
        out = layers.Dense(N_CLASSES, activation="softmax", name="maturity_probs")(x)

        model = keras.Model(inputs=inp, outputs=out, name="MaturityClassifier")
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=5e-4),
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"],
        )
        return model

    # ── Synthetic training data ───────────────────────────────────────────────

    @staticmethod
    def _generate_synthetic_data(n_per_class: int = 300) -> Tuple[np.ndarray, np.ndarray]:
        """
        Genera datos sintéticos de entrenamiento basados en perfiles
        de seguridad conocidos para cada nivel de madurez.
        """
        rng = np.random.default_rng(42)
        X_list, y_list = [], []

        profiles = {
            # Nivel → (score_norm, risk_rate, log_events, coverage) medias y std
            0: dict(score=0.02, risk=0.95, events=0.05, cov=0.00, std=0.02),
            1: dict(score=0.10, risk=0.75, events=0.15, cov=0.05, std=0.06),
            2: dict(score=0.25, risk=0.50, events=0.35, cov=0.15, std=0.08),
            3: dict(score=0.50, risk=0.25, events=0.55, cov=0.35, std=0.10),
            4: dict(score=0.75, risk=0.10, events=0.75, cov=0.60, std=0.08),
            5: dict(score=0.95, risk=0.02, events=0.90, cov=0.85, std=0.04),
        }

        for level, prof in profiles.items():
            for _ in range(n_per_class):
                row = []
                for _ in range(N_DOMAINS):
                    sn  = float(np.clip(rng.normal(prof["score"],  prof["std"]), 0, 1))
                    rr  = float(np.clip(rng.normal(prof["risk"],   prof["std"]), 0, 1))
                    le  = float(np.clip(rng.normal(prof["events"], prof["std"]), 0, 1))
                    cov = float(np.clip(rng.normal(prof["cov"],    prof["std"]), 0, 1))
                    row.extend([sn, rr, le, cov])
                X_list.append(row)
                y_list.append(level)

        X = np.array(X_list, dtype=np.float32)
        y = np.array(y_list, dtype=np.int32)
        idx = rng.permutation(len(X))
        return X[idx], y[idx]

    # ── Training ──────────────────────────────────────────────────────────────

    def fit(self, epochs: int = 40, verbose: int = 0) -> "MaturityClassifier":
        X, y = self._generate_synthetic_data(n_per_class=400)
        self.model_ = self._build_model()
        history = self.model_.fit(
            X, y,
            epochs=epochs,
            batch_size=64,
            validation_split=0.2,
            shuffle=True,
            verbose=verbose,
            callbacks=[
                keras.callbacks.EarlyStopping(patience=8, restore_best_weights=True),
                keras.callbacks.ReduceLROnPlateau(patience=4, factor=0.5, verbose=0),
            ],
        )
        self.train_losses_ = history.history["loss"]
        self.val_losses_   = history.history.get("val_loss", [])
        self.train_accs_   = history.history.get("accuracy", [])
        self.val_accs_     = history.history.get("val_accuracy", [])
        self._fitted = True
        return self

    # ── Inference ─────────────────────────────────────────────────────────────

    def predict_proba(self, stats: Dict[str, DomainStats]) -> np.ndarray:
        """Probabilidades para cada nivel 0–5 (shape: 6,)."""
        x = self.domain_stats_to_vector(stats).reshape(1, -1)
        return self.model_.predict(x, verbose=0).flatten()

    def predict_level(self, stats: Dict[str, DomainStats]) -> int:
        """Nivel de madurez predicho (0–5)."""
        return int(np.argmax(self.predict_proba(stats)))

    def predict_with_confidence(self, stats: Dict[str, DomainStats]) -> Dict:
        proba = self.predict_proba(stats)
        level = int(np.argmax(proba))
        confidence = float(proba[level])
        from rules.iso27001_controls import MATURITY_LEVELS
        return {
            "level":        level,
            "level_name":   MATURITY_LEVELS[level]["name"],
            "confidence":   round(confidence * 100, 1),
            "probabilities": {i: round(float(p) * 100, 1) for i, p in enumerate(proba)},
        }

    def summary(self) -> Dict:
        if not self._fitted:
            return {"fitted": False}
        return {
            "fitted": True,
            "architecture": f"{N_INPUT} → 64 → 32 → 16 → {N_CLASSES}",
            "parameters":   self.model_.count_params(),
            "epochs_trained":     len(self.train_losses_),
            "final_train_loss":   round(self.train_losses_[-1], 4) if self.train_losses_ else None,
            "final_val_loss":     round(self.val_losses_[-1], 4)   if self.val_losses_   else None,
            "final_val_accuracy": round(self.val_accs_[-1], 4)     if self.val_accs_     else None,
        }
