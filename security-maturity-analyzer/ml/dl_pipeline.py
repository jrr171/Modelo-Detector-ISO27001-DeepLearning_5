"""
Pipeline de Deep Learning
Orquesta el entrenamiento y la inferencia de los 3 modelos DL:
  1. Autoencoder     — detección de anomalías
  2. LSTM Detector   — detección de amenazas temporales
  3. MLP Classifier  — clasificación de nivel de madurez

Genera también los datos sintéticos de entrenamiento del LSTM
a partir de los eventos reales disponibles.
"""

import random
import numpy as np
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass, field

from analyzer.log_parser       import LogEntry
from analyzer.event_classifier import DomainStats
from analyzer.maturity_scorer  import MaturityResult
from ml.feature_extractor      import LogFeatureExtractor
from ml.autoencoder_model      import LogAutoencoder
from ml.lstm_model             import LSTMThreatDetector
from ml.maturity_classifier    import MaturityClassifier
from rules.iso27001_controls   import MATURITY_LEVELS

random.seed(42)
np.random.seed(42)


@dataclass
class DLResult:
    """Resultados consolidados de los 3 modelos DL."""
    # Autoencoder
    anomaly_scores:     np.ndarray = field(default_factory=lambda: np.array([]))
    is_anomaly:         np.ndarray = field(default_factory=lambda: np.array([]))
    anomaly_rate:       float = 0.0
    autoencoder_threshold: float = 0.0
    ae_train_loss:      List[float] = field(default_factory=list)
    ae_val_loss:        List[float] = field(default_factory=list)
    ae_summary:         Dict = field(default_factory=dict)

    # LSTM
    threat_probs:       np.ndarray = field(default_factory=lambda: np.array([]))
    threat_level:       Dict = field(default_factory=dict)
    lstm_train_loss:    List[float] = field(default_factory=list)
    lstm_val_loss:      List[float] = field(default_factory=list)
    lstm_train_acc:     List[float] = field(default_factory=list)
    lstm_val_acc:       List[float] = field(default_factory=list)
    lstm_summary:       Dict = field(default_factory=dict)

    # MLP Classifier
    dl_predicted_level: int = 0
    dl_level_name:      str = ""
    dl_confidence:      float = 0.0
    dl_probabilities:   Dict = field(default_factory=dict)
    mlp_train_loss:     List[float] = field(default_factory=list)
    mlp_val_loss:       List[float] = field(default_factory=list)
    mlp_train_acc:      List[float] = field(default_factory=list)
    mlp_val_acc:        List[float] = field(default_factory=list)
    mlp_summary:        Dict = field(default_factory=dict)

    # Comparación
    rule_based_level:   int = 0
    rule_based_score:   float = 0.0
    agreement:          bool = False
    dl_adjusted_score:  float = 0.0   # Score corregido por anomalías


def _separate_normal_attack(entries: List[LogEntry]) -> Tuple[List[LogEntry], List[LogEntry]]:
    """
    Separa eventos normales de potencialmente maliciosos usando
    heurísticas de keywords (sin modelos, solo para generar train data del LSTM).
    """
    import re
    attack_re = re.compile(
        r'\b(fail|invalid|denied|refused|attack|brute|flood|'
        r'scan|exploit|malware|ransomware|block|drop)\b', re.I
    )
    normal, attack = [], []
    for e in entries:
        if attack_re.search(e.message or ""):
            attack.append(e)
        else:
            normal.append(e)
    return normal, attack


def _augment_attack_entries(attack: List[LogEntry], normal: List[LogEntry],
                              target: int = 200) -> List[LogEntry]:
    """Si hay pocos ataques, genera más combinando con ruido."""
    if len(attack) >= target:
        return attack
    # Duplicar con pequeñas variaciones (simula más ataques)
    augmented = list(attack)
    pool = list(attack) if attack else list(normal[:50])
    while len(augmented) < target and pool:
        e = random.choice(pool)
        augmented.append(e)
    return augmented


class DLPipeline:
    """
    Pipeline completo: entrena los 3 modelos y ejecuta inferencia.
    Diseñado para ser cacheado con @st.cache_resource en Streamlit.
    """

    def __init__(self):
        self.autoencoder  = LogAutoencoder()
        self.lstm         = LSTMThreatDetector()
        self.classifier   = MaturityClassifier()
        self._trained     = False

    def train_all(
        self,
        entries: List[LogEntry],
        ae_epochs:   int = 25,
        lstm_epochs: int = 20,
        mlp_epochs:  int = 35,
        verbose:     int = 0,
    ) -> "DLPipeline":
        """Entrena los 3 modelos secuencialmente."""
        normal, attack = _separate_normal_attack(entries)

        # Garantizar mínimos para entrenamiento
        if len(normal) < 50:
            normal = entries   # usar todo si no hay suficientes normales
        if len(attack) < 30:
            attack = _augment_attack_entries(attack, normal, target=max(50, len(normal)//3))

        # ── 1. Autoencoder ────────────────────────────────────────────────
        self.autoencoder.fit(normal, epochs=ae_epochs, verbose=verbose)

        # ── 2. LSTM — usa el extractor ya entrenado del autoencoder ────────
        self.lstm.extractor = self.autoencoder.extractor
        self.lstm.extractor._fitted = True
        self.lstm.fit(normal, attack, epochs=lstm_epochs, verbose=verbose)

        # ── 3. MLP Classifier — datos sintéticos ──────────────────────────
        self.classifier.fit(epochs=mlp_epochs, verbose=verbose)

        self._trained = True
        return self

    def run(
        self,
        entries: List[LogEntry],
        domain_stats: Dict[str, DomainStats],
        maturity_result: MaturityResult,
    ) -> DLResult:
        """Ejecuta inferencia y construye DLResult."""
        res = DLResult(
            rule_based_level=maturity_result.overall_level,
            rule_based_score=maturity_result.overall_score,
        )

        # ── Autoencoder ───────────────────────────────────────────────────
        scores = self.autoencoder.anomaly_scores(entries)
        is_anom = self.autoencoder.predict_anomalies(entries)
        res.anomaly_scores        = scores
        res.is_anomaly            = is_anom
        res.anomaly_rate          = float(is_anom.mean() * 100)
        res.autoencoder_threshold = self.autoencoder.threshold_
        res.ae_train_loss         = self.autoencoder.train_losses_
        res.ae_val_loss           = self.autoencoder.val_losses_
        res.ae_summary            = self.autoencoder.summary()

        # ── LSTM ──────────────────────────────────────────────────────────
        threat_probs = self.lstm.predict_threat_probs(entries)
        res.threat_probs    = threat_probs
        res.threat_level    = self.lstm.overall_threat_level(entries)
        res.lstm_train_loss = self.lstm.train_losses_
        res.lstm_val_loss   = self.lstm.val_losses_
        res.lstm_train_acc  = self.lstm.train_accs_
        res.lstm_val_acc    = self.lstm.val_accs_
        res.lstm_summary    = self.lstm.summary()

        # ── MLP Classifier ───────────────────────────────────────────────
        pred = self.classifier.predict_with_confidence(domain_stats)
        res.dl_predicted_level = pred["level"]
        res.dl_level_name      = pred["level_name"]
        res.dl_confidence      = pred["confidence"]
        res.dl_probabilities   = pred["probabilities"]
        res.mlp_train_loss     = self.classifier.train_losses_
        res.mlp_val_loss       = self.classifier.val_losses_
        res.mlp_train_acc      = self.classifier.train_accs_
        res.mlp_val_acc        = self.classifier.val_accs_
        res.mlp_summary        = self.classifier.summary()

        # ── Score ajustado por anomalías ──────────────────────────────────
        # Penalizar score basado en tasa de anomalías detectadas por el AE
        anomaly_penalty = res.anomaly_rate * 0.3   # máximo -30 pts si 100% anómalo
        res.dl_adjusted_score = max(0, maturity_result.overall_score - anomaly_penalty)

        # ── Concordancia reglas vs DL ────────────────────────────────────
        res.agreement = (res.dl_predicted_level == maturity_result.overall_level)

        return res
