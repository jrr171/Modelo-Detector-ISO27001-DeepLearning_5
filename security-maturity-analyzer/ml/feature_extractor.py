"""
Feature Extractor
Convierte LogEntry objetos a vectores numéricos para alimentar los modelos de DL.

Produce dos tipos de features:
  - Numéricas (13 dims): hora, tipo de log, nivel, keywords, etc.
  - TF-IDF de texto (50 dims): representación semántica del mensaje
  - Combinado (63 dims): ambos concatenados → entrada del Autoencoder
"""

import re
import math
import numpy as np
from typing import List, Tuple, Dict
from collections import defaultdict

from analyzer.log_parser import LogEntry

# ── Constantes ───────────────────────────────────────────────────────────────

FAILURE_RE = re.compile(
    r'\b(fail|error|denied|invalid|refused|reject|block|drop|'
    r'critical|panic|abort|unauthoriz|breach|attack|intrusion)\b', re.I
)
SUCCESS_RE = re.compile(
    r'\b(accept|success|ok|allow|permit|establish|complet|'
    r'authenticat.*success|logged.on|session.open)\b', re.I
)
SSH_RE      = re.compile(r'\b(ssh|sshd|publickey|password.*ssh)\b', re.I)
WEB_RE      = re.compile(r'\b(http|https|nginx|apache|GET|POST|PUT)\b', re.I)
DB_RE       = re.compile(r'\b(mysql|postgresql|oracle|sql|database|db)\b', re.I)
FW_RE       = re.compile(r'\b(firewall|iptables|ufw|DROP|ACCEPT|DENY)\b', re.I)
CRYPTO_RE   = re.compile(r'\b(ssl|tls|aes|rsa|cipher|encrypt|decrypt|certificate)\b', re.I)
INCIDENT_RE = re.compile(r'\b(incident|alarm|alert|ticket|escalat|notification)\b', re.I)

LEVEL_MAP = {"DEBUG": 0.0, "INFO": 0.2, "WARNING": 0.5, "ERROR": 0.8, "CRITICAL": 1.0}

N_TFIDF  = 50   # dimensiones TF-IDF
N_NUMERIC = 13  # features numéricas
N_TOTAL   = N_NUMERIC + N_TFIDF   # 63 total


class LogFeatureExtractor:
    """
    Extrae features numéricas y TF-IDF de una lista de LogEntry.
    El vocabulario TF-IDF se construye con fit() antes de transform().
    """

    def __init__(self, max_vocab: int = N_TFIDF):
        self.max_vocab = max_vocab
        self.vocab_: Dict[str, int] = {}
        self.idf_: np.ndarray = np.array([])
        self._fitted = False

    # ── Tokenization ─────────────────────────────────────────────────────────

    @staticmethod
    def _tokenize(text: str) -> List[str]:
        """Extraer tokens significativos del mensaje."""
        text = text.lower()
        tokens = re.findall(r'[a-z]{3,}', text)
        stopwords = {"the","and","for","from","with","this","that","are","was",
                     "has","have","been","not","its","our","you","can","will",
                     "all","but","also","into","out","more","any","over","than"}
        return [t for t in tokens if t not in stopwords]

    # ── Numeric features ─────────────────────────────────────────────────────

    @staticmethod
    def _numeric_features(entry: LogEntry) -> np.ndarray:
        """13 features numéricas normalizadas."""
        msg = entry.message or ""

        hour = (entry.timestamp.hour / 23.0) if entry.timestamp else 0.5
        night = 1.0 if entry.timestamp and (entry.timestamp.hour < 6 or entry.timestamp.hour > 22) else 0.0
        level_val   = LEVEL_MAP.get(entry.level, 0.2)
        has_failure = 1.0 if FAILURE_RE.search(msg) else 0.0
        has_success = 1.0 if SUCCESS_RE.search(msg) else 0.0
        is_ssh      = 1.0 if SSH_RE.search(msg)     else 0.0
        is_web      = 1.0 if WEB_RE.search(msg)     else 0.0
        is_db       = 1.0 if DB_RE.search(msg)      else 0.0
        is_fw       = 1.0 if FW_RE.search(msg)      else 0.0
        is_crypto   = 1.0 if CRYPTO_RE.search(msg)  else 0.0
        is_incident = 1.0 if INCIDENT_RE.search(msg)else 0.0
        has_ip      = 1.0 if entry.source_ip         else 0.0
        msg_len     = min(len(msg) / 300.0, 1.0)

        return np.array([
            hour, night, level_val, has_failure, has_success,
            is_ssh, is_web, is_db, is_fw, is_crypto,
            is_incident, has_ip, msg_len,
        ], dtype=np.float32)

    # ── TF-IDF ────────────────────────────────────────────────────────────────

    def fit(self, entries: List[LogEntry]) -> "LogFeatureExtractor":
        """Construir vocabulario TF-IDF."""
        tf_counts: Dict[str, int] = defaultdict(int)
        df_counts: Dict[str, int] = defaultdict(int)
        n_docs = len(entries)

        for entry in entries:
            tokens = set(self._tokenize(entry.message or ""))
            for t in self._tokenize(entry.message or ""):
                tf_counts[t] += 1
            for t in tokens:
                df_counts[t] += 1

        # Seleccionar top-K por frecuencia (excl. muy raros y demasiado comunes)
        min_df = max(2, int(n_docs * 0.001))
        max_df = int(n_docs * 0.9)
        candidates = {
            t: c for t, c in tf_counts.items()
            if min_df <= df_counts[t] <= max_df
        }
        top_vocab = sorted(candidates, key=candidates.get, reverse=True)[:self.max_vocab]

        self.vocab_ = {t: i for i, t in enumerate(top_vocab)}
        idf_vals = []
        for t in top_vocab:
            idf_vals.append(math.log((n_docs + 1) / (df_counts[t] + 1)) + 1)
        self.idf_ = np.array(idf_vals, dtype=np.float32)
        # Pad if vocabulary smaller than max_vocab
        if len(self.idf_) < self.max_vocab:
            pad = self.max_vocab - len(self.idf_)
            self.idf_ = np.concatenate([self.idf_, np.ones(pad, dtype=np.float32)])
        self._fitted = True
        return self

    def _tfidf_vector(self, entry: LogEntry) -> np.ndarray:
        """TF-IDF vector of length max_vocab."""
        vec = np.zeros(self.max_vocab, dtype=np.float32)
        if not self._fitted:
            return vec
        tokens = self._tokenize(entry.message or "")
        if not tokens:
            return vec
        tf: Dict[str, float] = defaultdict(float)
        for t in tokens:
            if t in self.vocab_:
                tf[t] += 1.0
        for t, cnt in tf.items():
            idx = self.vocab_[t]
            vec[idx] = (cnt / len(tokens)) * self.idf_[idx]
        # L2 normalise
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec /= norm
        return vec

    def transform_one(self, entry: LogEntry) -> np.ndarray:
        """Devuelve vector de 63 features para una entrada."""
        num = self._numeric_features(entry)
        tfidf = self._tfidf_vector(entry)
        return np.concatenate([num, tfidf])

    def transform(self, entries: List[LogEntry]) -> np.ndarray:
        """Devuelve matriz (N, 63)."""
        return np.vstack([self.transform_one(e) for e in entries])

    def transform_numeric_only(self, entries: List[LogEntry]) -> np.ndarray:
        """Solo features numéricas (N, 13) — para el LSTM."""
        return np.vstack([self._numeric_features(e) for e in entries])

    def fit_transform(self, entries: List[LogEntry]) -> np.ndarray:
        self.fit(entries)
        return self.transform(entries)

    @property
    def feature_names(self) -> List[str]:
        num_names = [
            "hora_dia","horario_nocturno","nivel_log","keyword_fallo",
            "keyword_exito","es_ssh","es_web","es_bd","es_firewall",
            "es_crypto","es_incidente","tiene_ip","longitud_msg",
        ]
        tfidf_names = [f"tfidf_{t}" for t in list(self.vocab_.keys())[:self.max_vocab]]
        tfidf_names += [f"tfidf_pad_{i}" for i in range(self.max_vocab - len(tfidf_names))]
        return num_names + tfidf_names
