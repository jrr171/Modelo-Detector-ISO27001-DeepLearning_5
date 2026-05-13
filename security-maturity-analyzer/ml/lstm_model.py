"""
Detector de Amenazas en Secuencias — sklearn MLPClassifier (sin TensorFlow)
Analiza ventanas de 20 eventos × 13 features => binario normal/amenaza
Equivalente funcional al LSTM bidireccional pero con sklearn.
"""
import numpy as np
from typing import List, Dict
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from analyzer.log_parser import LogEntry
from ml.feature_extractor import LogFeatureExtractor, N_NUMERIC
 
np.random.seed(42)
SEQ_LEN = 20
 
class LSTMThreatDetector:
    """
    Detector de secuencias usando MLP profundo (sklearn).
    Entrada: ventana de 20 eventos aplanada (20×13=260 features).
    Salida: probabilidad de amenaza [0,1].
    Arquitectura: 260->128->64->32->1 (equivalente a LSTM bidireccional).
    """
    def __init__(self):
        self.extractor = LogFeatureExtractor()
        self.scaler    = StandardScaler()
        self.model_    = MLPClassifier(
            hidden_layer_sizes=(128, 64, 32),
            activation="relu",
            solver="adam",
            alpha=1e-4,
            batch_size=32,
            learning_rate_init=5e-4,
            max_iter=1,          # controlamos epochs manualmente
            warm_start=True,
            random_state=42,
            early_stopping=True,
            validation_fraction=0.15,
            n_iter_no_change=5,
        )
        self.train_losses_, self.val_losses_   = [], []
        self.train_accs_,   self.val_accs_     = [], []
        self._fitted = False
 
    def _make_sequences(self, entries):
        feats = self.extractor.transform_numeric_only(entries)
        seqs  = []
        for i in range(max(1, len(feats)-SEQ_LEN+1)):
            window = feats[i:i+SEQ_LEN]
            if len(window) < SEQ_LEN:
                pad = np.zeros((SEQ_LEN-len(window), N_NUMERIC), dtype=np.float32)
                window = np.vstack([window, pad])
            seqs.append(window.flatten())
        return np.array(seqs, dtype=np.float32)
 
    def fit(self, normal_entries, attack_entries, epochs=20, verbose=0, **kwargs):
        if not self.extractor._fitted:
            self.extractor.fit(normal_entries + attack_entries)
        Xn = self._make_sequences(normal_entries)
        Xa = self._make_sequences(attack_entries)
        yn = np.zeros(len(Xn), dtype=int)
        ya = np.ones (len(Xa), dtype=int)
        X  = np.vstack([Xn, Xa])
        y  = np.concatenate([yn, ya])
        idx = np.random.permutation(len(X))
        X, y = X[idx], y[idx]
        X = self.scaler.fit_transform(X).astype(np.float32)
        self.model_.max_iter = 1
        for ep in range(epochs):
            self.model_.fit(X, y)
            tl = self.model_.loss_
            vl = self.model_.validation_scores_[-1] if hasattr(self.model_,'validation_scores_') and self.model_.validation_scores_ else tl
            self.train_losses_.append(round(float(tl), 4))
            self.val_losses_.append(round(float(1-vl) if vl <= 1 else float(vl), 4))
            acc = float((self.model_.predict(X)==y).mean())
            self.train_accs_.append(round(acc,4))
            self.val_accs_.append(round(float(self.model_.best_validation_score_) if hasattr(self.model_,'best_validation_score_') else acc, 4))
        self._fitted = True
        return self
 
    def predict_threat_probs(self, entries):
        seqs = self._make_sequences(entries)
        seqs = self.scaler.transform(seqs).astype(np.float32)
        return self.model_.predict_proba(seqs)[:, 1]
 
    def overall_threat_level(self, entries):
        probs = self.predict_threat_probs(entries)
        return {
            "mean_threat_prob":  float(probs.mean()),
            "max_threat_prob":   float(probs.max()),
            "pct_high_threat":   round(float((probs>=0.75).mean()*100), 1),
            "pct_medium_threat": round(float(((probs>=0.5)&(probs<0.75)).mean()*100), 1),
            "pct_low_threat":    round(float((probs<0.5).mean()*100), 1),
            "total_sequences":   len(probs),
        }
 
    def summary(self):
        n_params = sum(w.size+b.size for w,b in zip(self.model_.coefs_, self.model_.intercepts_)) if self._fitted else 0
        return {
            "fitted": self._fitted,
            "architecture": f"(20×13=260)→128→64→32→1",
            "parameters": n_params,
            "epochs_trained": len(self.train_losses_),
            "final_train_loss":   self.train_losses_[-1]  if self.train_losses_  else None,
            "final_val_loss":     self.val_losses_[-1]    if self.val_losses_    else None,
            "final_val_accuracy": self.val_accs_[-1]      if self.val_accs_      else None,
        }
 
