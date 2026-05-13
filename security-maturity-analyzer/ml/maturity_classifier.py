"""
Clasificador MLP de Nivel de Madurez — sklearn MLPClassifier (sin TensorFlow)
Arquitectura: 24->64->32->16->6 (softmax)  Entrenado con datos sintéticos.
"""
import numpy as np
from typing import Dict, List, Tuple
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from analyzer.event_classifier import DomainStats
from rules.iso27001_controls import ISO27001_DOMAINS, MATURITY_LEVELS
 
np.random.seed(42)
N_DOMAINS, N_DOM_FEATS, N_CLASSES = 6, 4, 6
N_INPUT = N_DOMAINS * N_DOM_FEATS
 
class MaturityClassifier:
    """MLP 24->64->32->16->6 para clasificar nivel de madurez ISO 27001 (0-5)."""
    def __init__(self):
        self.scaler = StandardScaler()
        self.model_ = MLPClassifier(
            hidden_layer_sizes=(64, 32, 16),
            activation="relu",
            solver="adam",
            alpha=1e-4,
            batch_size=64,
            learning_rate_init=5e-4,
            max_iter=1,
            warm_start=True,
            random_state=42,
            early_stopping=True,
            validation_fraction=0.15,
            n_iter_no_change=8,
        )
        self.train_losses_, self.val_losses_   = [], []
        self.train_accs_,   self.val_accs_     = [], []
        self._fitted = False
 
    @staticmethod
    def domain_stats_to_vector(stats: Dict[str, DomainStats]) -> np.ndarray:
        ordered = list(ISO27001_DOMAINS.keys())
        max_ev  = max((s.total_events for s in stats.values()), default=1) or 1
        feats   = []
        for key in ordered:
            ds = stats.get(key)
            if ds is None or ds.total_events == 0:
                feats.extend([0.0, 1.0, 0.0, 0.0]); continue
            feats.extend([
                min(ds.total_events/max(max_ev,1), 1.0),
                min(ds.risk_rate, 1.0),
                min(np.log1p(ds.total_events)/8.0, 1.0),
                min((len(ds.unique_ips)+len(ds.unique_users))/40.0, 1.0),
            ])
        return np.array(feats, dtype=np.float32)
 
    @staticmethod
    def _synthetic_data(n_per=400):
        rng = np.random.default_rng(42)
        profiles = {
            0: (0.02,0.95,0.05,0.00,0.02), 1: (0.10,0.75,0.15,0.05,0.06),
            2: (0.25,0.50,0.35,0.15,0.08), 3: (0.50,0.25,0.55,0.35,0.10),
            4: (0.75,0.10,0.75,0.60,0.08), 5: (0.95,0.02,0.90,0.85,0.04),
        }
        X, y = [], []
        for lvl,(sc,rr,le,cv,std) in profiles.items():
            for _ in range(n_per):
                row=[]
                for _ in range(N_DOMAINS):
                    row.extend([float(np.clip(rng.normal(sc,std),0,1)),float(np.clip(rng.normal(rr,std),0,1)),
                                float(np.clip(rng.normal(le,std),0,1)),float(np.clip(rng.normal(cv,std),0,1))])
                X.append(row); y.append(lvl)
        X,y=np.array(X,dtype=np.float32),np.array(y,dtype=int)
        idx=rng.permutation(len(X)); return X[idx],y[idx]
 
    def fit(self, epochs=40, verbose=0):
        X, y = self._synthetic_data()
        Xs   = self.scaler.fit_transform(X)
        self.model_.max_iter = 1
        for ep in range(epochs):
            self.model_.fit(Xs, y)
            tl  = self.model_.loss_
            acc = float((self.model_.predict(Xs)==y).mean())
            vs  = float(getattr(self.model_,'best_validation_score_', acc))
            self.train_losses_.append(round(float(tl),4))
            self.val_losses_.append(round(1-vs if vs<=1 else float(tl),4))
            self.train_accs_.append(round(acc,4))
            self.val_accs_.append(round(vs,4))
        self._fitted = True
        return self
 
    def predict_proba(self, stats):
        x = self.domain_stats_to_vector(stats).reshape(1,-1)
        xs = self.scaler.transform(x)
        return self.model_.predict_proba(xs).flatten()
 
    def predict_level(self, stats):
        return int(np.argmax(self.predict_proba(stats)))
 
    def predict_with_confidence(self, stats):
        proba = self.predict_proba(stats)
        level = int(np.argmax(proba))
        return {
            "level": level,
            "level_name": MATURITY_LEVELS[level]["name"],
            "confidence": round(float(proba[level])*100, 1),
            "probabilities": {i: round(float(p)*100,1) for i,p in enumerate(proba)},
        }
 
    def summary(self):
        n_p = sum(w.size+b.size for w,b in zip(self.model_.coefs_,self.model_.intercepts_)) if self._fitted else 0
        return {
            "fitted": self._fitted,
            "architecture": f"{N_INPUT}→64→32→16→{N_CLASSES}",
            "parameters": n_p,
            "epochs_trained":     len(self.train_losses_),
            "final_train_loss":   self.train_losses_[-1]  if self.train_losses_  else None,
            "final_val_loss":     self.val_losses_[-1]    if self.val_losses_    else None,
            "final_val_accuracy": self.val_accs_[-1]      if self.val_accs_      else None,
        }
 
