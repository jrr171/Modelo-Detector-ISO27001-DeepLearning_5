"""
Autoencoder para Detección de Anomalías — NumPy puro (sin TensorFlow)
Arquitectura: 63->32->16->8->16->32->63  Adam optimizer + Early stopping
"""
import numpy as np
from typing import List, Dict
from analyzer.log_parser import LogEntry
from ml.feature_extractor import LogFeatureExtractor, N_TOTAL
 
np.random.seed(42)
 
def relu(x):      return np.maximum(0, x)
def relu_d(x):    return (x > 0).astype(np.float32)
def sigmoid(x):   return 1.0 / (1.0 + np.exp(-np.clip(x, -20, 20)))
def sigmoid_d(x): s = sigmoid(x); return s * (1 - s)
 
class LogAutoencoder:
    LAYERS = [N_TOTAL, 32, 16, 8, 16, 32, N_TOTAL]
    THRESHOLD_PERCENTILE = 95
 
    def __init__(self, lr=0.001, l2=1e-4):
        self.lr, self.l2 = lr, l2
        self.extractor = LogFeatureExtractor()
        self.W, self.b = [], []
        self.threshold_ = 0.05
        self.train_losses_, self.val_losses_ = [], []
        self._fitted = False
        self._min = self._max = None
        self._init_weights()
 
    def _init_weights(self):
        self.W, self.b = [], []
        for i in range(len(self.LAYERS)-1):
            fi, fo = self.LAYERS[i], self.LAYERS[i+1]
            self.W.append((np.random.randn(fi, fo) * np.sqrt(2.0/(fi+fo))).astype(np.float32))
            self.b.append(np.zeros((1, fo), dtype=np.float32))
 
    def _forward(self, X):
        acts, zs = [X], []
        for i,(W,b) in enumerate(zip(self.W, self.b)):
            z = acts[-1] @ W + b; zs.append(z)
            acts.append(sigmoid(z) if i==len(self.W)-1 else relu(z))
        return acts, zs
 
    def _backward(self, acts, zs, Xn):
        n, nL = Xn.shape[0], len(self.W)
        gW, gb = [None]*nL, [None]*nL
        delta = (acts[-1]-Xn) * sigmoid_d(zs[-1]) * 2 / n
        for i in reversed(range(nL)):
            gW[i] = acts[i].T @ delta + self.l2*self.W[i]
            gb[i] = delta.sum(axis=0, keepdims=True)
            if i > 0: delta = (delta @ self.W[i].T) * relu_d(zs[i-1])
        return gW, gb
 
    def fit(self, normal_entries, epochs=30, batch_size=64, verbose=0, validation_split=0.15):
        X = self.extractor.fit_transform(normal_entries)
        self._min = X.min(0); self._max = X.max(0)+1e-9
        Xn = ((X-self._min)/(self._max-self._min)).astype(np.float32)
        n_val = max(1, int(len(Xn)*validation_split))
        idx = np.random.permutation(len(Xn))
        Xval, Xtr = Xn[idx[:n_val]], Xn[idx[n_val:]]
        mW=[np.zeros_like(w) for w in self.W]; vW=[np.zeros_like(w) for w in self.W]
        mb=[np.zeros_like(b) for b in self.b]; vb=[np.zeros_like(b) for b in self.b]
        b1,b2,eps=0.9,0.999,1e-8
        best_val,no_imp,bestW,bestb = np.inf,0,[w.copy() for w in self.W],[b.copy() for b in self.b]
        for ep in range(1, epochs+1):
            perm=np.random.permutation(len(Xtr)); el=0; nb=0
            for s in range(0,len(Xtr),batch_size):
                batch=Xtr[perm[s:s+batch_size]]; acts,zs=self._forward(batch)
                el+=float(np.mean((acts[-1]-batch)**2)); nb+=1
                gW,gb=self._backward(acts,zs,batch)
                for i in range(len(self.W)):
                    mW[i]=b1*mW[i]+(1-b1)*gW[i]; vW[i]=b2*vW[i]+(1-b2)*gW[i]**2
                    mb[i]=b1*mb[i]+(1-b1)*gb[i]; vb[i]=b2*vb[i]+(1-b2)*gb[i]**2
                    mWh=mW[i]/(1-b1**ep); vWh=vW[i]/(1-b2**ep)
                    mbh=mb[i]/(1-b1**ep); vbh=vb[i]/(1-b2**ep)
                    self.W[i]-=self.lr*mWh/(np.sqrt(vWh)+eps)
                    self.b[i]-=self.lr*mbh/(np.sqrt(vbh)+eps)
            tl=el/max(nb,1); va,_=self._forward(Xval); vl=float(np.mean((va[-1]-Xval)**2))
            self.train_losses_.append(round(tl,6)); self.val_losses_.append(round(vl,6))
            if vl<best_val-1e-6: best_val=vl; no_imp=0; bestW=[w.copy() for w in self.W]; bestb=[b.copy() for b in self.b]
            else:
                no_imp+=1
                if no_imp>=5: break
        self.W,self.b=bestW,bestb
        atr,_=self._forward(Xtr); mse=np.mean((atr[-1]-Xtr)**2,axis=1)
        self.threshold_=float(np.percentile(mse,self.THRESHOLD_PERCENTILE)); self._fitted=True
        return self
 
    def _pre(self, entries):
        X=self.extractor.transform(entries)
        return ((X-self._min)/(self._max-self._min)).astype(np.float32)
 
    def reconstruction_errors(self, entries):
        Xn=self._pre(entries); a,_=self._forward(Xn); return np.mean((a[-1]-Xn)**2,axis=1)
 
    def anomaly_scores(self, entries):
        return np.clip(self.reconstruction_errors(entries)/(self.threshold_*2),0,1)*100
 
    def predict_anomalies(self, entries):
        return self.reconstruction_errors(entries)>self.threshold_
 
    def anomaly_rate(self, entries):
        return float(self.predict_anomalies(entries).mean())
 
    def summary(self):
        np_=sum(w.size+b.size for w,b in zip(self.W,self.b))
        return {"fitted":self._fitted,"architecture":"63→32→16→8→16→32→63",
                "parameters":np_,"threshold":round(self.threshold_,6) if self._fitted else None,
                "epochs_trained":len(self.train_losses_),
                "final_train_loss":self.train_losses_[-1] if self.train_losses_ else None,
                "final_val_loss":self.val_losses_[-1] if self.val_losses_ else None}
 
