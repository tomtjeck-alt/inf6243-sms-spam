"""
models.py
---------
Définition des modèles et de leurs grilles d'hyperparamètres.

Modèles classiques (scikit-learn), tous sous forme de Pipeline TF-IDF -> classificateur :
    - Naive Bayes multinomial
    - Régression logistique
    - SVM linéaire
    - Random Forest
    - KNN (distance cosinus)

Modèle d'apprentissage profond (PyTorch) :
    - BiLSTM sur des séquences de mots (embedding appris)
"""

import re
import time
from collections import Counter

import numpy as np
import torch
import torch.nn as nn
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

RANDOM_STATE = 42


# ---------------------------------------------------------------------------
# 1. Modèles classiques + grilles pour GridSearchCV
# ---------------------------------------------------------------------------
def get_classical_models():
    """Retourne {nom: (pipeline, grille_hyperparamètres)}.

    Les clés de grille 'tfidf__...' règlent la vectorisation, 'clf__...' le modèle.
    """
    def tfidf(**kw):
        return TfidfVectorizer(sublinear_tf=True, **kw)

    models = {
        "Naive Bayes": (
            Pipeline([("tfidf", tfidf()), ("clf", MultinomialNB())]),
            {
                "tfidf__ngram_range": [(1, 1), (1, 2)],
                "tfidf__min_df": [1, 2],
                "clf__alpha": [0.01, 0.05, 0.1, 0.5, 1.0],
            },
        ),
        "Régression logistique": (
            Pipeline([("tfidf", tfidf()),
                      ("clf", LogisticRegression(max_iter=2000, solver="liblinear",
                                                 random_state=RANDOM_STATE))]),
            {
                "tfidf__ngram_range": [(1, 1), (1, 2)],
                "clf__C": [0.1, 1, 10, 100],
                "clf__class_weight": [None, "balanced"],
            },
        ),
        "SVM linéaire": (
            Pipeline([("tfidf", tfidf()),
                      ("clf", LinearSVC(random_state=RANDOM_STATE))]),
            {
                "tfidf__ngram_range": [(1, 1), (1, 2)],
                "clf__C": [0.01, 0.1, 1, 10],
                "clf__class_weight": [None, "balanced"],
            },
        ),
        "Random Forest": (
            Pipeline([("tfidf", tfidf(min_df=2)),
                      ("clf", RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=-1))]),
            {
                "clf__n_estimators": [200, 400],
                "clf__max_depth": [None, 50],
                "clf__class_weight": [None, "balanced_subsample"],
            },
        ),
        "KNN": (
            Pipeline([("tfidf", tfidf()),
                      ("clf", KNeighborsClassifier(metric="cosine", algorithm="brute"))]),
            {
                "clf__n_neighbors": [1, 3, 5, 9, 15],
                "clf__weights": ["uniform", "distance"],
            },
        ),
    }
    return models


def get_scores(model, X):
    """Score continu (proba ou distance à l'hyperplan) pour les courbes ROC / PR."""
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, 1]
    return model.decision_function(X)


# ---------------------------------------------------------------------------
# 2. Apprentissage profond : BiLSTM (PyTorch)
# ---------------------------------------------------------------------------
_TOKEN_RE = re.compile(r"[a-z0-9_!]+")


def tokenize(text):
    return _TOKEN_RE.findall(text)


class Vocab:
    """Vocabulaire construit sur l'ensemble d'entraînement uniquement."""
    PAD, UNK = 0, 1

    def __init__(self, texts, min_freq=2):
        counts = Counter(tok for t in texts for tok in tokenize(t))
        self.itos = ["<pad>", "<unk>"] + sorted(w for w, c in counts.items() if c >= min_freq)
        self.stoi = {w: i for i, w in enumerate(self.itos)}

    def __len__(self):
        return len(self.itos)

    def encode(self, texts, max_len=60):
        arr = np.zeros((len(texts), max_len), dtype=np.int64)
        for i, t in enumerate(texts):
            ids = [self.stoi.get(tok, self.UNK) for tok in tokenize(t)][:max_len]
            arr[i, :len(ids)] = ids
        return arr


class BiLSTMClassifier(nn.Module):
    """Embedding -> BiLSTM -> max-pooling temporel -> dropout -> couche linéaire."""

    def __init__(self, vocab_size, emb_dim=64, hidden=64, dropout=0.3):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, emb_dim, padding_idx=0)
        self.lstm = nn.LSTM(emb_dim, hidden, batch_first=True, bidirectional=True)
        self.drop = nn.Dropout(dropout)
        self.fc = nn.Linear(2 * hidden, 1)

    def forward(self, x):
        mask = (x != 0).unsqueeze(-1)                 # ignore le padding
        h, _ = self.lstm(self.emb(x))
        h = h.masked_fill(~mask, -1e4).max(dim=1).values
        return self.fc(self.drop(h)).squeeze(-1)      # logits


def train_bilstm(X_tr, y_tr, X_val, y_val, vocab_size, emb_dim=64, hidden=64, dropout=0.3,
                 lr=1e-3, epochs=15, batch_size=64, patience=3, seed=RANDOM_STATE,
                 verbose=False):
    """Entraîne le BiLSTM avec arrêt précoce sur la perte de validation.

    X_* : tableaux d'indices (n, max_len) ; y_* : 0/1 ; vocab_size = len(Vocab).
    patience=None désactive l'arrêt précoce (entraînement final à nombre d'epochs fixé ;
    X_val/y_val ne servent alors qu'à l'affichage).
    Retourne (modèle_au_meilleur_epoch, historique).
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = BiLSTMClassifier(vocab_size, emb_dim, hidden, dropout)
    return _fit(model, X_tr, y_tr, X_val, y_val, lr, epochs, batch_size, patience, verbose)


def _fit(model, X_tr, y_tr, X_val, y_val, lr, epochs, batch_size, patience, verbose):
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    # pos_weight compense le déséquilibre (≈ 6.5 ham pour 1 spam)
    pos_weight = torch.tensor((y_tr == 0).sum() / max((y_tr == 1).sum(), 1), dtype=torch.float32)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    Xt, yt = torch.from_numpy(X_tr), torch.from_numpy(y_tr.astype(np.float32))
    Xv, yv = torch.from_numpy(X_val), torch.from_numpy(y_val.astype(np.float32))

    hist = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}
    best, best_state, bad = np.inf, None, 0
    for ep in range(epochs):
        t0 = time.time()
        model.train()
        perm = torch.randperm(len(Xt))
        tot, correct = 0.0, 0
        for i in range(0, len(Xt), batch_size):
            idx = perm[i:i + batch_size]
            opt.zero_grad()
            logits = model(Xt[idx])
            loss = loss_fn(logits, yt[idx])
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tot += loss.item() * len(idx)
            correct += ((logits > 0).float() == yt[idx]).sum().item()
        model.eval()
        with torch.no_grad():
            lv = model(Xv)
            vloss = loss_fn(lv, yv).item()
            vacc = ((lv > 0).float() == yv).float().mean().item()
        hist["train_loss"].append(tot / len(Xt))
        hist["train_acc"].append(correct / len(Xt))
        hist["val_loss"].append(vloss)
        hist["val_acc"].append(vacc)
        if verbose:
            print(f"epoch {ep + 1:2d}  train_loss={hist['train_loss'][-1]:.4f}  "
                  f"val_loss={vloss:.4f}  val_acc={vacc:.4f}  ({time.time() - t0:.1f}s)")
        if patience is None:                        # pas d'arrêt précoce : on garde le dernier epoch
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        elif vloss < best - 1e-4:
            best, bad = vloss, 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= patience:
                break
    model.load_state_dict(best_state)
    hist["best_epoch"] = (len(hist["val_loss"]) if patience is None
                          else int(np.argmin(hist["val_loss"])) + 1)
    return model, hist


def predict_bilstm(model, X):
    """Retourne (probabilités spam, prédictions 0/1)."""
    model.eval()
    with torch.no_grad():
        p = torch.sigmoid(model(torch.from_numpy(X))).numpy()
    return p, (p >= 0.5).astype(int)
