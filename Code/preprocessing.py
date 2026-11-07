"""
preprocessing.py
----------------
Chargement, nettoyage et séparation du jeu de données SMS Spam Collection (UCI).

Fonctions principales :
    load_data(path)            -> DataFrame (label, message, y)
    clean_text(text)           -> texte normalisé (minuscules, jetons spéciaux)
    add_meta_features(df)      -> ajoute des attributs numériques (longueur, chiffres, ...)
    split_data(df, ...)        -> train / validation / test stratifiés
"""

import re
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

RANDOM_STATE = 42

# Expressions régulières utilisées pour normaliser les messages.
# Plutôt que de supprimer les URLs, numéros, montants, on les remplace par un
# jeton générique : leur PRÉSENCE est très informative pour détecter le spam,
# mais leur valeur exacte ne l'est pas (et ferait exploser le vocabulaire).
_URL_RE = re.compile(r"(https?://\S+|www\.\S+|\S+\.(com|net|org|co\.uk)\S*)", re.I)
_EMAIL_RE = re.compile(r"\S+@\S+\.\S+")
_PHONE_RE = re.compile(r"\b\d{5,}\b")              # numéros longs (téléphones, codes courts)
_MONEY_RE = re.compile(r"[£$€]\s?\d+([.,]\d+)?|\d+([.,]\d+)?\s?(p|pence|pounds?)\b", re.I)
_NUM_RE = re.compile(r"\b\d+\b")
_NON_ALNUM_RE = re.compile(r"[^a-z0-9_\s!]")
_SPACES_RE = re.compile(r"\s+")


def load_data(path):
    """Charge le fichier SMSSpamCollection (TSV : label \\t message)."""
    path = Path(path)
    df = pd.read_csv(path, sep="\t", header=None, names=["label", "message"],
                     quoting=3, encoding="utf-8")  # quoting=3 : ignorer les guillemets
    df["y"] = (df["label"] == "spam").astype(int)  # 1 = spam, 0 = ham
    return df


def clean_text(text):
    """Normalise un SMS : minuscules + remplacement des entités par des jetons."""
    t = text.lower()
    t = _URL_RE.sub(" urltoken ", t)
    t = _EMAIL_RE.sub(" emailtoken ", t)
    t = _MONEY_RE.sub(" moneytoken ", t)
    t = _PHONE_RE.sub(" phonetoken ", t)
    t = _NUM_RE.sub(" numtoken ", t)
    t = _NON_ALNUM_RE.sub(" ", t)       # ponctuation retirée (sauf '!')
    t = t.replace("!", " ! ")
    return _SPACES_RE.sub(" ", t).strip()


def add_meta_features(df, text_col="message"):
    """Ajoute des attributs numériques calculés sur le texte BRUT (pour l'EDA)."""
    s = df[text_col]
    out = df.copy()
    out["n_chars"] = s.str.len()
    out["n_words"] = s.str.split().str.len()
    out["n_digits"] = s.str.count(r"\d")
    out["n_upper"] = s.str.count(r"[A-Z]")
    out["upper_ratio"] = out["n_upper"] / out["n_chars"].clip(lower=1)
    out["n_exclam"] = s.str.count("!")
    out["has_url"] = s.str.contains(_URL_RE).astype(int)
    out["has_money"] = s.str.contains(_MONEY_RE).astype(int)
    out["has_phone"] = s.str.contains(_PHONE_RE).astype(int)
    return out


def split_data(df, test_size=0.15, val_size=0.15, random_state=RANDOM_STATE):
    """Séparation stratifiée train / validation / test (par défaut 70/15/15).

    La stratification conserve la proportion de spam (~13 %) dans chaque partie,
    ce qui est essentiel avec des classes déséquilibrées.
    """
    train_val, test = train_test_split(df, test_size=test_size,
                                       stratify=df["y"], random_state=random_state)
    rel_val = val_size / (1 - test_size)
    train, val = train_test_split(train_val, test_size=rel_val,
                                  stratify=train_val["y"], random_state=random_state)
    return train.reset_index(drop=True), val.reset_index(drop=True), test.reset_index(drop=True)
