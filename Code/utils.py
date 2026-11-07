"""
utils.py
--------
Métriques et visualisations. Tous les graphiques ont un titre, des axes
étiquetés et une légende (exigence 3.4 de l'énoncé).
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (accuracy_score, auc, average_precision_score, confusion_matrix,
                             f1_score, precision_recall_curve, precision_score, recall_score,
                             roc_auc_score, roc_curve)
from sklearn.model_selection import learning_curve

sns.set_theme(style="whitegrid", context="notebook")
CLASS_NAMES = ["ham", "spam"]


def compute_metrics(y_true, y_pred, y_score=None):
    """Accuracy, précision / rappel / F1 (classe spam = positive), F1 macro, ROC-AUC, PR-AUC."""
    m = {
        "Accuracy": accuracy_score(y_true, y_pred),
        "Précision (spam)": precision_score(y_true, y_pred, zero_division=0),
        "Rappel (spam)": recall_score(y_true, y_pred),
        "F1 (spam)": f1_score(y_true, y_pred),
        "F1 macro": f1_score(y_true, y_pred, average="macro"),
    }
    if y_score is not None:
        m["ROC-AUC"] = roc_auc_score(y_true, y_score)
        m["PR-AUC"] = average_precision_score(y_true, y_score)
    return m


def plot_confusion(y_true, y_pred, title, ax=None, normalize=False):
    cm = confusion_matrix(y_true, y_pred)
    data = cm / cm.sum(axis=1, keepdims=True) if normalize else cm
    ax = ax or plt.subplots(figsize=(4, 3.5))[1]
    sns.heatmap(data, annot=cm if not normalize else True, fmt="d" if not normalize else ".2%",
                cmap="Blues", cbar=False, xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES, ax=ax)
    ax.set_title(title)
    ax.set_xlabel("Classe prédite")
    ax.set_ylabel("Classe réelle")
    return ax


def plot_model_comparison(results_df, metrics=("Accuracy", "Précision (spam)", "Rappel (spam)",
                                               "F1 (spam)"), title="Comparaison des modèles"):
    """Barplot groupé : une barre par métrique pour chaque modèle."""
    long = results_df[list(metrics)].reset_index().melt(id_vars=results_df.index.name or "index",
                                                         var_name="Métrique", value_name="Score")
    xcol = long.columns[0]
    fig, ax = plt.subplots(figsize=(11, 4.5))
    sns.barplot(data=long, x=xcol, y="Score", hue="Métrique", ax=ax)
    lo = max(0.0, long["Score"].min() - 0.05)
    ax.set_ylim(lo, 1.0)
    ax.set_title(title)
    ax.set_xlabel("Modèle")
    ax.set_ylabel("Score")
    ax.legend(title="Métrique", loc="lower right", ncol=len(metrics))
    for c in ax.containers:
        ax.bar_label(c, fmt="%.3f", fontsize=7, padding=1)
    plt.tight_layout()
    return fig


def plot_cv_boxplot(cv_scores, metric="F1 (spam)"):
    """Distribution des scores sur les k plis de la validation croisée."""
    df = pd.DataFrame(cv_scores)
    fig, ax = plt.subplots(figsize=(9, 4))
    sns.boxplot(data=df, ax=ax, color="lightsteelblue")
    sns.stripplot(data=df, ax=ax, color="black", size=4, label="pli")
    ax.set_title(f"Validation croisée 5 plis : {metric} par modèle")
    ax.set_xlabel("Modèle")
    ax.set_ylabel(metric)
    h, l = ax.get_legend_handles_labels()
    ax.legend(h[:1], ["score d'un pli"], loc="lower left")
    plt.tight_layout()
    return fig


def plot_learning_curve(estimator, X, y, title, cv, scoring="f1", ax=None):
    """Courbe d'apprentissage : score train vs validation selon la taille d'entraînement."""
    sizes, tr, va = learning_curve(estimator, X, y, cv=cv, scoring=scoring,
                                   train_sizes=np.linspace(0.1, 1.0, 8), n_jobs=-1)
    ax = ax or plt.subplots(figsize=(5, 3.8))[1]
    for scores, lab, col in [(tr, "Entraînement", "tab:blue"), (va, "Validation croisée", "tab:orange")]:
        mu, sd = scores.mean(1), scores.std(1)
        ax.plot(sizes, mu, "o-", color=col, label=lab)
        ax.fill_between(sizes, mu - sd, mu + sd, color=col, alpha=0.15)
    ax.set_title(title)
    ax.set_xlabel("Nombre d'exemples d'entraînement")
    ax.set_ylabel(f"Score ({scoring})")
    ax.legend(loc="lower right")
    return ax


def plot_dl_history(hist, title="BiLSTM : courbes d'apprentissage"):
    ep = np.arange(1, len(hist["train_loss"]) + 1)
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
    axes[0].plot(ep, hist["train_loss"], "o-", label="Entraînement")
    axes[0].plot(ep, hist["val_loss"], "o-", label="Validation")
    axes[0].axvline(hist["best_epoch"], ls="--", c="grey", label="meilleur epoch")
    axes[0].set(title="Perte (BCE pondérée)", xlabel="Epoch", ylabel="Perte")
    axes[1].plot(ep, hist["train_acc"], "o-", label="Entraînement")
    axes[1].plot(ep, hist["val_acc"], "o-", label="Validation")
    axes[1].set(title="Accuracy", xlabel="Epoch", ylabel="Accuracy")
    for a in axes:
        a.legend()
    fig.suptitle(title)
    plt.tight_layout()
    return fig


def plot_top_coefficients(feature_names, coefs, title, k=15):
    """Mots les plus associés au spam (coef > 0) et au ham (coef < 0)."""
    order = np.argsort(coefs)
    idx = np.r_[order[:k], order[-k:]]
    fig, ax = plt.subplots(figsize=(8, 7))
    colors = ["tab:green" if coefs[i] < 0 else "tab:red" for i in idx]
    ax.barh(np.array(feature_names)[idx], coefs[idx], color=colors)
    ax.set_title(title)
    ax.set_xlabel("Coefficient (→ spam, ← ham)")
    ax.set_ylabel("Terme")
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color="tab:red", label="indique spam"),
                       Patch(color="tab:green", label="indique ham")], loc="lower right")
    plt.tight_layout()
    return fig


def plot_importances(feature_names, importances, title, k=20):
    idx = np.argsort(importances)[-k:]
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.barh(np.array(feature_names)[idx], importances[idx], color="tab:purple",
            label="importance (réduction d'impureté)")
    ax.set_title(title)
    ax.set_xlabel("Importance moyenne")
    ax.set_ylabel("Terme")
    ax.legend(loc="lower right")
    plt.tight_layout()
    return fig


def plot_roc_pr(curves, title_suffix="(ensemble de test)"):
    """curves : {nom: (y_true, y_score)} -> courbes ROC et Précision-Rappel."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for name, (yt, ys) in curves.items():
        fpr, tpr, _ = roc_curve(yt, ys)
        axes[0].plot(fpr, tpr, label=f"{name} (AUC={auc(fpr, tpr):.3f})")
        p, r, _ = precision_recall_curve(yt, ys)
        axes[1].plot(r, p, label=f"{name} (AP={average_precision_score(yt, ys):.3f})")
    axes[0].plot([0, 1], [0, 1], "k--", lw=1, label="hasard")
    axes[0].set(title=f"Courbes ROC {title_suffix}", xlabel="Taux de faux positifs",
                ylabel="Taux de vrais positifs (rappel)")
    axes[1].set(title=f"Courbes Précision-Rappel {title_suffix}", xlabel="Rappel (spam)",
                ylabel="Précision (spam)")
    axes[0].legend(fontsize=8, loc="lower right")
    axes[1].legend(fontsize=8, loc="lower left")
    plt.tight_layout()
    return fig
