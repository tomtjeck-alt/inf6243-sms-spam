# Détection de spam SMS — INF6243 (UQO, automne 2026)

Étude réalisée dans le cadre du cours **INF6243 Techniques d'apprentissage** (Université du Québec en Outaouais) : classification de SMS en *spam* / *ham*.

- Analyse exploratoire, prétraitement du texte (TF-IDF) et caractéristiques dérivées.
- Modèles comparés : régression logistique, SVM, forêt aléatoire, BiLSTM, entre autres (voir `Code/models.py`).
- GridSearch, validation croisée k-fold, matrices de confusion, courbes ROC/PR, courbes d'apprentissage et importance des attributs.
- Figures dans `Code/figures/`, résultats chiffrés dans `Code/resultats/`.

## Exécution

```bash
pip install pandas scikit-learn matplotlib seaborn
jupyter notebook Code/notebook_principal.ipynb
```
