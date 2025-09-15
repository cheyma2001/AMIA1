import json
import os
import re
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split, RandomizedSearchCV, StratifiedKFold
from sklearn.metrics import accuracy_score, classification_report, roc_curve, auc
from xgboost import XGBClassifier , plot_tree

# Imports pour matrices/graphs =====
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import precision_recall_curve, average_precision_score


# --- Dossiers de sortie
os.makedirs('img', exist_ok=True)

# --- Chargement des données
df = pd.read_excel('srcData/rr.xlsx')

# === Regex précompilées (cohérence + perf)
RE_NUMERIC = re.compile(r'NUMBER|INTEGER|DECIMAL|FLOAT|NUMERIC', re.I)
RE_TEXT    = re.compile(r'CHAR|VARCHAR|VARCHAR2|TEXT|STRING', re.I)
RE_DATE    = re.compile(r'DATE|TIMESTAMP|TIME', re.I)
RE_KEYS    = re.compile(r'ID|CODE|NUM|KEY|REF|LIBL', re.I)
RE_MEAS    = re.compile(r'MONTANT|AMOUNT|VALEUR|VALUE|PRICE|PRIX|QTE|QTY|COUNT|TOTAL|SUM|MT_|SOLD_', re.I)
RE_CODE_PREFIX = re.compile(r'^CODE_|^TYPE_', re.I)
RE_FACT_PREFIX = re.compile(r'^FACT|MESURE|FLUX|TRANSACTION|EVNM|MVT_|ECRT_|BALN_', re.I)

# === GroupBy au niveau table
grouped = df.groupby('LIBELLE_DU_SEGMENT').agg({
    'NOM_EPURE_DE_LA_RUBRIQUE': ['count', lambda x: ' '.join(map(str, x))],
    'TYPE_DONNEES_COLONNE': lambda x: (
        sum(1 for t in x if RE_NUMERIC.search(str(t))),
        sum(1 for t in x if RE_TEXT.search(str(t))),
        sum(1 for t in x if RE_DATE.search(str(t)))
    ),
    'PK': lambda x: sum(1 for p in x if str(p).upper() == 'O'),
    'Table_Type': 'first'
}).reset_index()

# Renommage colonnes
grouped.columns = [
    'Table_Name', 'Num_Columns', 'Column_Names', 'Type_Counts', 'Num_PKs', 'Table_Type'
]

# Extraction des comptes par type
grouped['Num_Numeric'] = grouped['Type_Counts'].apply(lambda x: x[0])
grouped['Num_Text']    = grouped['Type_Counts'].apply(lambda x: x[1])
grouped['Num_Date']    = grouped['Type_Counts'].apply(lambda x: x[2])

# Sécuriser compteurs & ratios
grouped['Num_Columns'] = grouped['Num_Columns'].fillna(0).replace(0, 1)
grouped['Num_Numeric'] = grouped['Num_Numeric'].fillna(0)
grouped['Num_Text']    = grouped['Num_Text'].fillna(0)
grouped['Num_Date']    = grouped['Num_Date'].fillna(0)
grouped['Num_PKs']     = grouped['Num_PKs'].fillna(0).astype(int)

grouped['Pct_Numeric'] = grouped['Num_Numeric'] / grouped['Num_Columns']
grouped['Pct_Text']    = grouped['Num_Text']    / grouped['Num_Columns']
grouped['Pct_Date']    = grouped['Num_Date']    / grouped['Num_Columns']
grouped['PK_Ratio']    = grouped['Num_PKs']     / grouped['Num_Columns']

# Flags nom de table
grouped['Has_Code_Prefix'] = grouped['Table_Name'].apply(
    lambda x: 1 if RE_CODE_PREFIX.search(str(x)) else 0
)
grouped['Has_Measure_Prefix'] = grouped['Table_Name'].apply(
    lambda x: 1 if RE_FACT_PREFIX.search(str(x)) else 0
)

# Pré-agrège les colonnes par table
cols_by_table = (
    df.groupby('LIBELLE_DU_SEGMENT')['NOM_EPURE_DE_LA_RUBRIQUE']
      .apply(lambda s: [str(v).upper() for v in s.fillna('')])
      .to_dict()
)

def count_key_like_columns_fast(table_name: str) -> int:
    cols = cols_by_table.get(table_name, [])
    return sum(1 for c in cols if RE_KEYS.search(c))

def count_measure_like_columns_fast(table_name: str) -> int:
    cols = cols_by_table.get(table_name, [])
    return sum(1 for c in cols if RE_MEAS.search(c))

grouped['Num_Key_Like']     = grouped['Table_Name'].apply(count_key_like_columns_fast)
grouped['Num_Measure_Like'] = grouped['Table_Name'].apply(count_measure_like_columns_fast)

# Nettoyage colonnes temporaires
grouped = grouped.drop(columns=['Type_Counts', 'Column_Names'], errors='ignore')

# Encodage cible
grouped['Table_Type'] = grouped['Table_Type'].map({'FACT': 1, 'DIMENSION': 0})
grouped = grouped.dropna(subset=['Table_Type']).copy()
grouped['Table_Type'] = grouped['Table_Type'].astype(int)
grouped = grouped.fillna(0)

# Sauvegarde dataset + features d'entraînement
grouped.to_csv('dataset_ml.csv', index=False)
print("Dataset sauvegardé dans dataset_ml.csv")

X = grouped.drop(columns=['Table_Name', 'Table_Type'])
y = grouped['Table_Type']

training_feats = grouped[['Table_Name']].join(X)
# training_feats.to_csv('training_features_by_table.csv', index=False)
# print("Features d'entraînement sauvegardées dans training_features_by_table.csv")

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Gestion du déséquilibre
pos = int((y_train == 1).sum())
neg = int((y_train == 0).sum())
scale_pos_weight = (neg / pos) if pos > 0 else 1.0

# Modèle XGBoost
xgb_model = XGBClassifier(
    random_state=42,
    eval_metric='logloss',
    scale_pos_weight=scale_pos_weight
)

# Hyperparamètres pour RandomSearch
param_dist = {
    'n_estimators': [300, 500, 700],
    'max_depth': [2, 3, 4],
    'learning_rate': [0.01, 0.05, 0.1],
    'subsample': [0.6, 0.7, 0.8],
    'min_child_weight': [1, 3, 5],
    'gamma': [0, 0.05, 0.1]
}

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

random_search = RandomizedSearchCV(
    estimator=xgb_model,
    param_distributions=param_dist,
    n_iter=60,
    cv=cv,
    scoring='f1_weighted',
    n_jobs=-1,
    verbose=1,
    random_state=42
)

# Entraînement SANS early stopping pendant la recherche
random_search.fit(X_train, y_train)

print("Meilleurs hyperparamètres :", random_search.best_params_)

# ===== Réentraînement final =====
best_params = random_search.best_params_
best_model = XGBClassifier(
    **best_params,
    random_state=42,
    eval_metric='logloss',
    scale_pos_weight=scale_pos_weight
)

try:
    best_model.fit(X_train, y_train, eval_set=[(X_test, y_test)], early_stopping_rounds=50, verbose=False)
except TypeError:
    best_model.fit(X_train, y_train)

# Évaluation
y_pred = best_model.predict(X_test)
print("Précision :", accuracy_score(y_test, y_pred))
print("\nRapport de classification :\n", classification_report(y_test, y_pred, digits=4))

y_prob = best_model.predict_proba(X_test)[:, 1]
fpr, tpr, _ = roc_curve(y_test, y_prob)
roc_auc = auc(fpr, tpr)
print(f"AUC: {roc_auc:.2f}")

# ===== [AJOUT] ROC combinée Train vs Test =====
# Probabilités sur le train
y_prob_train = best_model.predict_proba(X_train)[:, 1]
fpr_tr, tpr_tr, _ = roc_curve(y_train, y_prob_train)
roc_auc_tr = auc(fpr_tr, tpr_tr)

# Graphique unique avec Train & Test
fig_roc, ax_roc = plt.subplots(figsize=(6, 5), constrained_layout=True)
ax_roc.plot(fpr_tr, tpr_tr, label=f'Train (AUC = {roc_auc_tr:.2f})')
ax_roc.plot(fpr, tpr, label=f'Test (AUC = {roc_auc:.2f})')
ax_roc.plot([0, 1], [0, 1], linestyle='--', label='Classif. aléatoire')
ax_roc.set_xlabel('Taux de faux positifs (FPR)')
ax_roc.set_ylabel('Taux de vrais positifs (TPR)')
ax_roc.set_title('Courbe ROC – Train vs Test', pad=12)
ax_roc.legend(loc='lower right')
ax_roc.grid(alpha=0.3)
plt.savefig('img/roc_curve_train_test.png', dpi=160)
plt.close(fig_roc)

print("Courbe ROC train/test sauvegardée : img/roc_curve_train_test.png")
# ===== [FIN AJOUT] =====

# ===== [AJOUT] Courbe Précision–Rappel (sur Test) =====
prec, rec, _ = precision_recall_curve(y_test, y_prob)
ap = average_precision_score(y_test, y_prob)

fig_pr, ax_pr = plt.subplots(figsize=(6, 5), constrained_layout=True)
ax_pr.plot(rec, prec, label=f'PR (AP = {ap:.2f})')
ax_pr.set_xlabel('Recall')
ax_pr.set_ylabel('Precision')
ax_pr.set_title('Courbe Précision–Rappel – FACT vs DIMENSION', pad=12)
ax_pr.legend(loc='lower left')
ax_pr.grid(alpha=0.3)
plt.savefig('img/precision_recall_curve.png', dpi=160)
plt.close(fig_pr)

print("Courbe PR sauvegardée : img/precision_recall_curve.png")
# ===== [FIN AJOUT] =====

# ===== [AJOUT] Matrices de confusion (brute + normalisée) =====
labels = [0, 1]
display_labels = ['DIMENSION (0)', 'FACT (1)']

# Brute
cm = confusion_matrix(y_test, y_pred, labels=labels)
fig, ax = plt.subplots(figsize=(5, 5))
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=display_labels)
disp.plot(ax=ax, cmap='Blues', values_format='d', colorbar=False)
ax.set_title('Matrice de confusion – XGBClassifier')
plt.tight_layout()
plt.savefig('img/confusion_matrix.png', dpi=160)
plt.close(fig)

# Normalisée (par ligne)
cm_norm = confusion_matrix(y_test, y_pred, labels=labels, normalize='true')
fig2, ax2 = plt.subplots(figsize=(5, 5))
disp_norm = ConfusionMatrixDisplay(confusion_matrix=cm_norm, display_labels=display_labels)
disp_norm.plot(ax=ax2, cmap='Blues', values_format='.2f', colorbar=True)
ax2.set_title('Matrice de confusion normalisée – XGBClassifier')
plt.tight_layout()
plt.savefig('img/confusion_matrix_normalized.png', dpi=160)
plt.close(fig2)

print("Matrices de confusion sauvegardées :")
print(" - img/confusion_matrix.png")
print(" - img/confusion_matrix_normalized.png")


#ajout de l arbre 
# plt.figure(figsize=(20,10))
# plot_tree(best_model,tree_idx=0,rankdir='LR')
# plt.savefig('img/arbre.png', dpi=160)
# # Sauvegarde modèle + features
best_model.save_model('best_xgb_model.json')
with open('best_xgb_features.json', 'w', encoding='utf-8') as f:
    json.dump(list(X.columns), f, ensure_ascii=False, indent=2)

print("Modèle sauvegardé dans best_xgb_model.json")
print("Features sauvegardées dans best_xgb_features.json")