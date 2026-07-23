# esve-dashboard
Analyse de rentabilité ESVE — Licence Data

## Lancement

```bash
pip install -r requirements.txt
streamlit run etape6_dashboard.py
```

Dans la barre latérale, le bouton **Charger une fiche de pointage** accepte les
fichiers `.csv`, `.xlsx` et `.xls`. Une fiche brute doit contenir au minimum les
colonnes `jour`, `heures_travaillees` et `revenu_fcfa`. Le dashboard calcule
alors automatiquement les coûts, la marge, le ROI, le score et la
classification. Les fiches déjà enrichies restent également acceptées. Sans
fichier chargé, le dashboard utilise `classification_ml_contrats.csv`.

Les données actives peuvent être téléchargées depuis le dashboard en formats
CSV, Excel `.xlsx` ou PDF. Le fichier chargé doit contenir les colonnes
utilisées par les indicateurs de rentabilité du dashboard ; en cas de colonne
manquante, la liste est affichée directement dans l’application.
