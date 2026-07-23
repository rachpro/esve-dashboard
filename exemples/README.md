# Fichiers d'essai

Chargez ces fichiers depuis la barre laterale du dashboard avec **Charger une fiche de pointage**.

- `fiche_pointage_standard.csv` : activite mixte avec journee courte, normale et longue.
- `fiche_pointage_faible_activite.csv` : heures faibles, manque a gagner important et recommandations d'augmentation.
- `fiche_pointage_forte_activite.csv` : machine utilisee 10 h ou plus par jour.
- `fiche_pointage_incomplete.csv` : fichier volontairement invalide pour tester le message de colonnes manquantes.

Chaque fiche valide est une fiche brute. Le dashboard calcule automatiquement les couts, la marge, le ROI, le score et la classification.

Pour tester le ROI, chargez aussi `classification_ml_contrats.csv` a la racine du projet, puis utilisez le selecteur **Base du ROI** :

- **Cout machine** : ROI d'environ 410 %.
- **Couts totaux** : ROI d'environ 216 %, carburant inclus.
