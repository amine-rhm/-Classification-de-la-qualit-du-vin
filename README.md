Projet . Machine Learning 1
🍷 Classification de la qualité du vin
Mohamed Amine Rahmouni


📌 Présentation
L'objectif est de prédire si un vin est « bon » (note ≥ 7) à partir de 11 mesures physico-chimiques, par classification supervisée. Les données sont le jeu Wine Quality de l'UCI (Cortez et al., 2009) : vins rouges et blancs du Vinho Verde, soit 6497 observations (5320 après suppression des doublons).

🧹 Préparation des données
Fusion des vins rouges et blancs, ajout de la variable type (encodée en type_white).
Suppression de 1177 doublons pour éviter la fuite de données entre entraînement et test.
Aucune valeur manquante ; valeurs extrêmes conservées (plausibles chimiquement).
Cible binaire : quality ≥ 7. Classes déséquilibrées : 19 % de bons vins.
Découpage 80/20 stratifié (4256 observations d'entraînement, 1064 de test) ; standardisation et sélection de variables placées dans un Pipeline pour éviter toute fuite.
Sélection de variables : filtre de Pearson (seuil 0,03, qui n'écarte que free sulfur dioxide) pour les modèles de base, et SelectKBest (k optimisé par validation croisée) pour les modèles optimisés.
🤖 Méthode
Les cinq modèles demandés sont comparés : régression logistique, k-NN, arbre de décision, Naive Bayes, SVM (noyau RBF). Deux modèles complémentaires non linéaires (Random Forest et Gradient Boosting) sont ajoutés pour vérifier si les modèles imposés atteignent un plafond. Hyperparamètres optimisés par Grid Search et Random Search avec validation croisée stratifiée à 5 plis, critère F1. Les trois meilleurs modèles sont ensuite comparés par validation croisée répétée (5×5).

📈 Résultats (jeu de test, modèles optimisés)
Modèle	Précision	Rappel	F1	AUC	F1 en validation croisée	Écart F1 train-test
Régression logistique	0,400	0,797	0,533	0,817	0,539 ± 0,020	0,007
k-NN	0,628	0,460	0,531	0,824	0,492 ± 0,023	0,469
Arbre de décision	0,366	0,733	0,488	0,784	0,515 ± 0,018	0,037
Naive Bayes	0,381	0,584	0,461	0,761	0,500 ± 0,011	0,037
SVM (RBF)	0,405	0,827	0,544	0,854	0,546 ± 0,011	0,041
Random Forest (complément)	0,561	0,634	0,595	0,872	0,562 ± 0,025	0,268
Gradient Boosting (complément)	0,476	0,777	0,590	0,867	0,551 ± 0,014	0,121
Modèle retenu : SVM (noyau RBF). Trois modèles ont des performances statistiquement indiscernables : SVM, Gradient Boosting et Random Forest (F1 en validation croisée répétée 5×5 : 0,546 ± 0,014, 0,550 ± 0,018 et 0,555 ± 0,026 ; aucun n'est meilleur que les autres dans plus de 68 % des plis). Nous les départageons selon le compromis biais/variance : le SVM a la variance la plus faible, le plus petit écart entre entraînement et test (0,04 contre 0,12 pour le boosting et 0,27 pour la forêt) et le meilleur rappel (0,83), pour un modèle moins complexe. L'avantage des modèles d'ensemble sur le jeu de test (F1 de 0,59 contre 0,54) n'est pas confirmé par la validation croisée : avec environ 200 bons vins dans le test, il est probablement en grande partie dû à la composition de cet échantillon.

🔍 Observations clés
L'accuracy est trompeuse : le k-NN atteint 0,846 mais ne retrouve que 46 % des bons vins.
Le k-NN surapprend fortement (F1 train 1,000 contre 0,531 en test), et le Random Forest aussi, dans une moindre mesure (0,863 contre 0,595).
Les modèles d'ensemble (Random Forest, Gradient Boosting) font un peu mieux sur le jeu de test, mais pas de façon significative en validation croisée répétée.
Les variables les plus liées à la qualité sont l'alcool, la densité, les chlorures et l'acidité volatile.
La standardisation est déterminante : F1 du SVM de 0,416 à 0,545, du k-NN de 0,281 à 0,477.
Le filtre de Pearson apporte peu : les meilleurs modèles gardent 11 à 12 variables avec SelectKBest.
⚠️ Obstacles rencontrés
Fort déséquilibre des classes, géré par stratification, pondération des classes et choix du F1.
Risque de fuite de données avec la standardisation hors validation croisée, corrigé en utilisant un Pipeline.
Compromis précision/rappel difficile : les modèles les plus sensibles (SVM, régression logistique) retrouvent environ 80 % des bons vins mais avec une précision d'environ 0,40, tandis que le k-NN est plus précis (0,63) mais manque la moitié des bons vins. Le signal contenu dans les variables physico-chimiques est faible.
▶️ Lancer le projet
git clone <URL-du-dépôt>
cd <nom-du-dépôt>
python -m venv venv
source venv/bin/activate        # Windows : venv\Scripts\activate
pip install -r requirements.txt
jupyter notebook projet_vin.ipynb
✅ Tests
python -m pytest -v tests/                          # tests unitaires
python -m pytest --nbmake projet_vin.ipynb          # exécute le notebook en entier
📁 Structure du dépôt
├── projet_vin.ipynb        # notebook principal
├── wine.py                 # fonctions réutilisables (préparation, split, pipeline)
├── tests/test_wine.py      # tests unitaires
├── winequality-red.csv     # données (UCI)
├── winequality-white.csv
├── requirements.txt
├── pytest.ini
└── README.md
📚 Outils
Python 3.9, pandas, NumPy, scikit-learn, SciPy, Matplotlib, Seaborn, pytest, nbmake. Données : UCI Machine Learning Repository.

Mini-glossaire
Les bases
Terme	Explication simple
Variable explicative (feature)	Une mesure qui sert à prédire (ex. le taux d'alcool).
Cible (target)	Ce qu'on veut prédire : ici « bon vin » (1) ou non (0).
Jeu d'entraînement	Les exemples avec lesquels le modèle apprend (80 %).
Jeu de test	Des exemples mis de côté, utilisés une seule fois à la fin pour vérifier que le modèle sait généraliser (20 %).
Surapprentissage	Le modèle apprend les exemples « par cœur » (y compris leur bruit) et se trompe sur de nouveaux vins.
Fuite de données	Une information du jeu de test « s'infiltre » dans l'entraînement, ce qui rend les scores trop beaux pour être vrais.
Classes déséquilibrées	Une catégorie est beaucoup plus rare que l'autre (19 % de bons vins contre 81 %).
Valeur aberrante (outlier)	Une mesure très éloignée des autres (ex. un vin à 65 g/L de sucre). Ici, elles sont conservées car plausibles.
Corrélation de Pearson	Un nombre entre −1 et 1 qui mesure le lien linéaire entre deux variables. Elle ne détecte pas les relations non linéaires.
Préparation et réglage
Terme	Explication simple
Standardisation	Mettre toutes les variables à la même échelle (moyenne 0, écart-type 1).
Pipeline	Une chaîne d'étapes (standardiser, sélectionner, prédire) appliquée de façon identique et sans fuite.
Sélection de variables	Ne garder que les variables utiles pour réduire le bruit. Ici : un filtre sur la corrélation, puis SelectKBest (garde les k meilleures variables).
Paramètre / hyperparamètre	Un paramètre est appris par le modèle (ex. un coefficient). Un hyperparamètre est un réglage choisi avant l'apprentissage (ex. le nombre de voisins 
).
Validation croisée	On découpe l'entraînement en 5 parties, on s'entraîne sur 4 et on teste sur la 5e, en tournant. Stratifiée : chaque partie garde la même proportion de bons vins.
Grid Search	On teste toutes les combinaisons d'hyperparamètres d'une grille. Exhaustif mais coûteux.
Random Search	On teste des combinaisons tirées au hasard. Plus rapide, et souvent presque aussi bon.
Pondération des classes (class_weight)	Une erreur sur un bon vin (rare) compte plus qu'une erreur sur un vin ordinaire, pour compenser le déséquilibre.
Modèle d'ensemble	Un modèle qui combine plusieurs arbres (Random Forest, Gradient Boosting), souvent plus puissant mais plus complexe.
Évaluation
Terme	Explication simple
Accuracy	Part de prédictions correctes. Trompeuse ici : dire « jamais bon » donnerait déjà 81 %.
Précision	Parmi les vins annoncés « bons », la part qui l'est vraiment.
Rappel	Parmi les vrais bons vins, la part que le modèle retrouve.
F1	Moyenne harmonique de la précision et du rappel : elle est basse dès que l'un des deux l'est. C'est notre critère principal.
Faux positif / faux négatif	Un vin ordinaire annoncé « bon » / un bon vin raté.
Matrice de confusion	Un tableau qui compte les vrais et faux positifs et négatifs.
Courbe ROC et AUC	La courbe compare les bons vins retrouvés aux faux positifs, selon le seuil. L'AUC la résume : 0,5 = hasard, 1 = parfait.
Seuil de décision	Le niveau de confiance à partir duquel le modèle dit « bon vin » (0,5 par défaut). Le baisser augmente le rappel mais réduit la précision.
Biais / variance	Un modèle à biais élevé est trop simple et rate la structure des données. Un modèle à variance élevée est trop souple et colle au bruit (surapprentissage). On cherche un compromis.
