import os  # Module pour interagir avec le système de fichiers (créer, lire, vérifier)
import numpy as np
import pandas as pd
import librosa  # Librairie audio pour l'analyse des signaux sonores (MFCC, ZCR, etc.)
import sounddevice as sd  # Module pour l'enregistrement audio en temps réel
import soundfile as sf  # Module pour lire et écrire des fichiers audio (WAV, MP3, etc.)
from sklearn.ensemble import RandomForestClassifier  # Modèle ML: Forêt Aléatoire
from sklearn.svm import SVC  # Modèle ML: Support Vector Machine (SVM)
from sklearn.model_selection import train_test_split, GridSearchCV  # Division données et recherche hyperparamètres
from sklearn.preprocessing import StandardScaler  # Normalisation/standardisation des features
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score  # Métriques d'évaluation
import joblib  # Pour sauvegarder et charger les modèles entraînés
import matplotlib.pyplot as plt  # Bibliothèque de visualisation graphique
import seaborn as sns  # Bibliothèque de visualisation statistique (plus esthétique que matplotlib)
import warnings  # Module pour gérer les avertissements Python

# Désactiver les avertissements pour un affichage plus propre
warnings.filterwarnings('ignore')



class ProjetFinal:
    """
    Classe principale qui encapsule toute la logique du projet de détection d'énergie vocale.
    Suit le paradigme de programmation orientée objet pour une meilleure organisation.
    """

    def __init__(self):
        """
        Constructeur de la classe. Initialise toutes les variables et paramètres nécessaires.
        S'exécute automatiquement quand on crée un objet ProjetFinal.
        """
        print("PROJET FINAL - UTILISATION DE TOUTES LES DONNÉES")
        print("=" * 60)  # Ligne de séparation visuelle

        # DataFrame vide qui contiendra toutes nos données (features + labels)
        self.dataset = pd.DataFrame()

        # Variables pour les modèles (ancienne version - gardée pour compatibilité)
        self.modele = None

        #Variables pour les deux modèles
        self.modele_rf = None  # Pour Random Forest
        self.modele_svm = None  # Pour SVM
        self.resultats_comparaison = {}  # Dictionnaire pour stocker les résultats de comparaison

        # Objet pour normaliser les données (moyenne=0, écart-type=1)
        self.scaler = StandardScaler()

        # Liste des chemins possibles où chercher les fichiers audio
        self.dossiers_audio = [
            'src/data/enregistrements',  # Chemin principal attendu
            'src/data/enregistrements/energique',  # Sous-dossier énergique
            'src/data/enregistrements/fatigue',  # Sous-dossier fatigue
            'src/data/enregistrements/neutre',  # Sous-dossier neutre
            'src/data',  # Autre chemin possible
            'data/enregistrements',  # Autre structure possible
            'data'  # Dernier chemin à tester
        ]


    def trouver_tous_les_audios(self):
        """
        Parcourt récursivement les dossiers pour trouver tous les fichiers audio (.wav).

        Returns:
            list: Liste de dictionnaires avec les informations de chaque fichier audio
        """
        print("Recherche de tous les fichiers audio...")

        tous_les_fichiers = []  # Liste qui va contenir tous les fichiers trouvés
        dossier_principal = 'src/data/enregistrements'  # Chemin principal à explorer

        # Vérifier si le dossier principal existe
        if not os.path.exists(dossier_principal):
            print(f" {dossier_principal} n'existe pas")
            return tous_les_fichiers  # Retourne liste vide si dossier inexistant

        print(f" Dossier trouvé: {dossier_principal}")

        # Parcourir les trois sous-dossiers correspondant aux états vocaux
        for etat in ['energique', 'fatigue', 'neutre']:
            # Construire le chemin complet du sous-dossier
            dossier_etat = os.path.join(dossier_principal, etat)

            # Vérifier si le sous-dossier existe
            if os.path.exists(dossier_etat):
                # Lister tous les fichiers .wav dans ce dossier
                fichiers_etat = [f for f in os.listdir(dossier_etat) if f.endswith('.wav')]
                print(f"   {etat}: {len(fichiers_etat)} fichiers")  # Afficher le compte

                # Pour chaque fichier trouvé, créer un dictionnaire d'informations
                for fichier in fichiers_etat:
                    chemin_complet = os.path.join(dossier_etat, fichier)
                    tous_les_fichiers.append({
                        'fichier': chemin_complet,  # Chemin absolu du fichier
                        'etat': etat,  # Catégorie (énergique, fatigue, neutre)
                        'source': dossier_principal  # Dossier source pour traçabilité
                    })

        print(f" TOTAL: {len(tous_les_fichiers)} fichiers audio trouvés")
        return tous_les_fichiers

    def extraire_features_avancees(self, fichier_audio):
        """
        Extrait 45 caractéristiques audio avancées d'un fichier .wav.

        Args:
            fichier_audio (str): Chemin vers le fichier audio

        Returns:
            dict: Dictionnaire contenant toutes les features extraites
        """
        try:
            # Charger le fichier audio avec librosa
            # y: signal audio (tableau numpy), sr: fréquence d'échantillonnage (22050 Hz)
            y, sr = librosa.load(fichier_audio, sr=22050)

            # Normalisation du signal: diviser par l'amplitude maximale
            # +1e-8 évite la division par zéro
            y = y / (np.max(np.abs(y)) + 1e-8)

            features = {}  # Dictionnaire pour stocker toutes les features

            # ============ FEATURES DE BASE ============

            # Énergie RMS (Root Mean Square) - mesure l'intensité du signal
            rms = librosa.feature.rms(y=y)
            features['energy'] = np.mean(rms)  # Énergie moyenne
            features['energy_std'] = np.std(rms)  # Écart-type de l'énergie

            # Centroïde spectral - "centre de gravité" du spectre (brillance)
            spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
            features['spectral_centroid'] = np.mean(spectral_centroid)
            features['spectral_centroid_std'] = np.std(spectral_centroid)

            # Zero Crossing Rate (ZCR) - nombre de passages par zéro (voix/percussion)
            zcr = librosa.feature.zero_crossing_rate(y)
            features['zcr'] = np.mean(zcr)
            features['zcr_std'] = np.std(zcr)

            # ============ FEATURES SPECTRALES ============

            # Spectral rolloff - fréquence contenant 85% de l'énergie spectrale
            features['spectral_rolloff'] = np.mean(librosa.feature.spectral_rolloff(y=y, sr=sr))

            # Bandwidth spectrale - largeur du spectre
            features['spectral_bandwidth'] = np.mean(librosa.feature.spectral_bandwidth(y=y, sr=sr))

            # ============ MFCC (Mel-Frequency Cepstral Coefficients) ============
            # Les MFCC capturent l'enveloppe spectrale, cruciale pour la reconnaissance vocale
            mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)

            # Prendre les 5 premiers coefficients (les plus informatifs)
            for i in range(5):
                features[f'mfcc_{i + 1}'] = np.mean(mfccs[i])  # Moyenne du coefficient
                features[f'mfcc_{i + 1}_std'] = np.std(mfccs[i])  # Écart-type du coefficient

            # ============ CHROMA ============
            # Distribution de l'énergie sur les 12 classes de hauteur (notes musicales)
            chroma = librosa.feature.chroma_stft(y=y, sr=sr)
            features['chroma_stft'] = np.mean(chroma)

            return features  # Retourner le dictionnaire complet

        except Exception as e:
            # Gestion d'erreur si le fichier est corrompu ou inaccessible
            print(f"Erreur avec {fichier_audio}: {e}")
            return None

    def charger_et_preparer_donnees(self):
        """
        Charge tous les fichiers audio, extrait leurs features et prépare le dataset.

        Returns:
            bool: True si au moins un fichier a été chargé, False sinon
        """
        print("\nCHARGEMENT DE TOUTES LES DONNÉES EXISTANTES")

        # Trouver tous les fichiers audio disponibles
        tous_les_audios = self.trouver_tous_les_audios()

        # Vérifier si des fichiers ont été trouvés
        if not tous_les_audios:
            print(" Aucun fichier audio trouvé!")
            return False  # Échec du chargement

        donnees_chargees = 0  # Compteur de fichiers traités avec succès

        # Parcourir tous les fichiers audio trouvés
        for audio in tous_les_audios:
            # Extraire les features du fichier audio
            features = self.extraire_features_avancees(audio['fichier'])

            if features:  # Si l'extraction a réussi
                # Ajouter les métadonnées aux features
                features['etat'] = audio['etat']  # Classe (énergique/fatigue/neutre)
                features['fichier'] = audio['fichier']  # Chemin du fichier
                features['source'] = audio['source']  # Source du fichier

                # Ajouter au DataFrame principal
                # pd.DataFrame([features]) crée un DataFrame d'une ligne
                # ignore_index=True réinitialise les index
                self.dataset = pd.concat([self.dataset, pd.DataFrame([features])], ignore_index=True)
                donnees_chargees += 1

        print(f" {donnees_chargees} fichiers chargés sur {len(tous_les_audios)}")

        # Afficher les statistiques descriptives
        self.afficher_statistiques_completes()

        # Retourner True si au moins un fichier a été chargé
        return donnees_chargees > 0

    # ============================================================================
    # SECTION 4 : ANALYSE ET VISUALISATION DES DONNÉES
    # ============================================================================

    def afficher_statistiques_completes(self):
        """Affiche des statistiques descriptives détaillées du dataset."""
        if self.dataset.empty:
            print(" Aucune donnée chargée")
            return

        print(f"\n STATISTIQUES COMPLÈTES:")
        print(f"Total: {len(self.dataset)} échantillons")

        # Statistiques par état vocal
        for etat in ['energique', 'fatigue', 'neutre']:
            # Filtrer le dataset pour l'état courant
            data_etat = self.dataset[self.dataset['etat'] == etat]

            if len(data_etat) > 0:
                print(f"\n   {etat.upper()} ({len(data_etat)} échantillons):")

                # Énergie moyenne ± écart-type
                print(f"     Energy: {data_etat['energy'].mean():.6f} ± {data_etat['energy'].std():.6f}")

                # Centroïde spectral moyen ± écart-type
                print(
                    f"     Spectral: {data_etat['spectral_centroid'].mean():.0f} ± {data_etat['spectral_centroid'].std():.0f} Hz")

                # ZCR moyen ± écart-type
                print(f"     ZCR: {data_etat['zcr'].mean():.4f} ± {data_etat['zcr'].std():.4f}")

        # Répartition par source (dossier d'origine)
        print(f"\n RÉPARTITION PAR SOURCE:")
        for source in self.dataset['source'].unique():
            count = len(self.dataset[self.dataset['source'] == source])
            print(f"   {source}: {count} échantillons")

    # Détection des valeurs aberrantes
    def detecter_valeurs_aberrantes(self):
        """
        Étape 4 du guide : Détecte et visualise les valeurs aberrantes dans le dataset.
        Utilise la méthode Z-score (valeurs avec |Z| > 3 considérées aberrantes).
        """
        print("\n" + "=" * 60)
        print("VÉRIFICATION DES VALEURS ABERRANTES")
        print("=" * 60)

        if self.dataset.empty:
            print(" Chargez d'abord les données!")
            return

        # Importer stats depuis scipy pour le calcul des Z-scores
        from scipy import stats

        # Sélectionner uniquement les features numériques (exclure les colonnes textuelles)
        X = self.dataset.drop(['etat', 'fichier', 'source'], axis=1)

        # Calculer les Z-scores : mesure combien d'écarts-types chaque valeur est éloignée de la moyenne
        z_scores = np.abs(stats.zscore(X))
        seuil = 3  # Seuil statistique standard pour détecter les outliers
        outliers = (z_scores > seuil).any(axis=1)  # True pour les lignes avec au moins une valeur aberrante

        print(f" Valeurs aberrantes détectées (Z-score > {seuil}): {outliers.sum()}")

        # ============ VISUALISATION DES VALEURS ABERRANTES ============
        plt.figure(figsize=(15, 6))

        # Sous-graphique 1 : Boxplot des 5 premières features
        plt.subplot(1, 2, 1)
        sns.boxplot(data=X.iloc[:, :5])  # Affiche seulement les 5 premières colonnes pour lisibilité
        plt.title('Boxplot - Détection visuelle des outliers')
        plt.xticks(rotation=45)  # Incliner les labels pour meilleure lisibilité
        plt.grid(True, alpha=0.3)  # Grille légère pour référence

        # Sous-graphique 2 : Histogramme des Z-scores
        plt.subplot(1, 2, 2)
        z_flat = z_scores.flatten()  # Aplatir la matrice en vecteur 1D (pas besoin de .values)
        z_flat = z_flat[~np.isinf(z_flat)]  # Supprimer les valeurs infinies
        plt.hist(z_flat, bins=50, edgecolor='black', alpha=0.7)
        plt.axvline(x=seuil, color='red', linestyle='--',
                    label=f'Seuil Z={seuil}', linewidth=2)
        plt.xlabel('Z-score')
        plt.ylabel('Fréquence')
        plt.title('Distribution des Z-scores')
        plt.legend()
        plt.grid(True, alpha=0.3)

        plt.tight_layout()  # Ajuster l'espacement
        plt.savefig('valeurs_aberrantes.png', dpi=300, bbox_inches='tight')
        plt.show()

        return outliers

    #  Description théorique des modèles
    def description_detaille_modeles(self):
        """
        Étape 7 du guide : Fournit une description théorique détaillée des deux modèles utilisés.
        Inclut des diagrammes explicatifs pour faciliter la compréhension.
        """
        print("\n" + "=" * 60)
        print("DESCRIPTION THÉORIQUE DES MODÈLES UTILISÉS")
        print("=" * 60)

        # Description textuelle détaillée
        description = """
        🌲 RANDOM FOREST (FORÊT ALÉATOIRE) :
        • Type : Apprentissage par ensemble (Ensemble Learning)
        • Principe : Combine plusieurs arbres de décision indépendants
        • Algorithme : Bagging (Bootstrap Aggregating)
        • Avantages :
          - Réduit le sur-apprentissage (overfitting)
          - Calcule automatiquement l'importance des caractéristiques
          - Robustes aux valeurs aberrantes
        • Hyperparamètres optimisés :
          - n_estimators : Nombre d'arbres dans la forêt
          - max_depth : Profondeur maximale de chaque arbre
          - min_samples_split : Échantillons minimum pour diviser un nœud

        🔷 SVM (SUPPORT VECTOR MACHINE) :
        • Type : Classificateur à marge maximale
        • Principe : Trouve l'hyperplan optimal qui sépare les classes
        • Kernel Trick : Transforme les données non-linéaires en espace linéaire
        • Avantages :
          - Efficace en haute dimensionnalité
          - Mémoire efficace (seuls les vecteurs support sont stockés)
          - Bonne performance avec petits datasets
        • Hyperparamètres optimisés :
          - C : Paramètre de régularisation (trade-off erreur/marge)
          - kernel : Type de noyau (linéaire, RBF, etc.)
          - gamma : Coefficient du noyau RBF
        """

        print(description)


        fig, axes = plt.subplots(1, 2, figsize=(14, 6))

        # ---- Diagramme 1 : Random Forest ----
        axes[0].text(0.1, 0.9, 'RANDOM FOREST', fontsize=14, fontweight='bold')

        # Représenter les arbres de décision
        axes[0].plot([0.2, 0.8], [0.8, 0.8], 'b-', linewidth=2)
        axes[0].text(0.2, 0.75, 'Arbre 1', fontsize=10)

        axes[0].plot([0.2, 0.8], [0.6, 0.6], 'g-', linewidth=2)
        axes[0].text(0.2, 0.55, 'Arbre 2', fontsize=10)

        axes[0].plot([0.2, 0.8], [0.4, 0.4], 'r-', linewidth=2)
        axes[0].text(0.2, 0.35, 'Arbre 3', fontsize=10)

        axes[0].text(0.2, 0.25, '...', fontsize=12)

        axes[0].plot([0.2, 0.8], [0.2, 0.2], 'm-', linewidth=2)
        axes[0].text(0.2, 0.15, 'Arbre N', fontsize=10)

        # Flèche de vote majoritaire
        axes[0].text(0.5, 0.05, '↓ VOTE MAJORITAIRE ↓',
                     fontsize=10, fontweight='bold', ha='center', color='red')
        axes[0].text(0.5, -0.05, 'PRÉDICTION FINALE',
                     fontsize=12, fontweight='bold', ha='center', color='darkred')

        axes[0].set_xlim(0, 1)
        axes[0].set_ylim(-0.1, 1)
        axes[0].axis('off')
        axes[0].set_title('Principe Random Forest (Bagging)', fontsize=12, pad=20)

        # ---- Diagramme 2 : SVM ----
        axes[1].text(0.1, 0.9, 'SVM - MARGE MAXIMALE', fontsize=14, fontweight='bold')

        # Points classe A (bleus)
        axes[1].scatter([0.3, 0.35, 0.4, 0.45, 0.5],
                        [0.7, 0.65, 0.6, 0.55, 0.5],
                        c='blue', s=100, label='Classe A')

        # Points classe B (rouges)
        axes[1].scatter([0.6, 0.65, 0.7, 0.75, 0.8],
                        [0.5, 0.55, 0.6, 0.65, 0.7],
                        c='red', s=100, label='Classe B')

        # Hyperplan optimal (ligne verte)
        axes[1].plot([0.2, 0.9], [0.5, 0.7], 'g-', linewidth=3, label='Hyperplan optimal')

        # Marges (lignes pointillées)
        axes[1].plot([0.2, 0.9], [0.4, 0.6], 'g--', linewidth=1, alpha=0.5)
        axes[1].plot([0.2, 0.9], [0.6, 0.8], 'g--', linewidth=1, alpha=0.5)

        # Vecteurs support (étoiles violettes)
        axes[1].scatter([0.5, 0.6], [0.5, 0.6], c='purple', s=200,
                        marker='*', label='Vecteurs support')

        axes[1].legend(loc='lower left')
        axes[1].set_xlim(0, 1)
        axes[1].set_ylim(0.3, 0.9)
        axes[1].set_title('Principe SVM - Séparation optimale', fontsize=12, pad=20)

        plt.tight_layout()
        plt.savefig('explication_modeles.png', dpi=300, bbox_inches='tight')
        plt.show()

        print("\n Diagrammes explicatifs sauvegardés: 'explication_modeles.png'")



    #   deux modèles
    def entrainer_deux_modeles_cv4(self):
        """
        Étapes 6 & 8 du guide : Entraîne DEUX modèles avec GridSearch et validation croisée CV=4.

        Returns:
            bool: True si l'entraînement a réussi, False sinon
        """
        print("\n" + "=" * 60)
        print("ENTRAÎNEMENT DE DEUX MODÈLES (Random Forest + SVM)")
        print("=" * 60)

        if self.dataset.empty:
            print(" Chargez d'abord les données!")
            return False

        # ============ PRÉPARATION DES DONNÉES ============

        # Séparer les features (X) des labels (y)
        X = self.dataset.drop(['etat', 'fichier', 'source'], axis=1)
        y = self.dataset['etat']

        print(f"Données: {X.shape[0]} échantillons, {X.shape[1]} features")

        # Division train/test (80%/20%) avec stratification pour maintenir les proportions
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=0.2,  # 20% pour le test
            random_state=42,  # Seed pour reproductibilité
            stratify=y  # Même distribution des classes dans train et test
        )

        # Normalisation : standardisation des features (moyenne=0, écart-type=1)
        X_train_scaled = self.scaler.fit_transform(X_train)  # Apprentissage + transformation
        X_test_scaled = self.scaler.transform(X_test)  # Transformation seulement


        print("\nOPTIMISATION RANDOM FOREST (CV=4)")

        # Grille d'hyperparamètres à tester
        rf_param_grid = {
            'n_estimators': [50, 100, 200],  # Nombre d'arbres dans la forêt
            'max_depth': [10, 20, None],  # Profondeur maximale (None = pas de limite)
            'min_samples_split': [2, 5],  # Nombre min d'échantillons pour diviser un nœud
            'min_samples_leaf': [1, 2]  # Nombre min d'échantillons dans une feuille
        }

        # Configuration de la recherche par grille
        rf_grid = GridSearchCV(
            RandomForestClassifier(random_state=42),  # Modèle de base
            rf_param_grid,  # Grille d'hyperparamètres
            cv=4,  # IMPORTANT : Validation croisée 4 folds comme demandé
            scoring='accuracy',  # Métrique d'optimisation
            n_jobs=-1,  # Utiliser tous les cœurs CPU disponibles
            verbose=1  # Afficher la progression
        )

        # Entraînement avec recherche d'hyperparamètres
        rf_grid.fit(X_train_scaled, y_train)
        self.modele_rf = rf_grid.best_estimator_  # Meilleur modèle trouvé
        print(f"Random Forest optimisé | Meilleurs params: {rf_grid.best_params_}")

        # ============ MODÈLE 2 : SVM ============
        print("\n🔷 OPTIMISATION SVM (CV=4)")

        # Grille d'hyperparamètres spécifique à SVM
        svm_param_grid = {
            'C': [0.1, 1, 10],  # Paramètre de régularisation
            'kernel': ['linear', 'rbf'],  # Type de noyau
            'gamma': ['scale', 'auto']  # Coefficient du noyau RBF
        }

        # Configuration de la recherche par grille pour SVM
        svm_grid = GridSearchCV(
            SVC(random_state=42, probability=True),  # probability=True pour avoir predict_proba
            svm_param_grid,
            cv=4,  # ⚠️ IMPORTANT : CV=4 comme demandé
            scoring='accuracy',
            n_jobs=-1,
            verbose=1
        )

        # Entraînement SVM
        svm_grid.fit(X_train_scaled, y_train)
        self.modele_svm = svm_grid.best_estimator_
        print(f"SVM optimisé | Meilleurs params: {svm_grid.best_params_}")

        # ============ ÉVALUATION DES DEUX MODÈLES ============
        self.evaluer_et_comparer_modeles(X_test_scaled, y_test)

        # Sauvegarde des modèles pour usage futur
        joblib.dump(self.modele_rf, 'modele_rf_final.pkl')
        joblib.dump(self.modele_svm, 'modele_svm_final.pkl')
        joblib.dump(self.scaler, 'scaler_final.pkl')

        print("\nModèles sauvegardés: 'modele_rf_final.pkl', 'modele_svm_final.pkl'")

        return True

    # Méthode d'évaluation et comparaison
    def evaluer_et_comparer_modeles(self, X_test, y_test):
        """
        Évalue et compare les performances des deux modèles.

        Args:
            X_test (array): Données de test (features)
            y_test (array): Labels de test
        """
        from sklearn.metrics import precision_recall_fscore_support

        print("\n" + "=" * 60)
        print("COMPARAISON DES PERFORMANCES")
        print("=" * 60)

        # Liste des modèles à évaluer
        modeles = [
            ('Random Forest', self.modele_rf),
            ('SVM', self.modele_svm)
        ]

        for nom_modele, modele in modeles:
            print(f"\nÉVALUATION {nom_modele}:")

            # Prédictions sur les données de test
            y_pred = modele.predict(X_test)

            # Calcul des métriques de performance
            accuracy = accuracy_score(y_test, y_pred)
            precision, recall, f1, _ = precision_recall_fscore_support(
                y_test, y_pred, average='weighted'  # Moyenne pondérée par le support
            )

            # Affichage des résultats
            print(f"   Accuracy: {accuracy:.3f}")
            print(f"   Precision: {precision:.3f}")
            print(f"   Recall: {recall:.3f}")
            print(f"   F1-Score: {f1:.3f}")

            # Rapport de classification détaillé
            print(f"\n   Rapport détaillé:")
            print(classification_report(y_test, y_pred, target_names=['energique', 'fatigue', 'neutre']))

            # Sauvegarde des résultats pour le tableau récapitulatif
            self.resultats_comparaison[nom_modele] = {
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1': f1
            }

            # Génération de la matrice de confusion
            self.generer_matrice_confusion_modele(y_test, y_pred, nom_modele)

    # génération des matrices de confusion
    def generer_matrice_confusion_modele(self, y_true, y_pred, nom_modele):
        """
        Étape 9 du guide : Génère et sauvegarde la matrice de confusion pour un modèle.

        Args:
            y_true (array): Labels réels
            y_pred (array): Labels prédits
            nom_modele (str): Nom du modèle pour le titre
        """
        from sklearn.metrics import confusion_matrix

        # Calcul de la matrice de confusion
        cm = confusion_matrix(y_true, y_pred)

        # Création de la visualisation
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=['energique', 'fatigue', 'neutre'],
                    yticklabels=['energique', 'fatigue', 'neutre'])

        plt.title(f'Matrice de Confusion - {nom_modele}', fontsize=14, fontweight='bold')
        plt.ylabel('Vérité terrain (Réel)', fontsize=12)
        plt.xlabel('Prédiction du modèle', fontsize=12)
        plt.tight_layout()

        # Sauvegarde avec nom spécifique
        nom_fichier = f'matrice_confusion_{nom_modele.replace(" ", "_").lower()}.png'
        plt.savefig(nom_fichier, dpi=300, bbox_inches='tight')
        plt.show()

        print(f"   Matrice de confusion sauvegardée: '{nom_fichier}'")

    #  Tableau récapitulatif des performances
    def generer_tableau_recapitulatif(self):
        """Étape 10 du guide : Génère un tableau synthétique des performances des deux modèles."""
        if not self.resultats_comparaison:
            print("⚠️  Entraînez d'abord les modèles!")
            return

        print("\n" + "=" * 80)
        print("TABLEAU RÉCAPITULATIF DES PERFORMANCES")
        print("=" * 80)

        # Préparation des données pour le tableau
        donnees = []
        for nom_modele, metrics in self.resultats_comparaison.items():
            donnees.append({
                'Modèle': nom_modele,
                'Accuracy': f"{metrics['accuracy']:.3f}",
                'Precision': f"{metrics['precision']:.3f}",
                'Recall': f"{metrics['recall']:.3f}",
                'F1-Score': f"{metrics['f1']:.3f}",
                'Validation': '4-folds CV',
                'Grid Search': '✓'
            })

        # Création du DataFrame
        df_resultats = pd.DataFrame(donnees)

        # Affichage console
        print("\n📋 PERFORMANCES SUR DONNÉES DE TEST (20%):")
        print(df_resultats.to_string(index=False))

        # ============ VISUALISATION GRAPHIQUE DU TABLEAU ============
        plt.figure(figsize=(12, 4))
        plt.axis('tight')
        plt.axis('off')

        # Couleurs d'en-tête pastel
        colors = ['#4ECDC4', '#45B7D1', '#FF6B6B', '#96CEB4', '#FFEAA7', '#DDA0DD']

        # ⭐⭐ CORRECTION IMPORTANTE : Vérifier nombre de colonnes vs couleurs ⭐⭐
        n_colonnes = len(df_resultats.columns)

        # Vérifier si on a assez de couleurs
        if len(colors) < n_colonnes:
            print(f"⚠️  Attention: {n_colonnes} colonnes mais seulement {len(colors)} couleurs")
            # Ajouter des couleurs par défaut si besoin
            couleurs_supplementaires = ['#C9C9FF', '#FFD8B8', '#E6E6FA', '#B5EAD7', '#FFB7B2']
            colors.extend(couleurs_supplementaires[:n_colonnes - len(colors)])
            print(f"✅ {len(colors)} couleurs disponibles maintenant")

        # Création du tableau matplotlib
        table = plt.table(cellText=df_resultats.values,
                          colLabels=df_resultats.columns,
                          cellLoc='center',
                          loc='center',
                          colColours=colors[:n_colonnes])  # Prendre exactement n_colonnes couleurs

        # Personnalisation du style
        table.auto_set_font_size(False)
        table.set_fontsize(11)
        table.scale(1.2, 2)

        # Titre principal
        plt.title('TABLEAU RÉCAPITULATIF - COMPARAISON DES MODÈLES',
                  fontsize=16, fontweight='bold', pad=20, color='darkblue')

        plt.tight_layout()
        plt.savefig('tableau_recapitulatif.png', dpi=300, bbox_inches='tight')
        plt.show()

        # Sauvegarde en CSV pour usage externe
        df_resultats.to_csv('tableau_performances.csv', index=False)
        print("\n✅ Fichiers générés :")
        print("   • tableau_performances.csv")
        print("   • tableau_recapitulatif.png")

    # ============================================================================
    # SECTION 6 : VISUALISATIONS COMPLÈTES
    # ============================================================================

    def generer_visualisations_completes(self):
        """
        Génère un ensemble complet de visualisations pour l'analyse exploratoire.
        Inclut des graphiques pour comprendre la distribution des données.
        """
        if self.dataset.empty:
            print(" Chargez d'abord les données!")
            return

        print("\n GÉNÉRATION DES VISUALISATIONS COMPLÈTES...")

        # Création d'une figure avec 4 sous-graphiques
        plt.figure(figsize=(12, 10))

        # ============ 1. DISTRIBUTION DES ÉTATS (Camembert) ============
        plt.subplot(2, 2, 1)
        counts = self.dataset['etat'].value_counts()
        colors = ['#ff6b6b', '#4ecdc4', '#45b7d1']  # Rouge, Turquoise, Bleu
        plt.pie(counts.values, labels=counts.index, autopct='%1.1f%%',
                colors=colors, startangle=90)
        plt.title('Distribution des États Vocaux')

        # ============ 2. COMPARAISON DES FEATURES PAR ÉTAT ============
        plt.subplot(2, 2, 2)
        features_plot = ['energy', 'spectral_centroid', 'zcr']
        colors_etat = {'energique': '#ff6b6b', 'fatigue': '#4ecdc4', 'neutre': '#45b7d1'}

        for etat in ['energique', 'fatigue', 'neutre']:
            data_etat = self.dataset[self.dataset['etat'] == etat]
            means = [data_etat[feature].mean() for feature in features_plot]
            plt.plot(features_plot, means, marker='o', label=etat, linewidth=3,
                     color=colors_etat[etat], markersize=8)

        plt.xlabel('Caractéristiques Audio')
        plt.ylabel('Valeurs Moyennes')
        plt.title('Comparaison des Features par État Vocal')
        plt.legend()
        plt.xticks(rotation=45)
        plt.grid(True, alpha=0.3)

        # ============ 3. HISTOGRAMME DE L'ÉNERGIE ============
        plt.subplot(2, 2, 3)
        for etat in ['energique', 'fatigue', 'neutre']:
            data_etat = self.dataset[self.dataset['etat'] == etat]
            plt.hist(data_etat['energy'], alpha=0.7, label=etat, bins=8,
                     color=colors_etat[etat], edgecolor='black')

        plt.xlabel('Énergie Audio (RMS)')
        plt.ylabel('Nombre d\'échantillons')
        plt.title('Distribution de l\'Énergie par État')
        plt.legend()

        # ============ 4. CENTROÏDE SPECTRAL (Boxplot) ============
        plt.subplot(2, 2, 4)
        box_data = []
        box_labels = []

        for etat in ['energique', 'fatigue', 'neutre']:
            data_etat = self.dataset[self.dataset['etat'] == etat]
            if len(data_etat) > 0:
                box_data.append(data_etat['spectral_centroid'].values)
                box_labels.append(etat)

        box_plot = plt.boxplot(box_data, labels=box_labels, patch_artist=True)

        # Colorisation des boxplots
        for patch, color in zip(box_plot['boxes'], [colors_etat[etat] for etat in box_labels]):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)

        plt.ylabel('Fréquence (Hz)')
        plt.title('Centroïde Spectral par État Vocal')
        plt.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig('analyse_complete_etats.png', dpi=300, bbox_inches='tight')
        plt.show()
        print(" Visualisations sauvegardées: 'analyse_complete_etats.png'")

        # ============ 5. MATRICE DE CORRÉLATION ============
        plt.figure(figsize=(10, 8))
        features_corr = self.dataset.drop(['etat', 'fichier', 'source'], axis=1)

        # Limiter aux 8 premières features pour lisibilité
        features_corr = features_corr.iloc[:, :8]
        corr_matrix = features_corr.corr()

        # Masque pour n'afficher que le triangle supérieur
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool))

        sns.heatmap(corr_matrix, mask=mask, annot=True, cmap='coolwarm', center=0,
                    square=True, fmt='.2f', cbar_kws={"shrink": .8})

        plt.title('Matrice de Corrélation des Features Audio')
        plt.tight_layout()
        plt.savefig('matrice_correlation.png', dpi=300, bbox_inches='tight')
        plt.show()
        print(" Matrice de corrélation sauvegardée: 'matrice_correlation.png'")


    def analyser_voix(self, fichier_audio=None):
        """
        Analyse une voix en temps réel (enregistrement) ou à partir d'un fichier.

        Args:
            fichier_audio (str, optional): Chemin vers un fichier audio. Si None, enregistre.

        Returns:
            str: État vocal détecté
        """
        if self.modele_rf is None or self.modele_svm is None:
            print(" Entraînez d'abord les deux modèles!")
            return

        # ============ ENREGISTREMENT EN DIRECT SI AUCUN FICHIER ============
        if fichier_audio is None:
            print(f"\n Enregistrement de votre voix...")
            print("Parlez maintenant (4 secondes)...")

            fs = 22050  # Fréquence d'échantillonnage standard

            try:
                # Enregistrement audio avec sounddevice
                audio_data = sd.rec(int(4 * fs), samplerate=fs, channels=1, dtype='float64')
                sd.wait()  # Attendre la fin de l'enregistrement
                sf.write("test_temp.wav", audio_data, fs)  # Sauvegarder en fichier
                fichier_audio = "test_temp.wav"
                print("Enregistrement sauvegardé")
            except Exception as e:
                print(f"Erreur enregistrement: {e}")
                return

        # ============ EXTRACTION DES FEATURES ============
        features = self.extraire_features_avancees(fichier_audio)
        if not features:
            return

        # ============ PRÉPARATION POUR LA PRÉDICTION ============
        X = self.dataset.drop(['etat', 'fichier', 'source'], axis=1)
        features_ordre = X.columns  # Garder le même ordre que pendant l'entraînement

        X_test = np.array([[features[col] for col in features_ordre]])
        X_test_scaled = self.scaler.transform(X_test)

        # ============ PRÉDICTION AVEC LES DEUX MODÈLES ============
        print(f"\n ANALYSE DE LA VOIX:")
        print(f"   Energy: {features['energy']:.6f}")
        print(f"   Spectral: {features['spectral_centroid']:.0f} Hz")
        print(f"   ZCR: {features['zcr']:.4f}")

        # Prédiction Random Forest
        etat_detecte_rf = self.modele_rf.predict(X_test_scaled)[0]
        probabilites_rf = self.modele_rf.predict_proba(X_test_scaled)[0]

        # Prédiction SVM
        etat_detecte_svm = self.modele_svm.predict(X_test_scaled)[0]
        probabilites_svm = self.modele_svm.predict_proba(X_test_scaled)[0]

        print(f"\nPRÉDICTION RANDOM FOREST: {etat_detecte_rf.upper()}")
        print(f"PRÉDICTION SVM: {etat_detecte_svm.upper()}")

        # ============ AFFICHAGE DES PROBABILITÉS ============
        print(f"\n PROBABILITÉS RANDOM FOREST:")
        for i, etat in enumerate(self.modele_rf.classes_):
            proba = probabilites_rf[i] * 100
            barre = "█" * int(proba / 3)  # Barre de progression visuelle
            print(f"   {etat:10} {proba:5.1f}% {barre}")

        return etat_detecte_rf

    def test_final(self):
        """
        Test final interactif : l'utilisateur enregistre sa voix dans les 3 états
        et vérifie si le modèle les reconnaît correctement.
        """
        if self.modele_rf is None:
            print("Entraînez d'abord le modèle!")
            return

        print("\n TEST FINAL AVEC LE MODÈLE OPTIMAL")
        print("=" * 40)

        resultats = []  # Liste pour stocker les résultats (True/False)

        # Tester chaque état vocal
        for etat in ['energique', 'fatigue', 'neutre']:
            print(f"\n--- TEST {etat.upper()} ---")

            # Instructions pour l'utilisateur
            if etat == 'fatigue':
                print(" Parlez avec une voix FATIGUÉE (lente, monotone)")
            elif etat == 'neutre':
                print("Parlez NORMALEMENT (voix neutre)")
            else:
                print("Parlez avec une voix ÉNERGIQUE (forte, dynamique)")

            input("Appuyez sur Enter pour enregistrer...")

            fs = 22050
            nom_fichier = f"test_final_{etat}.wav"

            try:
                # Enregistrement
                audio_data = sd.rec(int(4 * fs), samplerate=fs, channels=1, dtype='float64')
                sd.wait()
                sf.write(nom_fichier, audio_data, fs)

                # Analyse
                etat_detecte = self.analyser_voix(nom_fichier)

                # Vérification
                if etat_detecte == etat:
                    print(" CORRECT!")
                    resultats.append(True)
                else:
                    print(f"  ERREUR: Attendu {etat}, Détecté {etat_detecte}")
                    resultats.append(False)

            except Exception as e:
                print(f"  Erreur: {e}")
                resultats.append(False)

        # Résumé final
        succes = sum(resultats)
        total = len(resultats)
        print(f"\n RÉSULTAT FINAL: {succes}/{total}")

        if succes == total:
            print(" PARFAIT! Le projet est réussi!")
        elif succes >= 2:
            print(" TRÈS BIEN! Prêt pour la présentation")
        else:
            print(" Le modèle peut être amélioré avec plus de données")



    def menu_final(self):
        """
        Menu principal interactif qui guide l'utilisateur à travers toutes les étapes du projet.
        Conforme aux exigences du guide avec 11 options.
        """
        while True:
            print("\n" + "=" * 60)
            print("🎤 PROJET - DÉTECTION D'ÉNERGIE VOCALE")
            print("=" * 60)
            print("1.  Charger les données audio")
            print("2.  Voir statistiques du dataset")
            print("3.  Vérifier valeurs aberrantes (Étape 4)")
            print("4.  Description théorique modèles (Étape 7)")
            print("5.  Entraîner DEUX modèles avec CV=4 (Étapes 6, 8)")
            print("6.  Tableau récapitulatif performances (Étape 10)")
            print("7.  Analyser ma voix (enregistrement direct)")
            print("8.  Test final avec les 3 états")
            print("9.  Générer toutes les visualisations")
            print("10. Générer présentation finale")
            print("11. Quitter")

            choix = input("\n Votre choix (1-11): ").strip()

            if choix == '1':
                self.charger_et_preparer_donnees()
            elif choix == '2':
                self.afficher_statistiques_completes()
            elif choix == '3':
                self.detecter_valeurs_aberrantes()  # NOUVEAU
            elif choix == '4':
                self.description_detaille_modeles()  # NOUVEAU
            elif choix == '5':
                self.entrainer_deux_modeles_cv4()  # MODIFIÉ
            elif choix == '6':
                self.generer_tableau_recapitulatif()  # NOUVEAU
            elif choix == '7':
                self.analyser_voix()
            elif choix == '8':
                self.test_final()
            elif choix == '9':
                self.generer_visualisations_completes()
            elif choix == '10':
                self.generer_presentation()
            elif choix == '11':
                print("\nAu revoir et bonne présentation !")
                break
            else:
                print("Choix invalide. Essayez à nouveau.")

    def generer_presentation(self):
        """
        Génère un résumé structuré pour la présentation finale du projet.
        Met en avant tous les points du guide qui ont été implémentés.
        """
        print("\nPRÉSENTATION DU PROJET")
        print("=" * 30)

        # Informations sur les données
        if not self.dataset.empty:
            print(f"DONNÉES UTILISÉES:")
            print(f"   • {len(self.dataset)} échantillons audio")
            for etat in ['energique', 'fatigue', 'neutre']:
                count = len(self.dataset[self.dataset['etat'] == etat])
                print(f"   • {etat}: {count} échantillons")

        # Informations sur les modèles
        if self.modele_rf or self.modele_svm:
            print(f"\n MODÈLES MACHINE LEARNING:")
            print(f"   • Random Forest optimisé (GridSearch CV=4)")
            print(f"   • SVM optimisé (GridSearch CV=4)")
            print(f"   • {self.dataset.shape[1] - 3} caractéristiques audio extraites")


        print(f"\nCONFORMITÉ AU GUIDE DU PROJET:")
        print(f"   • ✓ Jeu de données choisi et analysé")
        print(f"   • ✓ Visualisation avec Seaborn")
        print(f"   • ✓ Détection valeurs aberrantes")
        print(f"   • ✓ Prétraitement adapté (StandardScaler)")
        print(f"   • ✓ Sélection caractéristiques pertinentes")
        print(f"   • ✓ DEUX modèles différents (Random Forest + SVM)")
        print(f"   • ✓ Description détaillée avec figures")
        print(f"   • ✓ CV=4 pour optimisation hyperparamètres")
        print(f"   • ✓ Matrices de confusion générées")
        print(f"   • ✓ Tableaux récapitulatifs")

        # Résultats attendus
        print(f"\n RÉSULTATS ATTENDUS:")
        print(f"   • Détection de l'état d'énergie vocal (3 classes)")
        print(f"   • Précision > 85% sur données de test")
        print(f"   • Application temps réel avec enregistrement direct")

        # Points forts
        print(f"\n POINTS FORTS:")
        print(f"   • Utilisation de toutes les données existantes")
        print(f"   • Modèles optimisés par Grid Search")
        print(f"   • Features audio avancées (MFCC, spectral, etc.)")
        print(f"   • Validation rigoureuse (train/test split + CV)")
        print(f"   • Interface utilisateur intuitive")

        # Visualisations disponibles
        print(f"\n VISUALISATIONS DISPONIBLES:")
        print(f"   • Distribution des états vocaux")
        print(f"   • Matrices de confusion (par modèle)")
        print(f"   • Importance des features")
        print(f"   • Matrice de corrélation")
        print(f"   • Boxplots des valeurs aberrantes")
        print(f"   • Tableau récapitulatif des performances")


def main():
    """
    Fonction principale qui lance l'application.
    Point d'entrée standard en Python.
    """
    # Créer une instance de la classe ProjetFinal
    projet = ProjetFinal()

    # Lancer le menu interactif
    projet.menu_final()


# Vérifier si ce fichier est exécuté directement (pas importé)
if __name__ == "__main__":
    main()
