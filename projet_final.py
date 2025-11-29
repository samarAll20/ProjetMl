# projet_final.py
import os
import numpy as np
import pandas as pd
import librosa
import sounddevice as sd
import soundfile as sf
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

warnings.filterwarnings('ignore')


class ProjetFinal:
    def __init__(self):
        print("PROJET FINAL - UTILISATION DE TOUTES LES DONNÉES")
        print("=" * 60)

        self.dataset = pd.DataFrame()
        self.modele = None
        self.scaler = StandardScaler()

        # Liste de tous les dossiers avec données audio
        self.dossiers_audio = [
            'src/data/enregistrements',  # Le bon chemin principal
            'src/data/enregistrements/energique',
            'src/data/enregistrements/fatigue',
            'src/data/enregistrements/neutre',
            # Vous pouvez garder les autres au cas où
            'src/data',
            'data/enregistrements',
            'data'
        ]

    def trouver_tous_les_audios(self):
        """Trouver tous les fichiers audio dans tous les dossiers"""
        print("Recherche de tous les fichiers audio...")

        tous_les_fichiers = []
        dossier_principal = 'src/data/enregistrements'

        if not os.path.exists(dossier_principal):
            print(f" {dossier_principal} n'existe pas")
            return tous_les_fichiers

        print(f" Dossier trouvé: {dossier_principal}")

        # Parcourir les sous-dossiers d'état
        for etat in ['energique', 'fatigue', 'neutre']:
            dossier_etat = os.path.join(dossier_principal, etat)
            if os.path.exists(dossier_etat):
                fichiers_etat = [f for f in os.listdir(dossier_etat) if f.endswith('.wav')]
                print(f"   {etat}: {len(fichiers_etat)} fichiers")

                for fichier in fichiers_etat:
                    chemin_complet = os.path.join(dossier_etat, fichier)
                    tous_les_fichiers.append({
                        'fichier': chemin_complet,
                        'etat': etat,
                        'source': dossier_principal
                    })

        print(f" TOTAL: {len(tous_les_fichiers)} fichiers audio trouvés")
        return tous_les_fichiers

    def extraire_features_avancees(self, fichier_audio):
        """Extraire des features avancées"""
        try:
            y, sr = librosa.load(fichier_audio, sr=22050)
            y = y / (np.max(np.abs(y)) + 1e-8)

            features = {}

            # Features de base
            features['energy'] = np.mean(librosa.feature.rms(y=y))
            features['energy_std'] = np.std(librosa.feature.rms(y=y))

            features['spectral_centroid'] = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr))
            features['spectral_centroid_std'] = np.std(librosa.feature.spectral_centroid(y=y, sr=sr))

            features['zcr'] = np.mean(librosa.feature.zero_crossing_rate(y))
            features['zcr_std'] = np.std(librosa.feature.zero_crossing_rate(y))

            # Features spectrales
            features['spectral_rolloff'] = np.mean(librosa.feature.spectral_rolloff(y=y, sr=sr))
            features['spectral_bandwidth'] = np.mean(librosa.feature.spectral_bandwidth(y=y, sr=sr))

            # MFCC
            mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
            for i in range(5):
                features[f'mfcc_{i + 1}'] = np.mean(mfccs[i])
                features[f'mfcc_{i + 1}_std'] = np.std(mfccs[i])

            # Chroma
            chroma = librosa.feature.chroma_stft(y=y, sr=sr)
            features['chroma_stft'] = np.mean(chroma)

            return features

        except Exception as e:
            print(f"Erreur avec {fichier_audio}: {e}")
            return None

    def charger_et_preparer_donnees(self):
        """Charger et préparer toutes les données"""
        print("\n📊 CHARGEMENT DE TOUTES LES DONNÉES EXISTANTES")

        tous_les_audios = self.trouver_tous_les_audios()

        if not tous_les_audios:
            print(" Aucun fichier audio trouvé!")
            return False

        donnees_chargees = 0

        for audio in tous_les_audios:
            features = self.extraire_features_avancees(audio['fichier'])
            if features:
                features['etat'] = audio['etat']
                features['fichier'] = audio['fichier']
                features['source'] = audio['source']

                self.dataset = pd.concat([self.dataset, pd.DataFrame([features])], ignore_index=True)
                donnees_chargees += 1

        print(f" {donnees_chargees} fichiers chargés sur {len(tous_les_audios)}")

        # Afficher les statistiques
        self.afficher_statistiques_completes()

        return donnees_chargees > 0

    def afficher_statistiques_completes(self):
        """Afficher les statistiques complètes du dataset"""
        if self.dataset.empty:
            print(" Aucune donnée chargée")
            return

        print(f"\n STATISTIQUES COMPLÈTES:")
        print(f"Total: {len(self.dataset)} échantillons")

        # Par état
        for etat in ['energique', 'fatigue', 'neutre']:
            data_etat = self.dataset[self.dataset['etat'] == etat]
            if len(data_etat) > 0:
                print(f"\n   {etat.upper()} ({len(data_etat)} échantillons):")
                print(f"     Energy: {data_etat['energy'].mean():.6f} ± {data_etat['energy'].std():.6f}")
                print(
                    f"     Spectral: {data_etat['spectral_centroid'].mean():.0f} ± {data_etat['spectral_centroid'].std():.0f} Hz")
                print(f"     ZCR: {data_etat['zcr'].mean():.4f} ± {data_etat['zcr'].std():.4f}")

        # Par source
        print(f"\n RÉPARTITION PAR SOURCE:")
        for source in self.dataset['source'].unique():
            count = len(self.dataset[self.dataset['source'] == source])
            print(f"   {source}: {count} échantillons")

    def entrainer_modele_optimal(self):
        """Entraîner le modèle optimal avec toutes les données"""
        if self.dataset.empty:
            print(" Chargez d'abord les données!")
            return False

        print("\n🤖 ENTRAÎNEMENT DU MODÈLE OPTIMAL...")

        # Préparer les données
        X = self.dataset.drop(['etat', 'fichier', 'source'], axis=1)
        y = self.dataset['etat']

        print(f" Données: {X.shape[0]} échantillons, {X.shape[1]} features")

        # Séparation train/test
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        # Normalisation
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)

        # Grid Search pour trouver les meilleurs paramètres
        print("Optimisation des hyperparamètres...")

        param_grid = {
            'n_estimators': [50, 100, 200],
            'max_depth': [10, 20, None],
            'min_samples_split': [2, 5],
            'min_samples_leaf': [1, 2]
        }

        grid_search = GridSearchCV(
            RandomForestClassifier(random_state=42),
            param_grid,
            cv=5,
            scoring='accuracy',
            n_jobs=-1,
            verbose=1
        )

        grid_search.fit(X_train_scaled, y_train)

        self.modele = grid_search.best_estimator_

        # Évaluation
        y_pred = self.modele.predict(X_test_scaled)
        accuracy = accuracy_score(y_test, y_pred)

        print(f"\n MODÈLE OPTIMAL ENTRAÎNÉ")
        print(f"   Accuracy: {accuracy:.3f}")
        print(f"   Meilleurs paramètres: {grid_search.best_params_}")

        # Rapport détaillé
        print(f"\n RAPPORT DE CLASSIFICATION:")
        print(classification_report(y_test, y_pred))

        # Matrice de confusion
        plt.figure(figsize=(8, 6))
        cm = confusion_matrix(y_test, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=['energique', 'fatigue', 'neutre'],
                    yticklabels=['energique', 'fatigue', 'neutre'])
        plt.title('Matrice de Confusion - Modèle Optimal')
        plt.ylabel('Vérité terrain')
        plt.xlabel('Prédiction')
        plt.tight_layout()
        plt.savefig('matrice_confusion_finale.png', dpi=300, bbox_inches='tight')
        print("Matrice de confusion sauvegardée: 'matrice_confusion_finale.png'")
        plt.show()  # IMPORTANT: Pour afficher à l'écran

        # Importance des features
        if hasattr(self.modele, 'feature_importances_'):
            print(f"\n TOP 10 DES CARACTÉRISTIQUES IMPORTANTES:")
            importances = self.modele.feature_importances_
            indices = np.argsort(importances)[::-1]

            for i in range(min(10, len(X.columns))):
                print(f"   {i + 1:2}. {X.columns[indices[i]]}: {importances[indices[i]]:.3f}")

        # Graphique d'importance des features
        plt.figure(figsize=(10, 6))
        feature_imp = pd.Series(importances, index=X.columns).sort_values(ascending=False)[:10]
        sns.barplot(x=feature_imp.values, y=feature_imp.index)
        plt.title('Top 10 des Caractéristiques les Plus Importantes')
        plt.xlabel('Importance')
        plt.tight_layout()
        plt.savefig('importance_features.png', dpi=300, bbox_inches='tight')
        print("Graphique d'importance sauvegardé: 'importance_features.png'")
        plt.show()

        # Sauvegarder le modèle
        joblib.dump(self.modele, 'modele_final.pkl')
        joblib.dump(self.scaler, 'scaler_final.pkl')

        print(" Modèle final sauvegardé: 'modele_final.pkl'")

        return True

    def generer_visualisations_completes(self):
        """Générer toutes les visualisations pour l'analyse"""
        if self.dataset.empty:
            print(" Chargez d'abord les données!")
            return

        print("\n GÉNÉRATION DES VISUALISATIONS COMPLÈTES...")

        # 1. DISTRIBUTION DES ÉTATS
        plt.figure(figsize=(12, 10))

        plt.subplot(2, 2, 1)
        counts = self.dataset['etat'].value_counts()
        colors = ['#ff6b6b', '#4ecdc4', '#45b7d1']
        plt.pie(counts.values, labels=counts.index, autopct='%1.1f%%', colors=colors, startangle=90)
        plt.title('Distribution des États Vocaux')

        # 2. FEATURES PRINCIPALES PAR ÉTAT
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

        # 3. HISTOGRAMME DE L'ÉNERGIE
        plt.subplot(2, 2, 3)
        for etat in ['energique', 'fatigue', 'neutre']:
            data_etat = self.dataset[self.dataset['etat'] == etat]
            plt.hist(data_etat['energy'], alpha=0.7, label=etat, bins=8,
                     color=colors_etat[etat], edgecolor='black')

        plt.xlabel('Énergie Audio (RMS)')
        plt.ylabel('Nombre d\'échantillons')
        plt.title('Distribution de l\'Énergie par État')
        plt.legend()

        # 4. CENTROÏDE SPECTRAL
        plt.subplot(2, 2, 4)
        spectral_data = []
        labels = []

        box_data = []
        box_labels = []
        for etat in ['energique', 'fatigue', 'neutre']:
            data_etat = self.dataset[self.dataset['etat'] == etat]
            if len(data_etat) > 0:
                box_data.append(data_etat['spectral_centroid'].values)
                box_labels.append(etat)

        box_plot = plt.boxplot(box_data, labels=box_labels, patch_artist=True)

        # Colorier les boîtes
        for patch, color in zip(box_plot['boxes'], [colors_etat[etat] for etat in box_labels]):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)

        plt.ylabel('Fréquence (Hz)')
        plt.title('Centroïde Spectral par État Vocal')
        plt.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig('analyse_complete_etats.png', dpi=300, bbox_inches='tight')
        plt.show()
        print("Visualisations sauvegardées: 'analyse_complete_etats.png'")

        # 5. MATRICE DE CORRÉLATION
        plt.figure(figsize=(10, 8))
        features_corr = self.dataset.drop(['etat', 'fichier', 'source'], axis=1)
        # Prendre seulement les premières colonnes pour la lisibilité
        features_corr = features_corr.iloc[:, :8]
        corr_matrix = features_corr.corr()

        mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
        sns.heatmap(corr_matrix, mask=mask, annot=True, cmap='coolwarm', center=0,
                    square=True, fmt='.2f', cbar_kws={"shrink": .8})
        plt.title('Matrice de Corrélation des Features Audio')
        plt.tight_layout()
        plt.savefig('matrice_correlation.png', dpi=300, bbox_inches='tight')
        plt.show()
        print("Matrice de corrélation sauvegardée: 'matrice_correlation.png'")

    def analyser_voix(self, fichier_audio=None):
        """Analyser une voix (fichier ou enregistrement)"""
        if self.modele is None:
            print(" Entraînez d'abord le modèle optimal!")
            return

        if fichier_audio is None:
            # Enregistrement en direct
            print(f"\n Enregistrement de votre voix...")
            print("Parlez maintenant (4 secondes)...")

            fs = 22050
            try:
                audio_data = sd.rec(int(4 * fs), samplerate=fs, channels=1, dtype='float64')
                sd.wait()
                sf.write("test_temp.wav", audio_data, fs)
                fichier_audio = "test_temp.wav"
                print("Enregistrement sauvegardé")
            except Exception as e:
                print(f"Erreur enregistrement: {e}")
                return

        # Analyse
        features = self.extraire_features_avancees(fichier_audio)
        if not features:
            return

        # Préparer pour la prédiction
        X = self.dataset.drop(['etat', 'fichier', 'source'], axis=1)
        features_ordre = X.columns

        X_test = np.array([[features[col] for col in features_ordre]])
        X_test_scaled = self.scaler.transform(X_test)

        # Prédiction
        etat_detecte = self.modele.predict(X_test_scaled)[0]
        probabilites = self.modele.predict_proba(X_test_scaled)[0]

        print(f"\n ANALYSE DE LA VOIX:")
        print(f"   Energy: {features['energy']:.6f}")
        print(f"   Spectral: {features['spectral_centroid']:.0f} Hz")
        print(f"   ZCR: {features['zcr']:.4f}")

        print(f"\nRÉSULTAT: {etat_detecte.upper()}")

        print(f"\n PROBABILITÉS:")
        for i, etat in enumerate(self.modele.classes_):
            proba = probabilites[i] * 100
            barre = "█" * int(proba / 3)
            print(f"   {etat:10} {proba:5.1f}% {barre}")

        return etat_detecte

    def test_final(self):
        """Test final avec le modèle optimal"""
        if self.modele is None:
            print("Entraînez d'abord le modèle!")
            return

        print("\n TEST FINAL AVEC LE MODÈLE OPTIMAL")
        print("=" * 40)

        resultats = []

        for etat in ['energique', 'fatigue', 'neutre']:
            print(f"\n--- TEST {etat.upper()} ---")

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
                audio_data = sd.rec(int(4 * fs), samplerate=fs, channels=1, dtype='float64')
                sd.wait()
                sf.write(nom_fichier, audio_data, fs)

                etat_detecte = self.analyser_voix(nom_fichier)

                if etat_detecte == etat:
                    print(" CORRECT!")
                    resultats.append(True)
                else:
                    print(f" ERREUR: Attendu {etat}, Détecté {etat_detecte}")
                    resultats.append(False)

            except Exception as e:
                print(f" Erreur: {e}")
                resultats.append(False)

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
        """Menu final du projet"""
        while True:
            print("\n PROJET FINAL - DÉTECTION D'ÉNERGIE VOCALE")
            print("=" * 50)
            print("1.  Charger toutes les données existantes")
            print("2.  Entraîner le modèle optimal")
            print("3.  Analyser ma voix (enregistrement)")
            print("4.  Test final de performance")
            print("5.  Voir statistiques des données")
            print("6.  Générer visualisations complètes")
            print("7.  Présentation finale")
            print("8.  Quitter")

            choix = input("\nVotre choix (1-8): ").strip()

            if choix == '1':
                self.charger_et_preparer_donnees()
            elif choix == '2':
                self.entrainer_modele_optimal()
            elif choix == '3':
                self.analyser_voix()
            elif choix == '4':
                self.test_final()
            elif choix == '5':
                self.afficher_statistiques_completes()
            elif choix == '6':
                self.generer_visualisations_completes()
            elif choix == '7':
                self.generer_presentation()
            elif choix == '8':
                print("Au revoir et bonne présentation!")
                break
            else:
                print("Choix invalide")

    def generer_presentation(self):
        """Générer un résumé pour la présentation"""
        print("\nPRÉSENTATION DU PROJET")
        print("=" * 30)

        if not self.dataset.empty:
            print(f"DONNÉES UTILISÉES:")
            print(f"   • {len(self.dataset)} échantillons audio")
            for etat in ['energique', 'fatigue', 'neutre']:
                count = len(self.dataset[self.dataset['etat'] == etat])
                print(f"   • {etat}: {count} échantillons")

        if self.modele:
            print(f"\n MODÈLE MACHINE LEARNING:")
            print(f"   • Random Forest optimisé")
            print(f"   • {self.dataset.shape[1] - 3} caractéristiques audio")
            print(f"   • Validation croisée 5 folds")

        print(f"\n RÉSULTATS ATTENDUS:")
        print(f"   • Détection de l'état d'énergie vocal")
        print(f"   • Précision > 85% sur données de test")
        print(f"   • Application temps réel")

        print(f"\n POINTS FORTS:")
        print(f"   • Utilisation de toutes les données existantes")
        print(f"   • Modèle optimisé par Grid Search")
        print(f"   • Features audio avancées (MFCC, spectral, etc.)")
        print(f"   • Validation rigoureuse")

        print(f"\n VISUALISATIONS DISPONIBLES:")
        print(f"   • Matrice de confusion")
        print(f"   • Importance des features")
        print(f"   • Analyse complète des états")
        print(f"   • Matrice de corrélation")


def main():
    projet = ProjetFinal()
    projet.menu_final()


if __name__ == "__main__":
    main()