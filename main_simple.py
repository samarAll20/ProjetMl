# main_simple.py
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.feature_selection import SelectKBest, f_classif
import warnings

warnings.filterwarnings('ignore')


class ProjetEnergieVoix:
    def __init__(self):
        self.data = None
        print("🚀 PROJET DE DÉTECTION D'ÉNERGIE VOCALE")
        print("=" * 50)

    def creer_structure(self):
        """Créer la structure de dossiers"""
        dossiers = [
            'data/enregistrements/fatigue',
            'data/enregistrements/neutre',
            'data/enregistrements/energique',
            'models',
            'resultats'
        ]

        for dossier in dossiers:
            os.makedirs(dossier, exist_ok=True)
            print(f"✅ Dossier créé: {dossier}")

    def generer_donnees_simulation(self):
        """Générer des données simulées réalistes"""
        print("\n🎵 Génération de données audio simulées...")

        np.random.seed(42)
        dataset = []

        # Profils acoustiques réalistes pour chaque état
        profils = {
            'fatigue': {
                'zero_crossing_rate': (0.04, 0.008),  # Très faible (voix monotone)
                'energy': (0.015, 0.004),  # Très faible énergie
                'spectral_centroid': (1100, 150),  # Fréquences très basses
                'spectral_rolloff': (2200, 250),  # Rolloff très bas
                'spectral_bandwidth': (1200, 100),  # Bande passante étroite
                'chroma_stft': (0.25, 0.04),  # Chroma très faible
                'mfcc_1': (-6, 0.8), 'mfcc_2': (-3, 0.8), 'mfcc_3': (-2, 0.8)
            },
            'neutre': {
                'zero_crossing_rate': (0.08, 0.01),  # Moyen
                'energy': (0.045, 0.005),  # Énergie moyenne
                'spectral_centroid': (2400, 180),  # Fréquences moyennes
                'spectral_rolloff': (4800, 280),  # Rolloff moyen
                'spectral_bandwidth': (1900, 120),  # Bande passante moyenne
                'chroma_stft': (0.55, 0.04),  # Chroma moyen
                'mfcc_1': (0, 0.8), 'mfcc_2': (0, 0.8), 'mfcc_3': (0, 0.8)
            },
            'energique': {
                'zero_crossing_rate': (0.14, 0.012),  # Élevé (voix dynamique)
                'energy': (0.075, 0.006),  # Haute énergie
                'spectral_centroid': (3600, 220),  # Fréquences hautes
                'spectral_rolloff': (7200, 320),  # Rolloff élevé
                'spectral_bandwidth': (2600, 140),  # Bande passante large
                'chroma_stft': (0.82, 0.05),  # Chroma fort
                'mfcc_1': (7, 0.9), 'mfcc_2': (3, 0.9), 'mfcc_3': (2, 0.9)
            }
        }

        for categorie, params in profils.items():
            for i in range(25):  # 25 échantillons par catégorie
                features = {}
                for feature, (mean, std) in params.items():
                    # Ajouter un peu de bruit pour plus de réalisme
                    valeur = np.random.normal(mean, std)
                    features[feature] = max(valeur, 0)  # Éviter les valeurs négatives

                features['categorie'] = categorie
                features['fichier'] = f"{categorie}_{i + 1:02d}.wav"
                dataset.append(features)

        self.data = pd.DataFrame(dataset)
        print(f"✅ Dataset créé: {len(self.data)} échantillons (25 par catégorie)")
        return self.data

    def analyser_donnees(self):
        """Analyse exploratoire des données"""
        print("\n📊 Analyse exploratoire des données...")

        # Statistiques descriptives
        print("\n📈 STATISTIQUES DESCRIPTIVES PAR CATÉGORIE:")
        stats = self.data.groupby('categorie').agg({
            'energy': ['mean', 'std', 'min', 'max'],
            'spectral_centroid': ['mean', 'std'],
            'zero_crossing_rate': ['mean', 'std'],
            'chroma_stft': ['mean', 'std']
        }).round(4)

        print(stats)

        # Sauvegarder les statistiques
        stats.to_csv('resultats/statistiques_descriptives.csv')
        print("💾 Statistiques sauvegardées dans: resultats/statistiques_descriptives.csv")

        return stats

    def visualiser_donnees(self):
        """Visualisations complètes avec seaborn"""
        print("\n📊 Création des visualisations...")

        # Configuration des styles
        plt.style.use('default')
        sns.set_palette("husl")

        # 1. Distribution des catégories
        plt.figure(figsize=(15, 12))

        plt.subplot(3, 2, 1)
        counts = self.data['categorie'].value_counts()
        plt.pie(counts.values, labels=counts.index, autopct='%1.1f%%', startangle=90)
        plt.title('Répartition des Catégories', fontsize=14, fontweight='bold')

        # 2. Boxplots des features principales
        features_principales = ['zero_crossing_rate', 'energy', 'spectral_centroid', 'chroma_stft']

        for i, feature in enumerate(features_principales, 2):
            plt.subplot(3, 2, i)
            sns.boxplot(data=self.data, x='categorie', y=feature)
            plt.title(f'Distribution de {feature}', fontsize=12)
            plt.xticks(rotation=45)
            plt.grid(True, alpha=0.3)

        # 3. Violin plots pour voir les densités
        plt.subplot(3, 2, 6)
        sns.violinplot(data=self.data, x='categorie', y='energy', inner='quartile')
        plt.title('Densité de Energy par Catégorie', fontsize=12)
        plt.xticks(rotation=45)
        plt.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig('resultats/visualisation_categories.png', dpi=300, bbox_inches='tight')
        plt.show()

        # 4. Matrice de corrélation
        plt.figure(figsize=(12, 10))
        features_numeriques = self.data.select_dtypes(include=[np.number]).columns

        # Sélectionner les 10 features les plus importantes
        features_selection = features_numeriques[:10]
        corr_matrix = self.data[features_selection].corr()

        mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
        sns.heatmap(corr_matrix, mask=mask, annot=True, cmap='RdBu_r', center=0,
                    fmt='.2f', square=True, cbar_kws={"shrink": .8})
        plt.title('Matrice de Corrélation des Caractéristiques Audio',
                  fontsize=14, fontweight='bold', pad=20)
        plt.tight_layout()
        plt.savefig('resultats/matrice_correlation.png', dpi=300, bbox_inches='tight')
        plt.show()

        # 5. Scatter plot avec regression
        plt.figure(figsize=(12, 5))

        plt.subplot(1, 2, 1)
        sns.scatterplot(data=self.data, x='energy', y='spectral_centroid',
                        hue='categorie', s=80, alpha=0.7)
        plt.title('Energy vs Spectral Centroid', fontsize=12, fontweight='bold')
        plt.xlabel('Energy')
        plt.ylabel('Spectral Centroid')
        plt.legend(title='Catégorie')
        plt.grid(True, alpha=0.3)

        plt.subplot(1, 2, 2)
        sns.scatterplot(data=self.data, x='zero_crossing_rate', y='chroma_stft',
                        hue='categorie', s=80, alpha=0.7)
        plt.title('ZCR vs Chroma STFT', fontsize=12, fontweight='bold')
        plt.xlabel('Zero Crossing Rate')
        plt.ylabel('Chroma STFT')
        plt.legend(title='Catégorie')
        plt.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig('resultats/scatter_plots.png', dpi=300, bbox_inches='tight')
        plt.show()

        print("💾 Visualisations sauvegardées dans le dossier 'resultats/'")

    def detecter_valeurs_aberrantes(self):
        """Détection des valeurs aberrantes"""
        print("\n🔍 Détection des valeurs aberrantes...")

        features_numeriques = self.data.select_dtypes(include=[np.number]).columns

        plt.figure(figsize=(15, 10))
        for i, feature in enumerate(features_numeriques[:6], 1):
            plt.subplot(2, 3, i)
            sns.boxplot(data=self.data, y=feature, x='categorie')
            plt.title(f'Valeurs aberrantes - {feature}')
            plt.xticks(rotation=45)
            plt.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig('resultats/valeurs_aberrantes.png', dpi=300, bbox_inches='tight')
        plt.show()

        # Détection statistique
        print("\n📊 DÉTECTION STATISTIQUE DES VALEURS ABERRANTES:")
        outlier_report = []

        for feature in features_numeriques[:6]:
            Q1 = self.data[feature].quantile(0.25)
            Q3 = self.data[feature].quantile(0.75)
            IQR = Q3 - Q1
            limite_inf = Q1 - 1.5 * IQR
            limite_sup = Q3 + 1.5 * IQR

            outliers = self.data[(self.data[feature] < limite_inf) | (self.data[feature] > limite_sup)]
            count_outliers = len(outliers)

            print(f"  {feature}: {count_outliers} valeurs aberrantes")
            outlier_report.append({
                'Feature': feature,
                'Valeurs_aberrantes': count_outliers,
                'Pourcentage': f"{(count_outliers / len(self.data)) * 100:.1f}%"
            })

        # Sauvegarder le rapport
        outlier_df = pd.DataFrame(outlier_report)
        outlier_df.to_csv('resultats/rapport_valeurs_aberrantes.csv', index=False)
        print("💾 Rapport des valeurs aberrantes sauvegardé")

    def selection_caracteristiques(self, X, y):
        """Sélection des caractéristiques les plus pertinentes"""
        print("\n🎯 Sélection des caractéristiques les plus pertinentes...")

        # Sélection des k meilleures features
        selector = SelectKBest(score_func=f_classif, k=8)
        X_selected = selector.fit_transform(X, y)

        # Obtenir les scores
        scores = selector.scores_
        feature_names = X.columns

        # Créer un DataFrame avec les scores
        feature_scores = pd.DataFrame({
            'Feature': feature_names,
            'Score_F': scores
        }).sort_values('Score_F', ascending=False)

        print("\n🏆 TOP 10 DES CARACTÉRISTIQUES LES PLUS IMPORTANTES:")
        print(feature_scores.head(10).to_string(index=False))

        # Visualisation
        plt.figure(figsize=(10, 6))
        top_features = feature_scores.head(10)
        sns.barplot(data=top_features, x='Score_F', y='Feature', palette='viridis')
        plt.title('Top 10 des Caractéristiques les Plus Importantes (Test F)')
        plt.xlabel('Score F')
        plt.tight_layout()
        plt.savefig('resultats/importance_caracteristiques.png', dpi=300, bbox_inches='tight')
        plt.show()

        # Sauvegarder les résultats
        feature_scores.to_csv('resultats/selection_caracteristiques.csv', index=False)

        return X_selected, selector, feature_scores

    def entrainer_modeles(self):
        """Entraînement et optimisation des modèles avec validation croisée"""
        print("\n🤖 Entraînement des modèles de Machine Learning...")

        # Préparation des données
        X = self.data.drop(['categorie', 'fichier'], axis=1)
        y = self.data['categorie']

        # Encodage des labels
        le = LabelEncoder()
        y_encoded = le.fit_transform(y)

        # Split des données (80% train, 20% test)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
        )

        # Standardisation
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        print(f"📊 DONNÉES PRÉPARÉES:")
        print(f"  - Ensemble d'entraînement: {X_train.shape[0]} échantillons")
        print(f"  - Ensemble de test: {X_test.shape[0]} échantillons")
        print(f"  - Nombre de caractéristiques: {X_train.shape[1]}")
        print(f"  - Distribution des classes: {np.unique(y_encoded, return_counts=True)}")

        # Sélection des caractéristiques
        X_train_selected, selector, feature_scores = self.selection_caracteristiques(
            pd.DataFrame(X_train_scaled, columns=X.columns), y_train
        )
        X_test_selected = selector.transform(X_test_scaled)

        # Définition des modèles et hyperparamètres pour GridSearch
        modeles = {
            'Random Forest': {
                'model': RandomForestClassifier(random_state=42),
                'params': {
                    'n_estimators': [50, 100, 150],
                    'max_depth': [10, 20, None],
                    'min_samples_split': [2, 5],
                    'min_samples_leaf': [1, 2]
                }
            },
            'SVM': {
                'model': SVC(random_state=42),
                'params': {
                    'C': [0.1, 1, 10],
                    'kernel': ['linear', 'rbf'],
                    'gamma': ['scale', 'auto']
                }
            },
            'Regression Logistique': {
                'model': LogisticRegression(random_state=42, max_iter=1000),
                'params': {
                    'C': [0.1, 1, 10],
                    'solver': ['liblinear', 'lbfgs'],
                    'penalty': ['l2']
                }
            }
        }

        resultats = {}

        for nom_modele, config in modeles.items():
            print(f"\n--- ENTRAÎNEMENT: {nom_modele} ---")

            # Grid Search avec validation croisée (CV=4 comme demandé)
            grid_search = GridSearchCV(
                config['model'],
                config['params'],
                cv=4,  # Validation croisée à 4 folds
                scoring='accuracy',
                n_jobs=-1,
                verbose=1
            )

            # Entraînement
            grid_search.fit(X_train_selected, y_train)

            # Prédictions
            y_pred = grid_search.predict(X_test_selected)
            accuracy = accuracy_score(y_test, y_pred)

            # Scores de validation croisée
            cv_scores = grid_search.cv_results_['mean_test_score']

            # Stockage des résultats
            resultats[nom_modele] = {
                'model': grid_search.best_estimator_,
                'best_params': grid_search.best_params_,
                'best_score': grid_search.best_score_,
                'accuracy_test': accuracy,
                'predictions': y_pred,
                'cv_scores': cv_scores,
                'grid_search': grid_search
            }

            print(f"✅ {nom_modele} - OPTIMISATION TERMINÉE")
            print(f"   Score CV (moyenne): {grid_search.best_score_:.3f}")
            print(f"   Accuracy sur test: {accuracy:.3f}")
            print(f"   Meilleurs paramètres: {grid_search.best_params_}")

        return resultats, X_test_selected, y_test, le, feature_scores

    def evaluer_modeles(self, resultats, y_test, label_encoder, feature_scores):
        """Évaluation complète des modèles"""
        print("\n📈 ÉVALUATION DÉTAILLÉE DES MODÈLES")
        print("=" * 60)

        labels = label_encoder.classes_

        # 1. Tableau récapitulatif des performances
        print("\n📊 TABLEAU RÉCAPITULATIF DES PERFORMANCES")
        print("-" * 60)

        resume_data = []
        for nom_modele, res in resultats.items():
            resume_data.append({
                'Modèle': nom_modele,
                'Score CV (moyenne)': f"{res['best_score']:.3f}",
                'Accuracy Test': f"{res['accuracy_test']:.3f}",
                'Écart': f"{(res['accuracy_test'] - res['best_score']):.3f}"
            })

        resume_df = pd.DataFrame(resume_data)
        print(resume_df.to_string(index=False))

        # Sauvegarder le tableau récapitulatif
        resume_df.to_csv('resultats/tableau_recapitulatif.csv', index=False)

        # 2. Matrices de confusion
        print("\n🎯 MATRICES DE CONFUSION")
        print("-" * 40)

        n_modeles = len(resultats)
        plt.figure(figsize=(6 * n_modeles, 5))

        for i, (nom_modele, res) in enumerate(resultats.items(), 1):
            plt.subplot(1, n_modeles, i)

            y_pred = res['predictions']
            cm = confusion_matrix(y_test, y_pred)

            # Heatmap avec annotations
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                        xticklabels=labels, yticklabels=labels,
                        cbar_kws={'label': 'Nombre d\'échantillons'})

            plt.title(f'{nom_modele}\nAccuracy: {res["accuracy_test"]:.3f}',
                      fontweight='bold', fontsize=12)
            plt.xlabel('Prédictions', fontweight='bold')
            plt.ylabel('Vérités terrain', fontweight='bold')

        plt.tight_layout()
        plt.savefig('resultats/matrices_confusion.png', dpi=300, bbox_inches='tight')
        plt.show()

        # 3. Rapports de classification détaillés
        print("\n📋 RAPPORTS DE CLASSIFICATION DÉTAILLÉS")
        print("=" * 50)

        rapports = {}
        for nom_modele, res in resultats.items():
            print(f"\n--- {nom_modele.upper()} ---")
            rapport = classification_report(y_test, res['predictions'],
                                            target_names=labels, digits=3,
                                            output_dict=True)
            print(classification_report(y_test, res['predictions'],
                                        target_names=labels, digits=3))
            rapports[nom_modele] = rapport

        # 4. Comparaison visuelle des performances
        plt.figure(figsize=(10, 6))

        model_names = list(resultats.keys())
        cv_scores = [resultats[nom]['best_score'] for nom in model_names]
        test_scores = [resultats[nom]['accuracy_test'] for nom in model_names]

        x = np.arange(len(model_names))
        width = 0.35

        plt.bar(x - width / 2, cv_scores, width, label='Score CV', alpha=0.8)
        plt.bar(x + width / 2, test_scores, width, label='Accuracy Test', alpha=0.8)

        plt.xlabel('Modèles')
        plt.ylabel('Score')
        plt.title('Comparaison des Performances des Modèles')
        plt.xticks(x, model_names)
        plt.legend()
        plt.grid(True, alpha=0.3)

        # Ajouter les valeurs sur les barres
        for i, v in enumerate(cv_scores):
            plt.text(i - width / 2, v + 0.01, f'{v:.3f}', ha='center')
        for i, v in enumerate(test_scores):
            plt.text(i + width / 2, v + 0.01, f'{v:.3f}', ha='center')

        plt.tight_layout()
        plt.savefig('resultats/comparaison_modeles.png', dpi=300, bbox_inches='tight')
        plt.show()

        print("💾 Tous les résultats sont sauvegardés dans le dossier 'resultats/'")

    def lancer_analyse_complete(self):
        """Lancer l'analyse complète du projet"""
        try:
            print("🚀 DÉMARRAGE DE L'ANALYSE COMPLÈTE")
            print("=" * 50)

            # 1. Créer la structure de dossiers
            self.creer_structure()

            # 2. Générer les données simulées
            data = self.generer_donnees_simulation()

            # 3. Analyse exploratoire des données
            self.analyser_donnees()

            # 4. Visualisations des données
            self.visualiser_donnees()

            # 5. Détection des valeurs aberrantes
            self.detecter_valeurs_aberrantes()

            # 6. Entraînement des modèles avec optimisation
            resultats, X_test, y_test, label_encoder, feature_scores = self.entrainer_modeles()

            # 7. Évaluation complète
            self.evaluer_modeles(resultats, y_test, label_encoder, feature_scores)

            # 8. Résumé final
            print("\n🎉" + "=" * 60)
            print("✅ PROJET TERMINÉ AVEC SUCCÈS!")
            print("=" * 60)
            print("\n📁 RÉSULTATS GÉNÉRÉS:")
            print("├── resultats/visualisation_categories.png")
            print("├── resultats/matrice_correlation.png")
            print("├── resultats/scatter_plots.png")
            print("├── resultats/valeurs_aberrantes.png")
            print("├── resultats/importance_caracteristiques.png")
            print("├── resultats/matrices_confusion.png")
            print("├── resultats/comparaison_modeles.png")
            print("├── resultats/statistiques_descriptives.csv")
            print("├── resultats/selection_caracteristiques.csv")
            print("├── resultats/rapport_valeurs_aberrantes.csv")
            print("└── resultats/tableau_recapitulatif.csv")
            print("\n✨ Le projet est prêt pour la présentation!")

        except Exception as e:
            print(f"❌ Erreur lors de l'exécution: {e}")
            print("💡 Vérifiez que toutes les dépendances sont installées")


def main():
    """Fonction principale"""
    projet = ProjetEnergieVoix()
    projet.lancer_analyse_complete()


if __name__ == "__main__":
    main()