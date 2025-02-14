import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import os
import sys
from datetime import datetime
import time

# Configuration des chemins
class Config:
    # Chemins des données
    EXTRAITS = {
        'ACCEPTED': "../../Data/CSV/extrait_accepted_2007_to_2018Q4.csv",
        'REJECTED': "../../Data/CSV/extrait_rejected_2007_to_2018Q4.csv",
        'OUTPUT_DIR': "extraits"
    }
    
    COMPLET = {
        'ACCEPTED': "../../Data/CSV/accepted_2007_to_2018Q4.csv",
        'REJECTED': "../../Data/CSV/rejected_2007_to_2018Q4.csv",
        'OUTPUT_DIR': "complet"
    }
    
    @staticmethod
    def create_output_dirs(base_dir):
        """Crée la structure de dossiers pour les sorties"""
        figures_dir = os.path.join(base_dir, "figures")
        results_dir = os.path.join(base_dir, "resultats")
        os.makedirs(figures_dir, exist_ok=True)
        os.makedirs(results_dir, exist_ok=True)
        return figures_dir, results_dir

class ProgressTracker:
    """
    Suit la progression de l'analyse et le temps d'exécution
    """
    def __init__(self, total_steps):
        self.total_steps = total_steps
        self.current_step = 0
        self.start_time = time.time()
        self.last_step_time = self.start_time
    
    def update(self, message):
        self.current_step += 1
        current_time = time.time()
        step_duration = current_time - self.last_step_time
        total_duration = current_time - self.start_time
        
        progress = (self.current_step / self.total_steps) * 100
        print(f"\n[{progress:.1f}% terminé] {message}")
        print(f"Temps pour cette étape : {step_duration:.2f} secondes")
        print(f"Temps total écoulé : {total_duration:.2f} secondes")
        
        self.last_step_time = current_time

class TeeOutput:
    """
    Redirige la sortie vers la console et un fichier
    """
    def __init__(self, filename):
        self.file = open(filename, 'w', encoding='utf-8')
        self.stdout = sys.stdout
        sys.stdout = self

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        sys.stdout = self.stdout
        self.file.close()

    def write(self, data):
        self.file.write(data)
        self.stdout.write(data)
        self.file.flush()
        self.stdout.flush()

    def flush(self):
        self.file.flush()
        self.stdout.flush()

def afficher_correspondances():
    """
    Affiche les correspondances entre les colonnes des deux fichiers
    """
    print("Correspondances entre les colonnes :")
    print("Prêts rejetés -> Prêts acceptés")
    print("-" * 40)
    print("Amount Requested -> loan_amnt")
    print("Risk_Score -> fico_range_low")
    print("Debt-To-Income Ratio -> dti")
    print("Zip Code -> zip_code")
    print("State -> addr_state")
    print("Employment Length -> emp_length")

def nettoyer_donnees(df, colonnes):
    """
    Nettoie les données en gérant les valeurs manquantes et les types
    """
    df_clean = df.copy()
    
    for col in colonnes:
        if col in df_clean.columns:
            # Afficher les statistiques avant nettoyage
            n_missing = df_clean[col].isna().sum()
            if n_missing > 0:
                print(f"Colonne {col}: {n_missing} valeurs manquantes")
            
            if df_clean[col].dtype == 'object':
                df_clean[col] = df_clean[col].fillna('Non spécifié')
            else:
                median_val = df_clean[col].median()
                df_clean[col] = df_clean[col].fillna(median_val)
                print(f"Colonne {col}: valeurs manquantes remplacées par la médiane ({median_val:.2f})")
    
    return df_clean

def charger_et_preparer_donnees(accepted_path, rejected_path):
    """
    Charge et prépare les données en identifiant les colonnes comparables
    """
    try:
        # Chargement des données
        accepted = pd.read_csv(accepted_path)
        rejected = pd.read_csv(rejected_path)
        
        print(f"\nNombre initial de prêts acceptés : {len(accepted)}")
        print(f"Nombre initial de prêts refusés : {len(rejected)}")
        
        # Afficher les correspondances
        afficher_correspondances()
        
        # Mapping des colonnes comparables entre les deux datasets
        colonnes_mapping = {
            'Amount Requested': 'loan_amnt',
            'Risk_Score': 'fico_range_low',
            'Debt-To-Income Ratio': 'dti',
            'Employment Length': 'emp_length',
            'State': 'addr_state',
            'Zip Code': 'zip_code'
        }
        
        # Création d'un dataframe unifié pour l'analyse
        accepted_subset = accepted[list(colonnes_mapping.values())].copy()
        accepted_subset['status'] = 1  # 1 pour accepté
        
        rejected_subset = rejected[list(colonnes_mapping.keys())].copy()
        rejected_subset.columns = list(colonnes_mapping.values())
        rejected_subset['status'] = 0  # 0 pour rejeté
        
        # Nettoyage des données
        print("\nNettoyage des données...")
        
        # Conversion du DTI en float (suppression du %)
        rejected_subset['dti'] = rejected_subset['dti'].str.rstrip('%').astype('float')
        
        # Nettoyage des données
        colonnes_a_nettoyer = ['loan_amnt', 'fico_range_low', 'dti', 'emp_length', 'addr_state', 'zip_code']
        accepted_subset = nettoyer_donnees(accepted_subset, colonnes_a_nettoyer)
        rejected_subset = nettoyer_donnees(rejected_subset, colonnes_a_nettoyer)
        
        # Combinaison des datasets
        data_complete = pd.concat([accepted_subset, rejected_subset], axis=0)
        data_complete = data_complete.reset_index(drop=True)
        
        # Afficher les statistiques descriptives
        print("\nStatistiques descriptives des variables numériques:")
        print(data_complete[['loan_amnt', 'fico_range_low', 'dti']].describe())
        
        print("\nDistribution des variables catégorielles:")
        for col in ['emp_length', 'addr_state']:
            print(f"\n{col}:")
            print(data_complete[col].value_counts().head())
        
        return data_complete
        
    except Exception as e:
        print(f"Erreur lors du chargement des données : {str(e)}")
        raise

def analyser_correlations_numeriques(data, figures_dir):
    """
    Analyse les corrélations entre les variables numériques et le statut d'acceptation
    """
    try:
        colonnes_numeriques = ['loan_amnt', 'fico_range_low', 'dti']
        
        print("\nAnalyse des variables numériques:")
        for col in colonnes_numeriques:
            stats_acceptes = data[data['status'] == 1][col].describe()
            stats_rejetes = data[data['status'] == 0][col].describe()
            
            print(f"\n{col}:")
            print("Prêts acceptés:")
            print(stats_acceptes)
            print("\nPrêts rejetés:")
            print(stats_rejetes)
            
            # Création d'un violin plot
            plt.figure(figsize=(10, 6))
            sns.violinplot(data=data, x='status', y=col)
            plt.title(f'Distribution de {col} selon le statut du prêt')
            plt.xlabel('Statut (0=Rejeté, 1=Accepté)')
            plt.savefig(os.path.join(figures_dir, f'distribution_{col}.png'))
            plt.close()
            
    except Exception as e:
        print(f"Erreur lors de l'analyse des corrélations numériques : {str(e)}")

def analyser_variables_categoriques(data, figures_dir):
    """
    Analyse l'influence des variables catégorielles sur l'acceptation
    """
    try:
        variables_cat = ['emp_length', 'addr_state']
        
        print("\nAnalyse des variables catégorielles:")
        for var in variables_cat:
            # Statistiques détaillées
            stats = data.groupby(var).agg({
                'status': ['count', 'mean', 'std']
            }).round(3)
            
            stats.columns = ['nombre_observations', 'taux_acceptation', 'ecart_type']
            stats = stats.sort_values('taux_acceptation', ascending=False)
            
            print(f"\nStatistiques pour {var}:")
            print(stats)
            
            # Test du chi2 pour l'indépendance
            contingence = pd.crosstab(data[var], data['status'])
            from scipy.stats import chi2_contingency
            chi2, p_value, _, _ = chi2_contingency(contingence)
            print(f"\nTest du chi2 pour {var}:")
            print(f"chi2 = {chi2:.2f}, p-value = {p_value:.4f}")
            
            # Visualisation
            plt.figure(figsize=(12, 6))
            sns.barplot(data=data, x=var, y='status', errorbar=('ci', 95))
            plt.title(f"Taux d'acceptation par {var}")
            plt.xlabel(var)
            plt.ylabel("Taux d'acceptation")
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.savefig(os.path.join(figures_dir, f'taux_acceptation_{var}.png'))
            plt.close()
            
    except Exception as e:
        print(f"Erreur lors de l'analyse des variables catégorielles : {str(e)}")

def calculer_importance_variables(data, figures_dir):
    """
    Calcule l'importance relative des variables dans la décision d'acceptation
    """
    try:
        # Préparation des variables numériques
        variables_numeriques = ['loan_amnt', 'fico_range_low', 'dti']
        X = data[variables_numeriques].copy()
        y = data['status'].copy()
        
        # Standardisation des variables
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        X = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)
        
        # Régression logistique
        from sklearn.linear_model import LogisticRegression
        model = LogisticRegression(random_state=42, max_iter=1000)
        model.fit(X, y)
        
        # Importance des variables
        importance = pd.DataFrame({
            'Variable': variables_numeriques,
            'Coefficient': model.coef_[0],
            'Importance': np.abs(model.coef_[0]),
            'Odds_Ratio': np.exp(model.coef_[0])
        })
        importance = importance.sort_values('Importance', ascending=False)
        
        print("\nImportance relative des variables:")
        print(importance)
        print("\nInterprétation des Odds Ratios:")
        for _, row in importance.iterrows():
            interpretation = "augmente" if row['Coefficient'] > 0 else "diminue"
            print(f"Une augmentation d'un écart-type de {row['Variable']} {interpretation} les chances d'acceptation par un facteur de {row['Odds_Ratio']:.2f}")
        
        # Visualisations
        plt.figure(figsize=(10, 6))
        sns.barplot(data=importance, x='Importance', y='Variable')
        plt.title("Importance relative des variables")
        plt.tight_layout()
        plt.savefig(os.path.join(figures_dir, 'importance_variables.png'))
        plt.close()
        
        # Matrice de corrélation
        plt.figure(figsize=(8, 6))
        correlation_matrix = data[variables_numeriques + ['status']].corr()
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0)
        plt.title("Matrice de corrélation")
        plt.tight_layout()
        plt.savefig(os.path.join(figures_dir, 'correlation_matrix.png'))
        plt.close()
        
    except Exception as e:
        print(f"Erreur lors du calcul de l'importance des variables : {str(e)}")

def executer_analyse(config, nom_analyse):
    """
    Exécute l'analyse complète pour un jeu de données
    """
    try:
        # Création des dossiers de sortie
        figures_dir, results_dir = Config.create_output_dirs(config['OUTPUT_DIR'])
        
        # Création du fichier de résultats
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = os.path.join(results_dir, f"analyse_resultats_{timestamp}.txt")
        
        # Initialisation du suivi de progression
        tracker = ProgressTracker(total_steps=4)  # 4 étapes principales
        
        # Redirection de la sortie
        with TeeOutput(output_file):
            print(f"=== Analyse {nom_analyse} démarrée le {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ===")
            print("=" * 80)
            
            # Chargement des données
            tracker.update("Chargement et préparation des données...")
            data = charger_et_preparer_donnees(config['ACCEPTED'], config['REJECTED'])
            
            # Analyses
            tracker.update("Analyse des corrélations numériques...")
            analyser_correlations_numeriques(data, figures_dir)
            
            tracker.update("Analyse des variables catégorielles...")
            analyser_variables_categoriques(data, figures_dir)
            
            tracker.update("Calcul de l'importance des variables...")
            calculer_importance_variables(data, figures_dir)
            
            print("\nAnalyse terminée.")
            print(f"Les graphiques ont été sauvegardés dans : {figures_dir}")
            print(f"Les résultats détaillés ont été sauvegardés dans : {output_file}")
            
            temps_total = time.time() - tracker.start_time
            print(f"\nTemps total d'exécution : {temps_total:.2f} secondes")
            print("=" * 80)
        
    except Exception as e:
        print(f"Erreur lors de l'analyse {nom_analyse} : {str(e)}")
        raise

def main():
    try:
        # Analyse des extraits
        print("\n=== Démarrage de l'analyse des extraits ===")
        executer_analyse(Config.EXTRAITS, "extraits")
        
        # Analyse des données complètes
        print("\n=== Démarrage de l'analyse des données complètes ===")
        executer_analyse(Config.COMPLET, "données complètes")
        
    except Exception as e:
        print(f"Erreur lors de l'exécution du programme : {str(e)}")

if __name__ == "__main__":
    main() 