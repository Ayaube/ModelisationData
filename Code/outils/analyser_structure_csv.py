import pandas as pd
import os
from typing import Dict, List

def analyze_csv_structure(file_path: str) -> Dict:
    """Analyse la structure d'un fichier CSV et retourne ses métadonnées."""
    print(f"\nAnalyse de {os.path.basename(file_path)}:")
    print("-" * 50)
    
    # Lire le CSV
    df = pd.read_csv(file_path)
    
    # Analyser chaque colonne
    structure = {
        'colonnes': [],
        'nombre_lignes': len(df),
        'nombre_colonnes': len(df.columns)
    }
    
    for colonne in df.columns:
        # Détecter le type de données
        dtype = df[colonne].dtype
        non_null = df[colonne].count()
        null_percentage = (len(df) - non_null) / len(df) * 100
        unique_values = df[colonne].nunique()
        
        # Échantillon de valeurs uniques (limité à 5)
        sample_values = df[colonne].dropna().unique()[:5]
        
        col_info = {
            'nom': colonne,
            'type': str(dtype),
            'valeurs_nulles_pc': f"{null_percentage:.2f}%",
            'valeurs_uniques': unique_values,
            'exemple_valeurs': sample_values.tolist()
        }
        structure['colonnes'].append(col_info)
        
        # Afficher les informations
        print(f"\nColonne: {colonne}")
        print(f"Type: {dtype}")
        print(f"Valeurs nulles: {null_percentage:.2f}%")
        print(f"Valeurs uniques: {unique_values}")
        print(f"Exemples: {sample_values.tolist()}")
    
    return structure

def main():
    # Obtenir le chemin absolu du script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(os.path.dirname(script_dir))
    
    # Fichiers à analyser
    files = [
        "Data/CSV/extrait_accepted_2007_to_2018Q4.csv",
        "Data/CSV/extrait_rejected_2007_to_2018Q4.csv"
    ]
    
    structures = {}
    
    # Analyser chaque fichier
    for file in files:
        file_path = os.path.join(project_root, file)
        structures[file] = analyze_csv_structure(file_path)

if __name__ == "__main__":
    main() 