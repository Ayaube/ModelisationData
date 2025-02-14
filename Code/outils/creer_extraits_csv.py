import pandas as pd
import os

def extract_sample(input_file, n_rows=1000):
    # Obtenir le chemin absolu du script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    # Remonter de deux niveaux pour atteindre la racine du projet
    project_root = os.path.dirname(os.path.dirname(script_dir))
    
    # Construire le chemin complet du fichier d'entrée
    input_path = os.path.join(project_root, input_file)
    
    # Construire le nom du fichier de sortie
    dirname = os.path.dirname(input_path)
    basename = os.path.basename(input_path)
    output_file = os.path.join(dirname, f"extrait_{basename}")
    
    print(f"Lecture depuis : {input_path}")
    print(f"Écriture vers : {output_file}")
    
    # Lire les premières lignes du CSV
    df = pd.read_csv(input_path, nrows=n_rows)
    
    # Sauvegarder l'extrait
    df.to_csv(output_file, index=False)
    print(f"Extrait créé : {output_file}")

def main():
    # Chemin des fichiers (relatifs à la racine du projet)
    files = [
        "Data/CSV/accepted_2007_to_2018Q4.csv",
        "Data/CSV/rejected_2007_to_2018Q4.csv"
    ]
    
    # Traiter chaque fichier
    for file in files:
        print(f"Traitement de {file}...")
        extract_sample(file)
        print("Terminé!")

if __name__ == "__main__":
    main() 