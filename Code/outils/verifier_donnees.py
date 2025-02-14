import pandas as pd
import sqlite3
import os
from typing import Dict, List, Tuple
import numpy as np

def load_csv_data(csv_path: str) -> pd.DataFrame:
    """Charge un fichier CSV et retourne un DataFrame."""
    return pd.read_csv(csv_path)

def load_sqlite_data(db_path: str, table_name: str) -> pd.DataFrame:
    """Charge une table SQLite et retourne un DataFrame."""
    with sqlite3.connect(db_path) as conn:
        query = f"SELECT * FROM {table_name}"
        return pd.read_sql_query(query, conn)

def compare_columns(csv_df: pd.DataFrame, sql_df: pd.DataFrame, file_name: str) -> None:
    """Compare les colonnes entre le CSV et la table SQLite."""
    print(f"\nComparaison des colonnes pour {file_name}:")
    print("-" * 50)
    
    csv_cols = set(csv_df.columns)
    sql_cols = set(sql_df.columns)
    
    # Colonnes présentes dans les deux
    common_cols = csv_cols.intersection(sql_cols)
    print(f"\nColonnes communes : {len(common_cols)}")
    
    # Colonnes uniquement dans le CSV
    csv_only = csv_cols - sql_cols
    if csv_only:
        print(f"\nColonnes uniquement dans le CSV : {len(csv_only)}")
        for col in sorted(csv_only):
            print(f"- {col}")
    
    # Colonnes uniquement dans SQLite
    sql_only = sql_cols - csv_cols
    if sql_only:
        print(f"\nColonnes uniquement dans SQLite : {len(sql_only)}")
        for col in sorted(sql_only):
            print(f"- {col}")

def compare_data(csv_df: pd.DataFrame, sql_df: pd.DataFrame, file_name: str) -> None:
    """Compare les données entre le CSV et la table SQLite."""
    print(f"\nComparaison des données pour {file_name}:")
    print("-" * 50)
    
    # Colonnes communes
    common_cols = list(set(csv_df.columns).intersection(set(sql_df.columns)))
    
    # Nombre de lignes
    print(f"\nNombre de lignes:")
    print(f"CSV : {len(csv_df)}")
    print(f"SQLite : {len(sql_df)}")
    
    # Comparer les valeurs pour chaque colonne
    for col in common_cols:
        print(f"\nColonne: {col}")
        
        # Vérifier si les types de données correspondent
        csv_type = csv_df[col].dtype
        sql_type = sql_df[col].dtype
        print(f"Types - CSV: {csv_type}, SQLite: {sql_type}")
        
        # Compter les valeurs nulles
        csv_nulls = csv_df[col].isna().sum()
        sql_nulls = sql_df[col].isna().sum()
        print(f"Valeurs nulles - CSV: {csv_nulls}, SQLite: {sql_nulls}")
        
        # Comparer les valeurs uniques
        csv_unique = csv_df[col].nunique()
        sql_unique = sql_df[col].nunique()
        print(f"Valeurs uniques - CSV: {csv_unique}, SQLite: {sql_unique}")
        
        # Vérifier si les valeurs correspondent
        if csv_type != object and sql_type != object:
            # Pour les colonnes numériques, comparer les statistiques
            csv_stats = csv_df[col].describe()
            sql_stats = sql_df[col].describe()
            print("\nStatistiques:")
            print(pd.concat([csv_stats, sql_stats], axis=1, keys=['CSV', 'SQLite']))
        else:
            # Pour les colonnes texte, comparer quelques exemples
            csv_sample = csv_df[col].dropna().head()
            sql_sample = sql_df[col].dropna().head()
            print("\nExemples de valeurs:")
            print("CSV:", csv_sample.tolist())
            print("SQLite:", sql_sample.tolist())

def main():
    # Obtenir le chemin absolu du script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(os.path.dirname(script_dir))
    
    # Chemins des fichiers
    db_path = os.path.join(project_root, "Data/SQLite/loandata.db")
    accepted_csv = os.path.join(project_root, "Data/CSV/extrait_accepted_2007_to_2018Q4.csv")
    rejected_csv = os.path.join(project_root, "Data/CSV/extrait_rejected_2007_to_2018Q4.csv")
    
    # Charger les données
    print("Chargement des données...")
    accepted_csv_df = load_csv_data(accepted_csv)
    rejected_csv_df = load_csv_data(rejected_csv)
    
    accepted_sql_df = load_sqlite_data(db_path, "accepted_loans")
    rejected_sql_df = load_sqlite_data(db_path, "rejected_loans")
    
    # Comparer les prêts acceptés
    print("\n=== Comparaison des prêts acceptés ===")
    compare_columns(accepted_csv_df, accepted_sql_df, "accepted_loans")
    compare_data(accepted_csv_df, accepted_sql_df, "accepted_loans")
    
    # Comparer les prêts rejetés
    print("\n=== Comparaison des prêts rejetés ===")
    compare_columns(rejected_csv_df, rejected_sql_df, "rejected_loans")
    compare_data(rejected_csv_df, rejected_sql_df, "rejected_loans")

if __name__ == "__main__":
    main() 