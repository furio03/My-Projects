import pandas as pd
import numpy as np
import os
import sys
import traceback

# Configurazione path per importare i tuoi moduli
current_dir = os.path.dirname(os.path.abspath(__file__))
repo_root = os.path.abspath(os.path.join(current_dir, ".."))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

# Import delle funzioni aggiornate (assumendo che siano nel file statistics_functions/posetic_clustering.py)
from statistics_functions.posetic_clustering import hybrid_posetic_clustering
from data_manipulation.read_data import read

def test_three_stage_hybrid_clustering():
    print("="*80)
    print("🧪 TEST: HYBRID 3-STAGE (BINNING + K-MEANS + POSET)")
    print("="*80)

    # 1. Caricamento dati di esempio
    # Se non hai il file, generiamo dei dati sintetici per il test
    path = os.path.join(repo_root, 'data', 'test_data.csv')
    
    if os.path.exists(path):
        df = read(path)
    else:
        print("⚠️ File non trovato, genero dati sintetici per il test...")
        np.random.seed(42)
        data = {
            'brand_identity': np.random.randint(0, 100, 1000),
            'loyalty_metric': np.random.randint(0, 100, 1000),
            'price_perception': np.random.randint(0, 50, 1000),
            'engagement_volume': np.random.randint(0, 10, 1000)
        }
        df = pd.DataFrame(data)

    # 2. Esecuzione del clustering con Binning Strategico
    try:
        results = hybrid_posetic_clustering(
            df, 
            stage1_k_range=(5, 15), 
            max_variables=4,
            n_bins=3,             # Riduce ogni variabile a 3 livelli (Low, Mid, High)
            auto_bin=True,        # Se True, calcola n_bins per stare sotto i 1000 nodi totali
            binning_strategy='quantile', # Divide i gruppi in parti uguali (20% - 20%...)
            fuzzy_domination='BrueggemannLerche'
        )

        # 3. Verifica della riduzione dello spazio
        print("\n" + "-"*30)
        print("🚀 VERIFICA PERFORMANCE")
        print(f"Riduzione spazio nodi: {results['space_reduction']:.1f}x più piccolo")
        print(f"Nodi finali nel reticolo: {results['cw_dims']}")
        
        # 4. Analisi dell'interpretazione
        print("\n" + "-"*30)
        print("📖 INTERPRETAZIONE LOGICA DEI CLUSTER")
        for sc_id, interp in results['interpretation'].items():
            if sc_id < 3: # Mostriamo solo i primi 3 per brevità
                print(f"\nSuperCluster {sc_id}:")
                for var, range_str in interp.items():
                    print(f"   - {var}: {range_str}")

        print("\n✅ TEST COMPLETATO CON SUCCESSO")
        
    except Exception as e:
        print(f"\n❌ ERRORE DURANTE IL TEST")
        traceback.print_exc()

if __name__ == "__main__":
    test_three_stage_hybrid_clustering()