import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, KBinsDiscretizer
from sklearn.metrics import silhouette_score
from data_manipulation.read_data import identify_variables
from pyLattice import CWDataSet

# Theoretically perfect

def hybrid_posetic_clustering(df: pd.DataFrame, 
                               stage1_k_range=(10, 50), 
                               max_variables=5,
                               binning_strategy='quantile',
                               n_bins=3,
                               auto_bin=True,
                               use_silhouette_stage1=True,
                               fuzzy_domination='BrueggemannLerche',
                               function_sep='total_separation') -> dict:
    """
    Clustering ibrido a TRE stadi con binning per velocità.
    """
    
    print("=" * 70)
    print("🚀 HYBRID THREE-STAGE CLUSTERING (con BINNING)")
    print("=" * 70)
    
    # ========================================================================
    # PREPARAZIONE DATI
    # ========================================================================
    var_types = identify_variables(df)
    quantitative_cols = var_types.get('quantitative', [])
    
    if not quantitative_cols:
        raise ValueError("Non sono state trovate variabili ordinali nel DataFrame")
    
    if len(quantitative_cols) > max_variables:
        variances = df[quantitative_cols].var()
        quantitative_cols = variances.nlargest(max_variables).index.tolist()
        print(f"\n⚠ Limitato a {max_variables} variabili più informative:")
        print(f"   {quantitative_cols}")
    
    df_ordinal = df[quantitative_cols].copy().dropna()
    n_samples = len(df_ordinal)
    
    print(f"\n📊 Dataset originale: {n_samples} righe × {len(quantitative_cols)} variabili")
    
    # ========================================================================
    # STAGE 0: BINNING
    # ========================================================================
    print("\n" + "=" * 70)
    print("STAGE 0: BINNING (Riduzione categorie ordinali)")
    print("=" * 70)
    
    if auto_bin:
        n_bins = _auto_detect_bins(df_ordinal, quantitative_cols, max_variables)
        print(f"\n🔍 Auto-detect: n_bins ottimale = {n_bins}")
    
    original_dims = [df_ordinal[col].nunique() for col in quantitative_cols]
    original_space_size = np.prod(original_dims)
    
    df_binned, bin_edges_dict, bin_labels_dict, actual_bins_per_col = _apply_binning(
        df_ordinal, 
        quantitative_cols, 
        n_bins, 
        binning_strategy
    )
    
    actual_cw_dims = [df_binned[col].nunique() for col in quantitative_cols]
    binned_space_size = np.prod(actual_cw_dims)
    
    print(f"\n📐 Spazio PoSet:")
    print(f"   PRIMA binning:  {' × '.join(map(str, original_dims))} = {original_space_size:,} nodi")
    print(f"   DOPO binning:   {' × '.join(map(str, actual_cw_dims))} = {binned_space_size:,} nodi")
    print(f"   🚀 Riduzione:   {original_space_size / binned_space_size:.1f}x più veloce!")
    
    print(f"\n✓ Binning completato con strategia '{binning_strategy}'")
    for col in quantitative_cols:
        actual_bins = actual_bins_per_col[col]
        print(f"   {col}: {original_dims[quantitative_cols.index(col)]} → {actual_bins} categorie")
        if len(bin_labels_dict[col]) <= 5:
            print(f"      Labels: {bin_labels_dict[col]}")
    
    # ========================================================================
    # STAGE 1: K-MEANS
    # ========================================================================
    print("\n" + "=" * 70)
    print("STAGE 1: K-MEANS CLUSTERING (Riduzione a profili tipici)")
    print("=" * 70)
    
    df_stage1 = df_binned.copy()
    if len(df_stage1) > 6000:
        df_stage1 = df_stage1.sample(6000, random_state=42)
        print(f"⚠ Campionati 6000 punti su {n_samples} per stage 1")
    
    df_normalized = df_stage1.copy()
    for col in quantitative_cols:
        max_val = df_binned[col].max()
        if max_val > 0:
            df_normalized[col] = df_stage1[col] / max_val
        else:
            df_normalized[col] = 0
    
    X_scaled = df_normalized.values
    
    min_k, max_k = stage1_k_range
    max_k = min(max_k, len(df_stage1) - 1)
    min_k = max(2, min(min_k, len(df_stage1) - 1))
    max_k = min(max_k, binned_space_size // 2)
    
    if use_silhouette_stage1:
        best_k = min_k
        best_score = -1
        best_model = None
        
        print(f"\n🔍 Ricerca k ottimale in range [{min_k}, {max_k}]...")
        for k in range(min_k, max_k + 1):
            kmeans = KMeans(n_clusters=k, init='k-means++', n_init=10, random_state=42)
            labels = kmeans.fit_predict(X_scaled)
            score = silhouette_score(X_scaled, labels)
            
            if score > best_score:
                best_score = score
                best_k = k
                best_model = kmeans
            
            if k % 5 == 0:
                print(f"   k={k:2d}: Silhouette={score:.3f}")
        
        print(f"\n✓ K ottimale trovato: {best_k} (Silhouette Score: {best_score:.3f})")
    else:
        best_k = max_k
        best_model = KMeans(n_clusters=best_k, init='k-means++', n_init=10, random_state=42)
        best_model.fit(X_scaled)
        best_score = silhouette_score(X_scaled, best_model.labels_)
        print(f"\n✓ K-means con k={best_k} fisso (Silhouette: {best_score:.3f})")
    
    df_full_normalized = df_binned.copy()
    for col in quantitative_cols:
        max_val = df_binned[col].max()
        if max_val > 0:
            df_full_normalized[col] = df_binned[col] / max_val
        else:
            df_full_normalized[col] = 0
    
    X_full_scaled = df_full_normalized.values
    stage1_labels = best_model.predict(X_full_scaled)
    
    centroids_normalized = best_model.cluster_centers_
    centroids_binned = centroids_normalized.copy()
    for i, col in enumerate(quantitative_cols):
        max_val = df_binned[col].max()
        centroids_binned[:, i] = centroids_normalized[:, i] * max_val
    
    centroids_binned = np.round(centroids_binned).astype(int)
    
    cluster_counts = np.bincount(stage1_labels, minlength=best_k)
    
    print(f"\n📋 Profili tipici estratti (in bin):")
    for i in range(min(5, best_k)):
        print(f"   Cluster {i}: {cluster_counts[i]:4d} clienti → {centroids_binned[i]}")
    if best_k > 5:
        print(f"   ... (altri {best_k - 5} profili)")
    
    # ========================================================================
    # STAGE 2: POSET CLUSTERING
    # ========================================================================
    print("\n" + "=" * 70)
    print("STAGE 2: POSET CLUSTERING (Diagramma di Hasse)")
    print("=" * 70)
    
    cw_dims = actual_cw_dims.copy()
    
    print(f"\n📐 Dimensioni component-wise: {cw_dims}")
    print(f"   Spazio PoSet: {binned_space_size:,} nodi possibili")
    print(f"   Profili da clusterizzare: {best_k}")
    
    # Normalizza centroidi a 0-based
    centroids_cw = centroids_binned.copy()
    for i, col in enumerate(quantitative_cols):
        min_val = df_binned[col].min()
        centroids_cw[:, i] = centroids_binned[:, i] - min_val
    
    # ========================================================================
    # 🔧 FIX: Crea array frequenze per TUTTI i punti del PoSet
    # ========================================================================
    print(f"\n🔧 Mappatura frequenze su PoSet completo...")
    
    # Genera TUTTI i punti possibili del PoSet
    from itertools import product as itertools_product
    all_points = list(itertools_product(*[range(dim) for dim in cw_dims]))
    
    # Crea dizionario: punto → frequenza
    point_to_freq = {tuple(row): 0 for row in all_points}
    
    # Assegna frequenze solo ai centroidi
    for centroid, count in zip(centroids_cw, cluster_counts):
        centroid_tuple = tuple(centroid)
        if centroid_tuple in point_to_freq:
            point_to_freq[centroid_tuple] = count
        else:
            # Se il centroide non è esattamente un punto del PoSet,
            # trova il punto più vicino
            print(f"   ⚠ Centroide {centroid_tuple} non nel PoSet, cerco punto più vicino...")
            closest_point = _find_closest_point(centroid_tuple, all_points, cw_dims)
            point_to_freq[closest_point] += count
    
    # Converti in lista ordinata (pyLattice usa ordine lessicografico)
    freq_list = [point_to_freq[point] for point in sorted(all_points)]
    
    # Verifica
    total_assigned = sum(freq_list)
    print(f"   ✓ Frequenze mappate: {total_assigned}/{n_samples} clienti")
    print(f"   ✓ Punti con freq > 0: {sum(1 for f in freq_list if f > 0)}/{len(freq_list)}")
    
    # ========================================================================
    # Crea CWDataSet con frequenze complete
    # ========================================================================
    print(f"\n🔨 Costruzione CWDataSet...")
    print(f"   Fuzzy domination: {fuzzy_domination}")
    print(f"   Separation function: {function_sep}")
    
    try:
        dataset = CWDataSet(
            cw=tuple(cw_dims),
            freq=freq_list,  # ORA SONO len(all_points) frequenze!
            fuzzy_domination_function=fuzzy_domination,
            t_norm_function='prod'
        )
    except Exception as e:
        raise ValueError(f"Errore nella creazione del CWDataSet: {str(e)}")
    
    # Clustering gerarchico
    print(f"\n⚙️ Esecuzione clustering gerarchico...")
    history_congruences, separations = dataset.classic_gerarchic_cluster(
        function_sep=function_sep
    )
    
    print(f"✓ Clustering completato: {len(history_congruences)} livelli gerarchici")
    
    final_congruence = history_congruences[-2] if len(history_congruences) > 1 else history_congruences[0]
    n_stage2_clusters = len(set(final_congruence))
    
    print(f"✓ Numero super-clusters finali (Stage 2): {n_stage2_clusters}")
    
    # ========================================================================
    # MAPPATURA: Devi mappare i centroidi originali ai loro indici nel PoSet
    # ========================================================================
    print("\n" + "=" * 70)
    print("MAPPATURA: Profili → Super-Clusters")
    print("=" * 70)
    
    # Crea mapping: centroide → indice nel PoSet → super-cluster
    centroids_to_poset_idx = {}
    for i, centroid in enumerate(centroids_cw):
        centroid_tuple = tuple(centroid)
        if centroid_tuple in all_points:
            poset_idx = sorted(all_points).index(centroid_tuple)
        else:
            closest = _find_closest_point(centroid_tuple, all_points, cw_dims)
            poset_idx = sorted(all_points).index(closest)
        centroids_to_poset_idx[i] = poset_idx
    
    # Mappa centroidi → super-clusters
    stage1_to_stage2 = {}
    for centroid_id, poset_idx in centroids_to_poset_idx.items():
        stage1_to_stage2[centroid_id] = final_congruence[poset_idx]
    
    # Mappa clienti → super-clusters
    final_clusters = np.array([stage1_to_stage2[stage1_label] 
                                for stage1_label in stage1_labels])
    
    # Profiling
    df_result = df_ordinal.copy()
    df_result['Binned_Profile'] = stage1_labels
    df_result['SuperCluster'] = final_clusters
    
    for col in quantitative_cols:
        df_result[f'{col}_binned'] = df_binned[col]
    
    group_profile = df_result.groupby('SuperCluster')[quantitative_cols].mean()
    group_profile['Count'] = df_result['SuperCluster'].value_counts().sort_index()
    
    group_profile_binned = df_result.groupby('SuperCluster')[[f'{col}_binned' for col in quantitative_cols]].mean()
    
    print(f"\n📊 Super-Clusters finali:")
    for i in range(n_stage2_clusters):
        n_customers = (final_clusters == i).sum()
        n_profiles = sum(1 for sc in stage1_to_stage2.values() if sc == i)
        print(f"   Super-Cluster {i}: {n_customers:4d} clienti da {n_profiles} profili")
    
    # ========================================================================
    # INTERPRETAZIONE
    # ========================================================================
    print("\n" + "=" * 70)
    print("INTERPRETAZIONE: Bin → Valori originali")
    print("=" * 70)
    
    interpretation = _interpret_clusters(
        group_profile_binned, 
        bin_edges_dict, 
        bin_labels_dict,
        quantitative_cols
    )
    
    for i in range(min(3, n_stage2_clusters)):
        print(f"\n   Super-Cluster {i} ({int(group_profile.loc[i, 'Count'])} clienti):")
        for col in quantitative_cols:
            bin_val = group_profile_binned.loc[i, f'{col}_binned']
            orig_val = group_profile.loc[i, col]
            interp = interpretation[i][col]
            print(f"      {col}: {orig_val:.1f} (bin {bin_val:.1f} = {interp})")
    
    # ========================================================================
    # RISULTATI
    # ========================================================================
    print("\n" + "=" * 70)
    print("✅ CLUSTERING COMPLETATO")
    print("=" * 70)
    print(f"\n📈 Performance:")
    print(f"   Stage 0 (Binning):  {original_space_size:,} → {binned_space_size:,} nodi")
    print(f"   Stage 1 (K-Means):  {n_samples:,} → {best_k} profili")
    print(f"   Stage 2 (PoSet):    {best_k} → {n_stage2_clusters} super-clusters")
    
    return {
        'binning_strategy': binning_strategy,
        'n_bins': n_bins,
        'actual_bins_per_col': actual_bins_per_col,
        'bin_edges': bin_edges_dict,
        'bin_labels': bin_labels_dict,
        'df_binned': df_binned,
        'space_reduction': original_space_size / binned_space_size,
        'stage1_k': best_k,
        'stage1_score': best_score,
        'stage1_labels': stage1_labels,
        'stage1_centroids_binned': centroids_binned,
        'stage1_counts': cluster_counts,
        'stage2_n_clusters': n_stage2_clusters,
        'stage2_congruences': history_congruences,
        'stage2_separations': separations,
        'stage2_dataset': dataset,
        'stage1_to_stage2_map': stage1_to_stage2,
        'final_clusters': final_clusters,
        'group_profile': group_profile,
        'group_profile_binned': group_profile_binned,
        'interpretation': interpretation,
        'df_result': df_result,
        'ordinal_vars': quantitative_cols,
        'cw_dims': cw_dims,
        'n_samples': n_samples
    }


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def _find_closest_point(centroid: tuple, all_points: list, cw_dims: list) -> tuple:
    """Trova il punto del PoSet più vicino al centroide (distanza Manhattan)."""
    min_dist = float('inf')
    closest = None
    
    for point in all_points:
        dist = sum(abs(c - p) for c, p in zip(centroid, point))
        if dist < min_dist:
            min_dist = dist
            closest = point
    
    return closest


def _auto_detect_bins(df: pd.DataFrame, cols: list, max_vars: int) -> int:
    """Auto-detect numero ottimale di bin."""
    n_vars = len(cols)
    target_space = 1000
    max_bins = int(target_space ** (1 / n_vars))
    optimal_bins = max(3, min(5, max_bins))
    
    print(f"\n🔍 Auto-detect bins:")
    print(f"   Variabili: {n_vars}")
    print(f"   Target space: < {target_space} nodi")
    print(f"   Max bins teorico: {max_bins}")
    print(f"   Bins scelto: {optimal_bins} (range 3-5)")
    
    return optimal_bins


def _apply_binning(df: pd.DataFrame, cols: list, n_bins: int, strategy: str):
    """Applica binning con gestione robusta."""
    df_binned = df.copy()
    bin_edges_dict = {}
    bin_labels_dict = {}
    actual_bins_per_col = {}
    
    for col in cols:
        values = df[[col]].values
        n_unique = df[col].nunique()
        col_n_bins = min(n_bins, n_unique)
        
        if col_n_bins < 2:
            df_binned[col] = 0
            bin_edges_dict[col] = [df[col].min(), df[col].max()]
            bin_labels_dict[col] = [f"[{df[col].min():.1f}, {df[col].max():.1f}]"]
            actual_bins_per_col[col] = 1
            print(f"   ⚠ {col}: solo {n_unique} valore(i) unico(i) → 1 bin")
            continue
        
        try:
            discretizer = KBinsDiscretizer(
                n_bins=col_n_bins, 
                encode='ordinal',
                strategy=strategy,
                subsample=None
            )
            
            binned_values = discretizer.fit_transform(values).astype(int).flatten()
            df_binned[col] = binned_values
            
            bin_edges = discretizer.bin_edges_[0]
            bin_edges_dict[col] = bin_edges
            
            actual_n_bins = len(bin_edges) - 1
            actual_bins_per_col[col] = actual_n_bins
            
            labels = []
            for i in range(actual_n_bins):
                left = bin_edges[i]
                right = bin_edges[i + 1]
                if i == actual_n_bins - 1:
                    labels.append(f"[{left:.1f}, {right:.1f}]")
                else:
                    labels.append(f"[{left:.1f}, {right:.1f})")
            bin_labels_dict[col] = labels
            
        except Exception as e:
            print(f"   ⚠ {col}: KBinsDiscretizer fallito, uso binning manuale")
            min_val = df[col].min()
            max_val = df[col].max()
            
            if min_val == max_val:
                df_binned[col] = 0
                bin_edges_dict[col] = [min_val, max_val]
                bin_labels_dict[col] = [f"[{min_val:.1f}]"]
                actual_bins_per_col[col] = 1
            else:
                edges = np.linspace(min_val, max_val, col_n_bins + 1)
                df_binned[col] = pd.cut(df[col], bins=edges, labels=False, include_lowest=True).astype(int)
                bin_edges_dict[col] = edges
                actual_bins_per_col[col] = col_n_bins
                
                labels = []
                for i in range(col_n_bins):
                    labels.append(f"[{edges[i]:.1f}, {edges[i+1]:.1f}]")
                bin_labels_dict[col] = labels
    
    return df_binned, bin_edges_dict, bin_labels_dict, actual_bins_per_col


def _interpret_clusters(group_profile_binned: pd.DataFrame, 
                        bin_edges_dict: dict, 
                        bin_labels_dict: dict,
                        cols: list) -> dict:
    """Interpreta cluster binned."""
    interpretation = {}
    
    for cluster_id in group_profile_binned.index:
        cluster_interp = {}
        for col in cols:
            bin_val = group_profile_binned.loc[cluster_id, f'{col}_binned']
            bin_idx = int(round(bin_val))
            bin_idx = max(0, min(bin_idx, len(bin_labels_dict[col]) - 1))
            cluster_interp[col] = bin_labels_dict[col][bin_idx]
        interpretation[cluster_id] = cluster_interp
    
    return interpretation


def auto_hybrid_posetic(df: pd.DataFrame, 
                        stage1_k_range=(10, 50),
                        n_bins=3,
                        auto_bin=True) -> pd.DataFrame:
    """Versione semplificata."""
    result = hybrid_posetic_clustering(
        df, 
        stage1_k_range=stage1_k_range,
        n_bins=n_bins,
        auto_bin=auto_bin
    )
    return result['group_profile']