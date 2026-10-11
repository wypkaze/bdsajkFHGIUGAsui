"""
Clustering and anchor selection functions.

This module provides functions for K-Means clustering, optimal k selection,
and anchor user selection based on clustering results.
"""

from typing import List, Tuple, Dict, Any
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score


def find_optimal_k(
    emb: np.ndarray,
    anchor_candidate: List[Dict[str, Any]],
    k_range: range = range(10, 101, 10),
    min_history_threshold: int = 10
) -> Tuple[int, Dict[str, List]]:
    """
    Find optimal k using Silhouette Score with anchor quality consideration.
    
    Args:
        emb: User embedding matrix [num_users, embedding_dim]
        anchor_candidate: List of candidate user dictionaries
        k_range: Range of k values to test
        min_history_threshold: Minimum history length threshold for valid anchors
    
    Returns:
        Tuple of (optimal_k, results_dict)
        - optimal_k: Optimal number of clusters
        - results_dict: Dictionary containing 'k', 'silhouette', 'valid_anchors' lists
    """
    results = {
        'k': [],
        'silhouette': [],
        'valid_anchors': []
    }
    
    print("Evaluating different k values using Silhouette Score...")
    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=0, max_iter=3000, n_init=10)
        labels = kmeans.fit_predict(emb)
        
        sil_score = silhouette_score(emb, labels)
        
        valid_count = 0
        for i in range(k):
            cluster_indices = np.where(labels == i)[0]
            max_len = max(
                [len(anchor_candidate[idx].get('profile', []))
                 for idx in cluster_indices],
                default=0
            )
            if max_len > min_history_threshold:
                valid_count += 1
        
        results['k'].append(k)
        results['silhouette'].append(sil_score)
        results['valid_anchors'].append(valid_count)
        
        print(f"k={k:3d}: Silhouette={sil_score:.4f}, Valid={valid_count}/{k}")
    
    # Combine Silhouette Score and valid anchor ratio
    silhouette_norm = np.array(results['silhouette']) / max(results['silhouette'])
    #valid_ratio = np.array(results['valid_anchors']) / np.array(results['k'])
    
    combined_score = silhouette_norm  #0.3 * valid_ratio
    optimal_idx = int(np.argmax(combined_score))
    optimal_k = results['k'][optimal_idx]
    
    print(f"\n{'='*60}")
    print(f"Optimal k: {optimal_k} (Silhouette={results['silhouette'][optimal_idx]:.4f})")
    print(f"{'='*60}")
    
    return optimal_k, results


def plot_k_selection_results(
    results: Dict[str, List],
    optimal_k: int,
    save_path: str = 'k_selection_analysis.png'
) -> None:
    """
    Plot the k selection analysis results.
    
    Args:
        results: Dictionary containing 'k', 'silhouette', 'valid_anchors' lists
        optimal_k: The optimal k value selected
        save_path: Path to save the plot
    """
    valid_ratio = np.array(results['valid_anchors']) / np.array(results['k'])
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    axes[0].plot(results['k'], results['silhouette'], 'bo-', linewidth=2)
    axes[0].set_xlabel('Number of Clusters (k)')
    axes[0].set_ylabel('Silhouette Score')
    axes[0].set_title('Silhouette Score vs k (higher is better)')
    axes[0].axvline(optimal_k, color='r', linestyle='--', label=f'Optimal k={optimal_k}')
    axes[0].legend()
    axes[0].grid(True)
    
    axes[1].plot(results['k'], valid_ratio, 'go-', linewidth=2)
    axes[1].set_xlabel('Number of Clusters (k)')
    axes[1].set_ylabel('Valid Anchor Ratio')
    axes[1].set_title('Valid Anchor Ratio vs k')
    axes[1].axvline(optimal_k, color='r', linestyle='--')
    axes[1].grid(True)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.show()


def select_anchor_users(
    labels: np.ndarray,
    k: int,
    anchor_candidate: List[Dict[str, Any]],
    min_history: int = 10
) -> List[int]:
    """
    Select anchor users from clusters.
    
    For each cluster, selects the user with the most history records.
    Only selects users with history length greater than min_history.
    
    Args:
        labels: Cluster labels for each user
        k: Number of clusters
        anchor_candidate: List of candidate user dictionaries
        min_history: Minimum history length threshold
    
    Returns:
        List of selected anchor user indices
    """
    selected_indices = []
    
    for i in range(k):
        cluster_indices = np.where(labels == i)[0]
        
        if len(cluster_indices) == 0:
            continue
        
        max_len = 0
        selected_index = None
        
        for idx in cluster_indices:
            if idx < len(anchor_candidate):
                history_len = len(anchor_candidate[idx].get('profile', []))
                if history_len > max_len:
                    max_len = history_len
                    selected_index = int(idx)
        
        if selected_index is not None:
            print(f"Cluster {i:3d}: {len(cluster_indices):4d} users, max history = {max_len:3d}")
            
            if max_len > min_history:
                selected_indices.append(selected_index)
    
    return selected_indices


def perform_kmeans_clustering(
    emb: np.ndarray,
    k: int,
    random_state: int = 0,
    max_iter: int = 3000,
    n_init: int = 10
) -> Tuple[KMeans, np.ndarray]:
    """
    Perform K-Means clustering on embeddings.
    
    Args:
        emb: Embedding matrix [num_samples, embedding_dim]
        k: Number of clusters
        random_state: Random seed for reproducibility
        max_iter: Maximum number of iterations
        n_init: Number of initializations to try
    
    Returns:
        Tuple of (kmeans_model, labels)
    """
    kmeans = KMeans(
        n_clusters=k,
        random_state=random_state,
        max_iter=max_iter,
        n_init=n_init
    )
    kmeans.fit(emb)
    return kmeans, kmeans.labels_
