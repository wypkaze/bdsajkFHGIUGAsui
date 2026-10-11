#!/usr/bin/env python3
"""
Anchor User Selection Script

Selects top-N users with most history from each cluster.
Default: K=16 clusters, 4 anchors per cluster.

Usage:
    python anchor_select.py --task_name movie_tagging --k 16 --anchors_per_cluster 4
    
    # Skip encoding if embeddings exist
    python anchor_select.py --task_name movie_tagging --skip_encoding
"""

import sys
import os

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))  # PAPPO/
sys.path.insert(0, _ROOT)
sys.path.insert(0, os.path.join(_ROOT, 'src'))

import torch
import json
import argparse
import numpy as np
from pathlib import Path

from selection import perform_kmeans_clustering


def load_candidate_users(candidate_path: str) -> list:
    """Load candidate users from JSON file."""
    print(f"Loading candidate users from {candidate_path}...")
    with open(candidate_path, 'r', encoding='utf-8') as f:
        anchor_candidate = json.load(f)
    print(f"Loaded {len(anchor_candidate)} candidate users")
    return anchor_candidate


def load_prompt_template(candidate_path: str) -> dict:
    """Load prompt template from JSON file."""
    prompt_path = Path(candidate_path).parent.parent / 'prompt' / 'prompt.json'
    if not prompt_path.exists():
        prompt_path = Path(_ROOT) / 'prompt' / 'prompt.json'
    
    print(f"Loading prompt template from {prompt_path}...")
    with open(prompt_path, 'r', encoding='utf-8') as f:
        prompt_template = json.load(f)
    return prompt_template


def load_deberta_model(model_path: str):
    """Load DeBERTa model from local path."""
    from transformers import DebertaV2Tokenizer, DebertaV2Model
    from selection import get_first_k_tokens, split_batch
    from tqdm import tqdm
    
    print(f"Loading DeBERTa model from {model_path}...")
    model_path_obj = Path(model_path)
    
    if model_path_obj.exists() and model_path_obj.is_dir():
        print(f"Detected local model directory: {model_path}")
        tokenizer = DebertaV2Tokenizer.from_pretrained(
            model_path, local_files_only=True
        )
        model = DebertaV2Model.from_pretrained(
            model_path, local_files_only=True
        ).cuda()
    else:
        print(f"Loading from HuggingFace: {model_path}")
        tokenizer = DebertaV2Tokenizer.from_pretrained(model_path)
        model = DebertaV2Model.from_pretrained(model_path).cuda()
    
    model.eval()
    print("Model loaded successfully")
    return tokenizer, model, get_first_k_tokens, split_batch, tqdm


def generate_embeddings(anchor_candidate: list, prompt_template: dict, tokenizer, model, task_name: str, get_first_k_tokens, split_batch, tqdm):
    """Generate user embeddings from history."""
    BATCH_SIZE = 32
    print("\nGenerating user embeddings...")
    all_user_emb = []
    
    for user in tqdm(anchor_candidate, desc="Processing users"):
        history_embeddings_list = []
        visible_history_list = user.get('profile', [])
        
        if not visible_history_list:
            empty_emb = torch.zeros(1, 1024)
            all_user_emb.append(empty_emb)
            continue
        
        # Truncate history
        for p in visible_history_list:
            for key, value in p.items():
                if value is not None and isinstance(value, str):
                    p[key] = get_first_k_tokens(value, 368)
        
        # Format history with prompt template
        try:
            user_nl_history_list = [
                prompt_template[task_name]['retrieval_history'].format(**p)
                for p in visible_history_list
            ]
        except KeyError:
            user_nl_history_list = [p.get('text', str(p)) for p in visible_history_list]
        
        # Batch processing
        user_nl_history_list_batched = split_batch(user_nl_history_list, BATCH_SIZE)
        
        for batch in user_nl_history_list_batched:
            if not batch:
                continue
            
            with torch.no_grad():
                inputs = tokenizer(
                    batch,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=512
                ).to(model.device)
                
                outputs = model(**inputs)
                last_hidden_states = outputs.last_hidden_state
                
                # Mean pooling
                attention_mask = inputs['attention_mask']
                attention_mask = attention_mask.unsqueeze(-1).expand(last_hidden_states.size()).float()
                masked_hidden_states = last_hidden_states * attention_mask
                summed = torch.sum(masked_hidden_states, 1)
                count = torch.clamp(attention_mask.sum(1), min=1e-9)
                mean_pooled = summed / count
            
            history_embeddings_list.append(mean_pooled.cpu())
        
        # Aggregate user embedding
        if history_embeddings_list:
            history_embedding_concat = torch.cat(history_embeddings_list, dim=0)
            user_embedding = history_embedding_concat.mean(dim=0, keepdim=True)
        else:
            user_embedding = torch.zeros(1, 1024)
        
        all_user_emb.append(user_embedding)
    
    all_user_emb = torch.cat(all_user_emb, dim=0)
    print(f"User embeddings shape: {all_user_emb.size()}")
    return all_user_emb


def perform_clustering(emb: np.ndarray, k: int):
    """Perform K-Means clustering."""
    print(f"\nPerforming K-Means clustering with k={k}...")
    _, labels = perform_kmeans_clustering(
        emb, k,
        random_state=0,
        max_iter=3000,
        n_init=10
    )
    return labels


def select_anchors_by_history(labels: np.ndarray, k: int, anchor_candidate: list, anchors_per_cluster: int = 4):
    """Select top-N users with most history from each cluster."""
    clusters_info = {}
    
    for i in range(k):
        cluster_indices = np.where(labels == i)[0]
        if len(cluster_indices) == 0:
            continue
        
        # Sort by history count (descending)
        sorted_indices = sorted(
            cluster_indices,
            key=lambda idx: len(anchor_candidate[idx].get('profile', [])),
            reverse=True
        )
        
        # Select top anchors_per_cluster
        selected_indices = sorted_indices[:min(anchors_per_cluster, len(sorted_indices))]
        
        clusters_info[i] = {
            'anchor_indices': [int(idx) for idx in selected_indices],
            'anchor_ids': [anchor_candidate[idx]['user_id'] for idx in selected_indices]
        }
    
    return clusters_info


def compute_cluster_centers(emb: np.ndarray, labels: np.ndarray, k: int):
    """Compute center for each cluster."""
    centers = {}
    for i in range(k):
        cluster_mask = labels == i
        cluster_indices = np.where(cluster_mask)[0]
        if len(cluster_indices) > 0:
            centers[i] = emb[cluster_indices].mean(axis=0).tolist()
    return centers


def save_results(clusters_info: dict, cluster_centers: dict, anchor_candidate: list,
                output_dir: str, task_name: str, k: int, anchors_per_cluster: int):
    """Save anchor user IDs and cluster information."""
    # Build output structure
    output_data = {
        'task_name': task_name,
        'mode': 'history',
        'k': k,
        'anchors_per_cluster': anchors_per_cluster,
        'clusters': {},
        'all_anchor_ids': [],
        'total_anchors': 0
    }
    
    for cluster_id, info in clusters_info.items():
        output_data['clusters'][str(cluster_id)] = {
            'center': cluster_centers.get(cluster_id, []),
            'anchor_ids': info['anchor_ids'],
            'anchor_indices': info['anchor_indices']
        }
        output_data['all_anchor_ids'].extend(info['anchor_ids'])
    
    output_data['total_anchors'] = len(output_data['all_anchor_ids'])
    
    # Save to file
    anchor_dir = Path(output_dir) / 'anchors_by_his' / task_name / f'k{k}'
    anchor_dir.mkdir(parents=True, exist_ok=True)
    
    anchor_file = anchor_dir / 'anchor_user_idx.json'
    with open(anchor_file, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2)
    
    print(f"\nSaved {output_data['total_anchors']} anchors to {anchor_file}")
    print(f"Clusters: {len(clusters_info)}, Anchors per cluster: {anchors_per_cluster}")


def save_embeddings(all_user_emb: torch.Tensor, output_dir: str, task_name: str):
    """Save user embeddings."""
    emb_dir = Path(output_dir) / 'emb' / task_name
    emb_dir.mkdir(parents=True, exist_ok=True)
    
    emb_file = emb_dir / 'user_history_emb.pt'
    torch.save(all_user_emb, emb_file)
    print(f"Saved embeddings to {emb_file}")


def main():
    parser = argparse.ArgumentParser("Anchor User Selection")
    parser.add_argument(
        "--task_name",
        default="movie_tagging",
        type=str,
        help="Name of the task"
    )
    parser.add_argument(
        "--candidate_path",
        default=None,
        type=str,
        help="Path to candidate user JSON file"
    )
    parser.add_argument(
        "--k",
        default=16,
        type=int,
        help="Number of clusters (default: 16)"
    )
    parser.add_argument(
        "--anchors_per_cluster",
        default=4,
        type=int,
        help="Number of anchors to select per cluster (default: 4)"
    )
    parser.add_argument(
        "--model_path",
        default=os.path.join(_ROOT, "deberta"),
        type=str,
        help="Path to DeBERTa model"
    )
    parser.add_argument(
        "--output_dir",
        default=os.path.join(_ROOT, "anchors"),
        type=str,
        help="Output directory"
    )
    parser.add_argument(
        "--skip_encoding",
        action="store_true",
        help="Skip encoding, load pre-computed embeddings"
    )
    
    args = parser.parse_args()
    
    # Build paths
    if args.candidate_path is None:
        args.candidate_path = os.path.join(_ROOT, "data", args.task_name, "user_anchor_candidate.json")
    
    # Load candidate users
    anchor_candidate = load_candidate_users(args.candidate_path)
    
    # Generate or load embeddings
    if args.skip_encoding:
        emb_path = Path(args.output_dir) / 'emb' / args.task_name / 'user_history_emb.pt'
        print(f"\nLoading pre-computed embeddings from {emb_path}...")
        all_user_emb = torch.load(emb_path)
        emb = all_user_emb.numpy()
        print(f"Embeddings shape: {emb.shape}")
    else:
        # Load prompt template
        prompt_template = load_prompt_template(args.candidate_path)
        
        # Load DeBERTa model
        tokenizer, model, get_first_k_tokens, split_batch, tqdm = load_deberta_model(args.model_path)
        
        # Generate embeddings
        all_user_emb = generate_embeddings(anchor_candidate, prompt_template, tokenizer, model, args.task_name, get_first_k_tokens, split_batch, tqdm)
        emb = all_user_emb.numpy()
        
        # Save embeddings
        save_embeddings(all_user_emb, args.output_dir, args.task_name)
    
    # Perform clustering
    labels = perform_clustering(emb, args.k)
    
    # Select anchors
    print(f"\nSelecting {args.anchors_per_cluster} anchors per cluster...")
    clusters_info = select_anchors_by_history(labels, args.k, anchor_candidate, args.anchors_per_cluster)
    
    # Compute cluster centers
    cluster_centers = compute_cluster_centers(emb, labels, args.k)
    
    # Save results
    save_results(clusters_info, cluster_centers, anchor_candidate,
                args.output_dir, args.task_name, args.k, args.anchors_per_cluster)
    
    print('\nDone!')


if __name__ == '__main__':
    main()
