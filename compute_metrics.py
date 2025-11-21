import os
import cv2
import json
import numpy as np
import torch
import shutil
from pathlib import Path
import argparse

# TorchReID imports
from torchreid.tools.feature_extractor import KPRFeatureExtractor
from torchreid.scripts.builder import build_config
from torchreid.metrics.distance import compute_distance_matrix_using_bp_features
from torchreid.metrics.rank import eval_market1501
from load_market1501 import load_market1501_for_kpr, process_keypoints     

# Paths
root = "D:/Computer_Science_Uottawa/semester/other/Project_code/keypoint_promptable_reidentification/torchreid/data/datasets/Market-1501-v15.09.15"
ann_root = "D:/Computer_Science_Uottawa/semester/other/Project_code/keypoint_promptable_reidentification/torchreid/data/datasets/Market-1501-v15.09.15/external_annotation/pifpaf_keypoints_pifpaf_maskrcnn_filtering"
config_path = "D:/Computer_Science_Uottawa/semester/other/Project_code/keypoint_promptable_reidentification/configs/kpr/solider/kpr_market_test.yaml"
filtered_output_path = "D:/Computer_Science_Uottawa/semester/other/Project_code/keypoint_promptable_reidentification/filtered_keypoints"
results_dir = "evaluation_results"
cache_file = "D:/Computer_Science_Uottawa/semester/other/Project_code/keypoint_promptable_reidentification/market1501_cache.pkl"


# Build config
args = {
    'config_file': config_path,
    'sources': None,
    'targets': None,
    'transforms': None,
    'root': root,
    'save_dir': "",
    'opts': [],
    'job_id': None,
    'inference_enabled': False
}


args = argparse.Namespace(**args)
kpr_cfg = build_config(args=args, config_path=args.config_file, display_diff=False)


def print_results(results, max_rank=10):
    """Print evaluation results in a formatted way."""
    print("\n" + "="*70)
    print("EVALUATION RESULTS")
    print("="*70)
    print(f"Mean Average Precision (mAP): {results['mAP']:.2%}")
    print(f"\nCumulative Matching Characteristics (CMC):")
    for r in [1, 5, 10, 20]:
        if r <= len(results['CMC']):
            print(f"  Rank-{r:2d}: {results['CMC'][r-1]:.2%}")
    print(f"\nNumber of valid queries: {results['num_valid_queries']}")
    print("="*70 + "\n")


def save_results(results, save_dir):
    """Save evaluation results to JSON file."""
    os.makedirs(save_dir, exist_ok=True)
    
    # Convert numpy arrays to lists for JSON serialization
    results_serializable = {
        'mAP': float(results['mAP']),
        'CMC': results['CMC'].tolist(),
        'all_AP': results['all_AP'].tolist(),
        'num_valid_queries': int(results['num_valid_queries'])
    }
    
    output_path = os.path.join(save_dir, 'evaluation_results.json')
    with open(output_path, 'w') as f:
        json.dump(results_serializable, f, indent=2)
    
    print(f"Results saved to: {output_path}")

  
def extract_features_in_batches(extractor, samples, batch_size=32):
  all_embeddings = []
  all_visibility = []
  all_masks = []
  all_samples = []

  total_batches = len(samples)//batch_size
  with torch.no_grad():
    for i in range(0, len(samples), batch_size):
        torch.cuda.empty_cache() 

        batch = samples[i:i + batch_size]
        updated_batch, embeddings, visibility_scores, masks = extractor(batch)

        all_samples.extend(updated_batch)
        all_embeddings.append(embeddings.cpu())
        all_visibility.append(visibility_scores.cpu())
        all_masks.append(masks.cpu())

        current_batch = i//batch_size
        print("completed batch: ", current_batch,"/", total_batches)

  # Concatenate along batch dimension
  all_embeddings = torch.cat(all_embeddings, dim=0)
  all_visibility = torch.cat(all_visibility, dim=0)
  all_masks = torch.cat(all_masks, dim=0)

  return all_samples, all_embeddings, all_visibility, all_masks 
def main():
    """Main evaluation pipeline."""
    
    # Create results directory
    os.makedirs(results_dir, exist_ok=True)
    
    # Step 1: Load dataset
    print("="*70)
    print("STEP 1: Loading Dataset")
    print("="*70)
    
    (query_samples, gallery_samples, 
     query_pids, gallery_pids,
     query_camids, gallery_camids,
     query_paths, gallery_paths) = load_market1501_for_kpr(
        root, 
        ann_root,
        filtered_output_path,
        save_filtered=True,
        cache_file=cache_file
    )
    
    # Step 2: Initialize model
    print("\n" + "="*70)
    print("STEP 2: Initializing KPR Model")
    print("="*70)
    extractor = KPRFeatureExtractor(kpr_cfg)
    print("Model loaded successfully")
    
    # Step 3: Extract features
    print("\n" + "="*70)
    print("STEP 3: Extracting Features")
    print("="*70)
    
    with torch.no_grad():
      print("\nProcessing query samples...")
      query_samples, q_emb, q_vis, q_masks = extract_features_in_batches(extractor, query_samples, batch_size=1)
      print(f"Query embeddings shape: {q_emb.shape}")

      print("\nProcessing gallery samples...")
      gallery_samples, g_emb, g_vis, g_masks = extract_features_in_batches(extractor, gallery_samples, batch_size=1)
      print(f"Gallery embeddings shape: {g_emb.shape}")

    
    # Step 4: Compute distance matrix
    print("\n" + "="*70)
    print("STEP 4: Computing Distance Matrix")
    print("="*70)
    
    distance_matrix, body_parts_distmat = compute_distance_matrix_using_bp_features(
        q_emb,
        g_emb,
        q_vis,
        g_vis,
        use_gpu=kpr_cfg.use_gpu,
        use_logger=False
    )
    
    # Convert to numpy and scale to [0, 1]
    distmat = distance_matrix.cpu().detach().numpy() / 2
    print(f"Distance matrix shape: {distmat.shape}")
    print(f"Distance range: [{distmat.min():.3f}, {distmat.max():.3f}]")
    print(f"Mean distance: {distmat.mean():.3f}")
    
    # Step 5: Evaluate
    print("\n" + "="*70)
    print("STEP 5: Computing Evaluation Metrics")
    print("="*70)
    
    results = eval_market1501(
        distmat,
        query_pids,
        gallery_pids,
        query_camids,
        gallery_camids,
        50,
        query_samples, 
        gallery_samples
    )
    
    # Print results
    print_results(results)
    
    # Save results
    save_results(results, results_dir)
    
    print("\n" + "="*70)
    print("EVALUATION COMPLETE!")
    print("="*70)
    print(f"Results saved to: {results_dir}")
    print(f"Filtered annotations saved to: {filtered_output_path}")


if __name__ == "__main__":
    main()