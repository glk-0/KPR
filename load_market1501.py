import os
import cv2
import json
import numpy as np
import torch
from torchreid.tools.feature_extractor import KPRFeatureExtractor
from torchreid.scripts.builder import build_config
from torchreid.metrics.distance import compute_distance_matrix_using_bp_features
from torchreid.utils.visualization.display_kpr_samples import display_kpr_reid_samples_grid, display_distance_matrix
from pathlib import Path
import shutil
from tqdm import tqdm
import pickle


# Paths
root = "D:/Computer_Science_Uottawa/semester/other/Project_code/keypoint_promptable_reidentification/torchreid/data/datasets/Market-1501-v15.09.15"
ann_root = "D:/Computer_Science_Uottawa/semester/other/Project_code/keypoint_promptable_reidentification/torchreid/data/datasets/Market-1501-v15.09.15/external_annotation/pifpaf_keypoints_pifpaf_maskrcnn_filtering"
config_path = "D:/Computer_Science_Uottawa/semester/other/Project_code/keypoint_promptable_reidentification/configs/kpr/solider/kpr_market_test.yaml"
filtered_output_path = "D:/Computer_Science_Uottawa/semester/other/Project_code/keypoint_promptable_reidentification/filtered_keypoints"
cache_file = "D:/Computer_Science_Uottawa/semester/other/Project_code/keypoint_promptable_reidentification/market1501_cache.pkl"
#colab_dataset = "/content/drive/MyDrive/keypoint_promptable_reidentification/torchreid/data/datasets/Market-1501-v15.09.15"
# Build config
import argparse
args = argparse.Namespace(
    config_file=config_path,
    sources=None,
    targets=None,
    transforms=None,
    root=root,
    save_dir="",           
    opts=[],                
    job_id=None,
    inference_enabled=False
)
kpr_cfg = build_config(args=args, config_path=args.config_file, display_diff=True)


def process_keypoints(kps, img_shape):
    """
    Process keypoints: handle both normalized and pixel coordinates.
    The KPR model expects pixel coordinates, not normalized.
    
    Args:
        kps: np.array of shape (N, 3), each row is (x, y, c)
        img_shape: (H, W, C)
    
    Returns:
        np.array of shape (N, 3) in pixel coordinates
    """
    if kps.size == 0:
        return kps
        
    H, W = img_shape[:2]
    kps_processed = kps.copy()
    
    # Detect if keypoints are normalized (values in [0, 1] range)
    max_x = np.max(np.abs(kps[:, 0]))
    max_y = np.max(np.abs(kps[:, 1]))
    
    # If clearly normalized (max values around 1.0 or less)
    if max_x <= 1.0 and max_y <= 1.0:
        # Convert from normalized [0,1] to pixel coordinates
        kps_processed[:, 0] = kps_processed[:, 0] * W
        kps_processed[:, 1] = kps_processed[:, 1] * H
    
    # Clip to valid bounds and set confidence=0 for out-of-bounds points
    valid_x = (kps_processed[:, 0] >= 0) & (kps_processed[:, 0] < W)
    valid_y = (kps_processed[:, 1] >= 0) & (kps_processed[:, 1] < H)
    valid = valid_x & valid_y
    
    # Clip coordinates
    kps_processed[:, 0] = np.clip(kps_processed[:, 0], 0, W - 1)
    kps_processed[:, 1] = np.clip(kps_processed[:, 1], 0, H - 1)
    
    # Mask invalid keypoints by setting confidence to 0
    kps_processed[~valid, 2] = 0.0
    
    return kps_processed

def extract_pid_camid_from_filename(img_name):
    """
    Extract person ID and camera ID from Market-1501 filename.
    Format: 0001_c1s1_000000_00.jpg -> pid=1, camid=1
    
    Args:
        img_name: Image filename (e.g., "0001_c1s1_000000_00.jpg")
    
    Returns:
        tuple: (person_id, camera_id)
    """
    basename = os.path.basename(img_name).replace("_keypoints.json", "")
    parts = basename.split('_')
    pid = int(parts[0])
    camid = int(parts[1][1])  # 'c1' -> 1
    return pid, camid


def load_market1501_for_kpr(root_dir, ann_root, filtered_output_path, save_filtered=True, cache_file = 'market1501_cache.pkl'):
    """
    Load Market-1501 dataset with KPR-compatible format.
    Returns samples, PIDs, camera IDs, and image paths.
    
    Args:
        root_dir: Path to Market-1501 root directory
        ann_root: Path to annotation root directory
        filtered_output_path: Path where filtered JSONs will be saved
        save_filtered: Whether to save filtered annotations
    
    Returns:
        tuple: (query_samples, gallery_samples, query_pids, gallery_pids, 
                query_camids, gallery_camids, query_paths, gallery_paths)
    """
    print("Checking for cached data...")
        
    # Attempt to load from cache 
    if os.path.exists(cache_file):
        try:
            with open(cache_file, 'rb') as f:
                cached_data = pickle.load(f)
            
            # Unpack the 8 elements from the cache
            (query_samples, gallery_samples, 
            query_pids, gallery_pids,
            query_camids, gallery_camids,
            query_paths, gallery_paths) = cached_data
            
            print(f"Data loaded successfully from cache: {cache_file}")
            print(f"Query: {len(query_samples)} samples, Gallery: {len(gallery_samples)} samples")
            return cached_data

        except Exception as e:
            print(f" Cache file found but failed to load ({e}). Recomputing data...")
            

    print("Cache not found or corrupted. Loading and processing data using load_market1501_for_kpr...")
    def load_kpr_samples(img_folder, ann_folder, filter_invalid=False, is_query=False):
        samples = []
        pids = []
        camids = []
        img_paths = []
        skipped = {"no_ann": 0, "no_img": 0, "invalid_id": 0, "multi_target": 0, "load_error": 0}
        
        # # Create output directory for filtered JSONs
        # if save_filtered:
        #     if is_query:
        #         output_json_dir = os.path.join(filtered_output_path, 'query')
        #     else:
        #         output_json_dir = os.path.join(filtered_output_path, 'gallery')
            
        #     os.makedirs(output_json_dir, exist_ok=True)
        #     print(f"  Filtered JSONs will be saved to: {output_json_dir}")
        
        img_files = sorted([f for f in os.listdir(img_folder) 
                           if f.lower().endswith(('.jpg'))])
        
        for img_name in tqdm(img_files, desc=f"  Loading {'query' if is_query else 'gallery'}"):
            # Filter invalid IDs
            if filter_invalid and img_name.startswith("-1"):
                skipped["invalid_id"] += 1
                continue
            
            # Construct paths
            img_path = os.path.join(img_folder, img_name)
            ann_path = os.path.join(ann_folder, img_name + "_keypoints.json")
            
            # Check annotation exists
            if not os.path.exists(ann_path):
                skipped["no_ann"] += 1
                continue
            
            # Load image
            img = cv2.imread(img_path)
            if img is None:
                skipped["no_img"] += 1
                continue
            
            try:
                # Load keypoints annotation
                with open(ann_path, "r") as f:
                    kp_data = json.load(f)
                
                keypoints_xyc = []
                negative_kps = []
                
                for entry in kp_data:
                    if entry["is_target"]:
                        keypoints_xyc.append(entry["keypoints"])
                    else:
                        negative_kps.append(entry["keypoints"])
                
                # Must have exactly one target
                if len(keypoints_xyc) != 1:
                    skipped["multi_target"] += 1
                    continue
                
                # Save filtered JSON
                if save_filtered:
                    output_json_path = os.path.join(output_json_dir, os.path.basename(ann_path))
                    shutil.copy2(ann_path, output_json_path)
                
                # Extract PID and camera ID from filename
                pid, camid = extract_pid_camid_from_filename(img_name)
                
                # Process keypoints to pixel coordinates
                target_kps = process_keypoints(
                    np.array(keypoints_xyc[0]), 
                    img.shape
                )
                
                # Handle negative keypoints properly
                if negative_kps:
                    neg_kps_processed = []
                    for neg_person_kps in negative_kps:
                        processed = process_keypoints(
                            np.array(neg_person_kps),
                            img.shape
                        )
                        neg_kps_processed.append(processed)
                    neg_kps = np.array(neg_kps_processed)
                else:
                    # Empty array with correct shape for no negative keypoints
                    neg_kps = np.zeros((0, 17, 3))

                # if is_query:
                #     colab_path = colab_dataset+"/query/"+img_name
                # else:
                #     colab_path = colab_dataset+"/bounding_box_test/"+img_name
                # Create sample dictionary
                sample = {
                    "img_path": img_path,
                    #"image": img,
                    "keypoints_xyc": target_kps,
                    "negative_keypoints_xyc": neg_kps,
                    "negative_kps": neg_kps
                }
                
                samples.append(sample)
                pids.append(pid)
                camids.append(camid)
                img_paths.append(img_path)
                
            except Exception as e:
                skipped["load_error"] += 1
                continue
        
        print(f"  Loaded: {len(samples)} samples")
        print(f"  Skipped: {skipped}")
        return samples, np.array(pids), np.array(camids), img_paths

    root_dir = Path(root_dir)
    ann_root = Path(ann_root)

    print("\n" + "="*70)
    print("Loading Market-1501 Dataset for Evaluation")
    print("="*70)
    
    print("\n[Query Set]")
    query_samples, query_pids, query_camids, query_paths = load_kpr_samples(
        os.path.join(root_dir, "query"),
        os.path.join(ann_root, "query"),
        filter_invalid=True,
        is_query=True
    )
    
    print("\n[Gallery Set]")
    gallery_samples, gallery_pids, gallery_camids, gallery_paths = load_kpr_samples(
        os.path.join(root_dir, "bounding_box_test"),
        os.path.join(ann_root, "bounding_box_test"),
        filter_invalid=True,
        is_query=False
    )
    
    print("\n" + "="*70)
    print(f"Dataset loaded successfully!")
    print(f"Query: {len(query_samples)} samples")
    print(f"Gallery: {len(gallery_samples)} samples")
    print(f"Unique identities in query: {len(np.unique(query_pids))}")
    print(f"Unique identities in gallery: {len(np.unique(gallery_pids))}")
    print("="*70 + "\n")

    data_to_cache = (query_samples, gallery_samples, query_pids, gallery_pids, 
            query_camids, gallery_camids, query_paths, gallery_paths)

    print(f"Saving loaded data to cache: {cache_file}...")

    try:
        with open(cache_file, 'wb') as f:
            pickle.dump(data_to_cache, f)
        print(f" Data successfully cached.")
    except Exception as e:
        print(f" Failed to save data to cache: {e}")

    return (query_samples, gallery_samples, query_pids, gallery_pids, 
            query_camids, gallery_camids, query_paths, gallery_paths)


if __name__ == "__main__":
    # Load dataset
    print("Loading Market-1501 dataset...")
    query_samples, gallery_samples = load_market1501_for_kpr(root, ann_root)
    
    # Subset for quick testing
    print(f"\nUsing subset: 14 query + 14 gallery samples")
    query_samples = query_samples[:14]
    gallery_samples = gallery_samples[:14]
    
    # Visualize loaded samples with keypoints
    print("\nVisualizing samples (before feature extraction)...")
    display_mode = 'save'
    os.makedirs('market1501_demo/results', exist_ok=True)
    display_kpr_reid_samples_grid(
        query_samples + gallery_samples, 
        display_mode=display_mode, 
        save_path='market1501_demo/results/samples_input.png'
    )
    print("Saved input visualization")
    
    # Initialize KPR feature extractor
    print("\nInitializing KPR model...")
    extractor = KPRFeatureExtractor(kpr_cfg)
    print("Model loaded")
    
    
    # Extract features
    print("\nExtracting features...")
    print("Processing query samples...")
    query_samples, q_emb, q_vis, q_masks = extractor(query_samples)
    print(f"Query embeddings shape: {q_emb.shape}")
    
    print("Processing gallery samples...")
    gallery_samples, g_emb, g_vis, g_masks = extractor(gallery_samples)
    print(f"Gallery embeddings shape: {g_emb.shape}")
    print("Feature extraction complete")
    
    # Visualize with part attention masks
    print("\nVisualizing part attention masks...")
    display_kpr_reid_samples_grid(
        query_samples + gallery_samples, 
        display_mode=display_mode, 
        save_path='market1501_demo/results/samples_with_masks.png'
    )
    print("Saved attention visualization")
    
    # Compute distance matrix
    print("\nComputing distance matrix...")
    distance_matrix, body_parts_distmat = compute_distance_matrix_using_bp_features(
        q_emb,
        g_emb,
        q_vis,
        g_vis,
        use_gpu=kpr_cfg.use_gpu,
        use_logger=False
    )
    distances = distance_matrix.cpu().detach().numpy() / 2  # Scale to [0, 1]
    print(f"Distance matrix shape: {distances.shape}")
    
    # Visualize distance matrix
    print("\nVisualizing distance matrix...")
    display_distance_matrix(
        distances, 
        query_samples, 
        gallery_samples, 
        display_mode=display_mode, 
        save_path='market1501_demo/results/distance_matrix.png'
    )
    print("Saved distance matrix visualization")
    
    # Print results summary
    print("\n" + "="*60)
    print("RESULTS SUMMARY")
    print("="*60)
    print(f"Distance matrix shape: {distances.shape}")
    print(f"Distance range: [{distances.min():.3f}, {distances.max():.3f}]")
    print(f"Mean distance: {distances.mean():.3f}")
    print("\nSample distances (first 5x5):")
    print(distances[:5, :5])
    print("\nPipeline completed successfully!")
    print("="*60)