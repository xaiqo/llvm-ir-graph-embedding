import os
import json
import argparse
from datasets import load_dataset
from tqdm import tqdm

def download_poj104(output_dir):
    """
    Downloads the POJ-104 dataset from HuggingFace and organizes it for our pipeline.
    Structure:
      output_dir/
        raw/
          train/
            class_id/
              file_id.cpp
          test/
            ...
        labels.json
    """
    print("Downloading POJ-104 from HuggingFace...")
    dataset = load_dataset("google/code_x_glue_cc_clone_detection_poj104", split=None)
    
    print("Dataset loaded. Processing splits...")
    splits = ['train', 'validation', 'test']
    mapping_dir = {'train': 'train', 'validation': 'val', 'test': 'test'}
    
    stats = {k: 0 for k in mapping_dir.values()}
    
    for split in splits:
        if split not in dataset: continue
        
        data_split = dataset[split]
        target_dir = os.path.join(output_dir, mapping_dir[split])
        os.makedirs(target_dir, exist_ok=True)
        
        print(f"Processing {split} set ({len(data_split)} samples)...")
        
        for item in tqdm(data_split):
            file_id = item['id']
            code = item['code']
            label = str(item['label'])
            
            class_dir = os.path.join(target_dir, label)
            os.makedirs(class_dir, exist_ok=True)
            
            file_path = os.path.join(class_dir, f"{file_id}.cpp")
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(code)
            
            stats[mapping_dir[split]] += 1
            
    print("Download complete.")
    print("Stats:", stats)
    print(f"Data saved to: {output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="data/raw/poj104", help="Output directory")
    args = parser.parse_args()
    
    download_poj104(args.output)


