import argparse
import os
import torch
import subprocess
from ml_core.train import GraphModule
from ml_core.data.toon_parser import TOONParser
from torch_geometric.data import Data
from ml_core.utils.encoder import CodeBERTEncoder

from data_pipeline.compile_and_extract import compile_to_ir, extract_graph

def predict(src_file, ckpt_path):
    print(f"Processing {src_file}...")
    
    temp_dir = "temp_inference"
    os.makedirs(temp_dir, exist_ok=True)
    
    print("  Compiling to IR...")
    ir_file = compile_to_ir(src_file, temp_dir, optimize=True)
    if not ir_file:
        print("  Error: Compilation failed.")
        return
        
    print("  Extracting Graph (TOON)...")
    toon_file = extract_graph(ir_file, temp_dir)
    if not toon_file:
        print("  Error: Graph extraction failed.")
        return

    print("  Parsing Graph...")
    parser = TOONParser()
    raw = parser.parse_file(toon_file)

    
    print(f"Loading Model from {ckpt_path}...")
    model = GraphModule.load_from_checkpoint(ckpt_path)
    model.eval()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    
    if model.hparams.use_bert:
        encoder = CodeBERTEncoder()
        texts = [f"{n['category']}: {n.get('label','')} {n.get('type_id','')}" for n in raw['nodes']]
        x = encoder.encode(texts).to(device)
    else:
        print("  Warning: Discrete mode requires vocabulary mapping. Use BERT mode for portability.")
        return

    with torch.no_grad():
        out = model(data)
        pred_class = out.argmax(dim=1).item()
        
    print(f"  Predicted Class: {pred_class}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("file", help="C++ source file")
    parser.add_argument("--ckpt", required=True, help="Model checkpoint")
    args = parser.parse_args()
    
    predict(args.file, args.ckpt)

