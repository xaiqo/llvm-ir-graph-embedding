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
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    try:
        model = GraphModule.load_from_checkpoint(ckpt_path, map_location=device)
    except Exception as e:
        print(f"Error loading checkpoint: {e}")
        return
        
    model.eval()
    model.to(device)
    
    # Nodes
    node_id_map = {node['id']: i for i, node in enumerate(raw.get('nodes', []))}
    
    if model.hparams.use_bert:
        encoder = CodeBERTEncoder(device=device)
        texts = [f"{n['category']}: {n.get('label','')} {n.get('type_id','')}" for n in raw.get('nodes', [])]
        x = encoder.encode(texts) # Encoder handles device placement
    else:
        print("  Error: Discrete mode requires vocabulary mapping. Use BERT mode for inference demo.")
        return

    # Edges
    edge_type_map = {
        "Control_Next": 0, "Control_Jump": 1, "Data_Use": 2, 
        "Type_Of": 3, "Memory_Alias": 4, "Memory_MustAlias": 5
    }
    
    edge_index = []
    edge_attr = []
    
    for edge in raw.get('edges', []):
        if edge['src'] in node_id_map and edge['dst'] in node_id_map:
            src = node_id_map[edge['src']]
            dst = node_id_map[edge['dst']]
            edge_index.append([src, dst])
            edge_attr.append(edge_type_map.get(edge['type'], 0))
            
    if not edge_index:
        edge_index = torch.zeros((2, 0), dtype=torch.long).to(device)
        edge_attr = torch.zeros((0), dtype=torch.long).to(device)
    else:
        edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous().to(device)
        edge_attr = torch.tensor(edge_attr, dtype=torch.long).to(device)
        
    # Batch wrapper (PyG models expect batch index)
    batch = torch.zeros(x.size(0), dtype=torch.long).to(device)
    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, batch=batch)

    print("  Running Inference...")
    with torch.no_grad():
        out = model(data)
        probs = torch.exp(out)
        pred_class = probs.argmax(dim=1).item()
        confidence = probs[0, pred_class].item()
        
    print(f"  Predicted Class: {pred_class} (Confidence: {confidence:.2f})")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("file", help="C++ source file")
    parser.add_argument("--ckpt", required=True, help="Model checkpoint")
    args = parser.parse_args()
    
    predict(args.file, args.ckpt)
