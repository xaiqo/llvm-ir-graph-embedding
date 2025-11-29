import torch
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import numpy as np
import pytorch_lightning as pl
from ml_core.train import GraphModule
from ml_core.data.loader import LLVMGraphDataset
from torch_geometric.loader import DataLoader
from torch_geometric.nn import global_mean_pool
import argparse
import os

def visualize_embeddings(ckpt_path, data_dir, output_file="tsne_plot.png", limit=500):
    """
    Loads a trained model, extracts embeddings for a subset of graphs,
    and visualizes them using t-SNE.
    """
    print(f"Loading checkpoint: {ckpt_path}")

    try:
        model = GraphModule.load_from_checkpoint(ckpt_path, map_location="cpu", weights_only=False)
    except TypeError:
        model = GraphModule.load_from_checkpoint(ckpt_path, map_location="cpu")
        
    model.eval()
    model.freeze()
    
    print("Loading dataset...")
    dataset = LLVMGraphDataset(root=data_dir, use_bert=model.hparams.use_bert)
    loader = DataLoader(dataset, batch_size=1, shuffle=True)
    
    embeddings = []
    labels = []
    
    print(f"Extracting embeddings (limit={limit})...")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    
    count = 0
    for data in loader:
        if count >= limit: break
        data = data.to(device)
        
        # Forward pass up to pooling
        # We need to access the internal GNN
        with torch.no_grad():
            x, edge_index, edge_type, batch = data.x, data.edge_index, data.edge_attr, data.batch
            
            if model.hparams.use_bert:
                h = model.model.input_proj(x.float())
            else:
                h_op = model.model.opcode_embedding(x[:, 0])
                h_cat = model.model.category_embedding(x[:, 1])
                h = h_op + h_cat
                
            h = model.model.conv1(h, edge_index, edge_type)
            h = torch.relu(h)
            h = model.model.conv2(h, edge_index, edge_type)
            h = torch.relu(h)
            h = model.model.conv3(h, edge_index, edge_type)
            
            graph_emb = global_mean_pool(h, batch)
            
            embeddings.append(graph_emb.cpu().numpy())

            labels.append(np.random.randint(0, 5)) 
            
        count += 1
        
    embeddings = np.concatenate(embeddings, axis=0)
    
    print("Running t-SNE...")
    tsne = TSNE(n_components=2, random_state=42)
    emb_2d = tsne.fit_transform(embeddings)
    
    print("Plotting...")
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(emb_2d[:, 0], emb_2d[:, 1], c=labels, cmap='tab10', alpha=0.7)
    plt.colorbar(scatter)
    plt.title(f"t-SNE of Code Embeddings (Hybrid GNN)")
    plt.savefig(output_file)
    print(f"Saved plot to {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", required=True, help="Path to model checkpoint (.ckpt)")
    parser.add_argument("--data", required=True, help="Path to graph data directory")
    parser.add_argument("--output", default="tsne_plot.png", help="Output image file")
    args = parser.parse_args()
    
    # Handle glob pattern for checkpoint manually if passed as string
    import glob
    if "*" in args.ckpt:
        files = glob.glob(args.ckpt)
        if not files:
            print("No checkpoint found matching pattern.")
            exit(1)
        args.ckpt = files[-1] # Take the latest
        print(f"Auto-selected checkpoint: {args.ckpt}")
    
    visualize_embeddings(args.ckpt, args.data, args.output)


