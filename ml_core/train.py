import torch
from ml_core.data.loader import LLVMGraphDataset
from ml_core.models.gnn import HybridGNN
from torch_geometric.loader import DataLoader

def train():
    use_bert = False 
    dataset = LLVMGraphDataset(root="data/processed/graphs", use_bert=use_bert)
    
    loader = DataLoader(dataset, batch_size=32, shuffle=True)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    num_features = dataset.num_node_features()
    if not use_bert:
        num_features = 1000 # Dummy vocab size
        
    model = HybridGNN(num_node_features=num_features, num_edge_types=6, use_bert=use_bert).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    print(f"Starting training on {device} (BERT={use_bert})...")
    
    model.train()
    for epoch in range(10):
        total_loss = 0
        for data in loader:
            data = data.to(device)
            optimizer.zero_grad()
            out = model(data)
            
            y = torch.randint(0, 10, (data.num_graphs,)).to(device)
            
            loss = torch.nn.functional.nll_loss(out, y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            
        print(f"Epoch {epoch}, Loss: {total_loss}")

if __name__ == "__main__":
    import os
    if not os.path.exists("data/processed/graphs"):
        print("Please run pipeline first to generate graphs.")
    else:
        train()

