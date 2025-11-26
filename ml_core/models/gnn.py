import torch
import torch.nn.functional as F
from torch_geometric.nn import RGCNConv, global_mean_pool

class HybridGNN(torch.nn.Module):
    def __init__(self, num_node_features, num_edge_types, hidden_dim=64, num_classes=10, use_bert=False):
        super(HybridGNN, self).__init__()
        self.use_bert = use_bert
        
        if not self.use_bert:
            # Embeddings for discrete inputs (Opcode ID)
            # num_node_features here isvocab size
            self.opcode_embedding = torch.nn.Embedding(num_node_features, hidden_dim)
            self.category_embedding = torch.nn.Embedding(3, hidden_dim)
            input_dim = hidden_dim
        else:
            # Linear projection for continuous inputs (CodeBERT vectors)
            # num_node_features here is 768
            self.input_proj = torch.nn.Linear(num_node_features, hidden_dim)
            input_dim = hidden_dim
        
        # GNN Layers (Relational GCN)
        self.conv1 = RGCNConv(input_dim, hidden_dim, num_relations=num_edge_types)
        self.conv2 = RGCNConv(hidden_dim, hidden_dim, num_relations=num_edge_types)
        self.conv3 = RGCNConv(hidden_dim, hidden_dim, num_relations=num_edge_types)
        
        # Readout & Classifier
        self.fc1 = torch.nn.Linear(hidden_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, num_classes)
        
    def forward(self, data):
        x, edge_index, edge_type, batch = data.x, data.edge_index, data.edge_attr, data.batch
        
        if not self.use_bert:
            # x[:, 0] is opcode_idx, x[:, 1] is category_idx
            h_op = self.opcode_embedding(x[:, 0])
            h_cat = self.category_embedding(x[:, 1])
            h = h_op + h_cat 
        else:
            # x is [num_nodes, 768]
            h = self.input_proj(x.float())
        
        # Message Passing
        h = self.conv1(h, edge_index, edge_type)
        h = F.relu(h)
        h = F.dropout(h, p=0.2, training=self.training)
        
        h = self.conv2(h, edge_index, edge_type)
        h = F.relu(h)
        
        h = self.conv3(h, edge_index, edge_type)
        
        # Global Pooling
        h = global_mean_pool(h, batch)
        
        # Classification
        h = F.relu(self.fc1(h))
        out = self.fc2(h)
        
        return F.log_softmax(out, dim=1)

