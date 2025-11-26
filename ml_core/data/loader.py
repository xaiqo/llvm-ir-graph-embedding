import os
import torch
from torch_geometric.data import Dataset, Data
from tqdm import tqdm
from ml_core.utils.encoder import CodeBERTEncoder
from ml_core.data.toon_parser import TOONParser

class LLVMGraphDataset(Dataset):
    def __init__(self, root, use_bert=False, transform=None, pre_transform=None):
        super(LLVMGraphDataset, self).__init__(root, transform, pre_transform)
        self.graph_files = [os.path.join(root, f) for f in os.listdir(root) if f.endswith('.toon')]
        self.use_bert = use_bert
        self.parser = TOONParser()
        
        # Initialize Encoder only if needed and valid
        self.bert_encoder = None
        if self.use_bert:
            try:
                self.bert_encoder = CodeBERTEncoder()
            except Exception as e:
                print(f"Warning: Could not load CodeBERT ({e}). Falling back to Opcode IDs.")
                self.use_bert = False

        # Mapping for Opcodes and Edge Types (Simple Vocabulary)
        self.opcode_to_idx = {"<UNK>": 0}
        self.edge_type_to_idx = {
            "Control_Next": 0, 
            "Control_Jump": 1, 
            "Data_Use": 2, 
            "Type_Of": 3, 
            "Memory_Alias": 4, 
            "Memory_MustAlias": 5
        }

    def len(self):
        return len(self.graph_files)

    def get(self, idx):
        file_path = self.graph_files[idx]
        
        # Parse TOON format
        raw = self.parser.parse_file(file_path)
        
        # 1. Process Nodes
        nodes = raw.get('nodes', [])
        node_id_map = {node['id']: i for i, node in enumerate(nodes)}
        
        if self.use_bert and self.bert_encoder:
            # Hybrid Mode: Use CodeBERT embeddings
            texts = []
            for node in nodes:
                label = node.get('label', 'UNK')
                category = node.get('category', 'Unknown')
                # TOON uses 'type_id' instead of 'data_type' from old JSON
                ret_type = node.get('type_id', '') 
                
                text = f"{category}: {label} {ret_type}".strip()
                texts.append(text)
            
            x = self.bert_encoder.encode(texts)
            
        else:
            # Simple Mode: Opcode IDs
            x = []
            for node in nodes:
                label = node.get('label', 'UNK')
                if label not in self.opcode_to_idx:
                    self.opcode_to_idx[label] = len(self.opcode_to_idx)
                
                # Map category string to int
                cat_map = {'Instruction': 0, 'Value': 1, 'Type': 2}
                cat_idx = cat_map.get(node.get('category'), 0)
                x.append([self.opcode_to_idx[label], cat_idx])
            
            x = torch.tensor(x, dtype=torch.long)

        # 2. Process Edges
        edge_index = []
        edge_type = []
        
        edges = raw.get('edges', [])
        for edge in edges:
            if edge['src'] in node_id_map and edge['dst'] in node_id_map:
                src_idx = node_id_map[edge['src']]
                dst_idx = node_id_map[edge['dst']]
                edge_index.append([src_idx, dst_idx])
                
                etype = edge.get('type', 'Control_Next')
                edge_type.append(self.edge_type_to_idx.get(etype, 0))
                
        if not edge_index:
            edge_index = torch.zeros((2, 0), dtype=torch.long)
            edge_type = torch.zeros((0), dtype=torch.long)
        else:
            edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
            edge_type = torch.tensor(edge_type, dtype=torch.long)

        return Data(x=x, edge_index=edge_index, edge_attr=edge_type)

    def num_node_features(self):
        if self.use_bert:
            return 768 
        return 2