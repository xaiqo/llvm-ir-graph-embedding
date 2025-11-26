import torch
from transformers import AutoTokenizer, AutoModel

class CodeBERTEncoder:
    def __init__(self, model_name="microsoft/codebert-base", device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Loading CodeBERT ({model_name}) to {self.device}...")
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(self.device)
        self.model.eval()

    @torch.no_grad()
    def encode(self, texts, batch_size=32):
        """
        Encodes a list of text strings into vectors using CodeBERT.
        Returns a tensor of shape [len(texts), hidden_dim].
        """
        all_embeddings = []
        
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i : i + batch_size]
            
            # Tokenize
            inputs = self.tokenizer(
                batch_texts, 
                padding=True, 
                truncation=True, 
                max_length=64, # Instructions are usually short
                return_tensors="pt"
            ).to(self.device)
            
            # Forward pass
            outputs = self.model(**inputs)
            
            # Shape: [batch_size, hidden_dim]
            cls_embeddings = outputs.last_hidden_state[:, 0, :]
            all_embeddings.append(cls_embeddings.cpu())
            
        if not all_embeddings:
            return torch.empty(0, 768)
            
        return torch.cat(all_embeddings, dim=0)




