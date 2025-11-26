import os
import torch
import pytorch_lightning as pl
from torch_geometric.loader import DataLoader
from ml_core.data.loader import LLVMGraphDataset
from ml_core.models.gnn import HybridGNN
import torch.nn.functional as F

class GraphModule(pl.LightningModule):
    def __init__(self, hparams):
        super().__init__()
        self.save_hyperparameters(hparams)
        
        # If BERT is used, input dim is 768, else it's vocab size (dummy 1000)
        input_dim = 768 if self.hparams.use_bert else 1000
        
        self.model = HybridGNN(
            num_node_features=input_dim,
            num_edge_types=6,
            hidden_dim=self.hparams.hidden_dim,
            num_classes=self.hparams.num_classes,
            use_bert=self.hparams.use_bert
        )

    def forward(self, data):
        return self.model(data)

    def training_step(self, batch, batch_idx):
        # Dummy labels (replace with real labels from dataset)
        y = torch.randint(0, self.hparams.num_classes, (batch.num_graphs,)).to(self.device)
        
        out = self(batch)
        loss = F.nll_loss(out, y)
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.lr)

    def train_dataloader(self):
        dataset = LLVMGraphDataset(
            root=self.hparams.data_dir, 
            use_bert=self.hparams.use_bert
        )
        return DataLoader(dataset, batch_size=self.hparams.batch_size, shuffle=True, num_workers=4)

# Hydra Entry Point
import hydra
from omegaconf import DictConfig

@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig):
    pl.seed_everything(cfg.seed)
    
    model = GraphModule(cfg.model)
    trainer = pl.Trainer(
        max_epochs=cfg.train.epochs,
        accelerator="auto",
        devices=1,
        log_every_n_steps=10
    )
    
    trainer.fit(model)

if __name__ == "__main__":
    main()
