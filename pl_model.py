import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np
import dataPreprocess
import model
from pytorch_lightning.loggers import TensorBoardLogger


class MosquitoClassifier(pl.LightningModule):
    def __init__(self, num_classes=6, learning_rate=2e-4):
        super().__init__()
        self.save_hyperparameters()
        
        # Model architecture
        self.model = model.resnet18(1, num_classes)  # 1 channel, 6 classes
        self.learning_rate = learning_rate
        self.criterion = nn.CrossEntropyLoss()
        
        # Metrics
        self.best_valid_acc = 0
        self.best_model_report = ''
        
    def forward(self, x):
        return self.model(x)
    
    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.learning_rate)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200, eta_min=0)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_loss"
            }
        }
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        z = x.shape
        logits = self(x)
        loss = self.criterion(logits, y)
        
        # Calculate accuracy
        preds = torch.argmax(logits, dim=1)
        acc = (preds == y).float().mean()
        
        # Log metrics
        self.log('train_loss', loss, prog_bar=True)
        self.log('train_acc', acc, prog_bar=True)
        
        return loss
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)
        
        # Calculate accuracy
        preds = torch.argmax(logits, dim=1)
        acc = (preds == y).float().mean()
        
        # Store predictions for metrics calculation
        self.log('val_loss', loss)
        self.log('val_acc', acc)
        
        return loss
    
        
def main(train_csv, valid_csv, batch_size=4, num_epochs=55, learning_rate=2e-4, dataset_name='wingbeats'):
    # Data preparation
    label_list = {
        "aegypti": 0, 
        "albopictus": 1, 
        "arabiensis": 2, 
        "gambiae": 3, 
        "quinquefasciatus": 4, 
        "pipiens": 5
    }
    
    # Create datasets
    train_data = dataPreprocess.AudioDataset(train_csv, label_list, "train", dataset_name)
    valid_data = dataPreprocess.AudioDataset(valid_csv, label_list, "validation", dataset_name)
    
    # Create dataloaders
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(valid_data, batch_size=batch_size, shuffle=False)
    

    
    # Initialize model
    model = MosquitoClassifier(num_classes=6, learning_rate=learning_rate)
    
    # Configure callbacks
    checkpoint_callback = ModelCheckpoint(
        monitor='val_acc',
        dirpath='./checkpoints',
        filename='resnet-attention-{epoch:02d}-{val_acc:.2f}',
        save_top_k=1,
        mode='max'
    )
    
    # Initialize trainer
    trainer = pl.Trainer(
        max_epochs=num_epochs,
        accelerator='auto',  # Automatically detect GPU/CPU
        callbacks=[checkpoint_callback],
        enable_progress_bar=True,
        enable_model_summary=True,
    )
    
    # Train the model
    trainer.fit(model, train_loader, valid_loader, ckpt_path=r"D:\ICS Project\Project\checkpoints\resnet-attention-epoch=16-val_acc=0.86.ckpt")

    
    logger = TensorBoardLogger("tb_logs", name="Mosquito Classifier")

    # Print final results
    print(f"Best validation accuracy: {model.best_valid_acc:.4f}")
    print("\nBest model classification report:")
    print(model.best_model_report)
    print(f"\nBest model checkpoint saved at: {checkpoint_callback.best_model_path}")

if __name__ == "__main__":
    train_csv = r"D:\ICS Project\Project\Output\trainData_Wingbeats.csv"
    valid_csv = r"D:\ICS Project\Project\Output\valiData_Wingbeats.csv"
    
    main(
        train_csv=train_csv,
        valid_csv=valid_csv,
        batch_size=6,
        num_epochs=26,
        learning_rate=2e-4,
        dataset_name='wingbeats'
    )