from dataPreprocess import AudioDataset
import torch
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt
from typing import List, Optional
from torch.utils.data import DataLoader
import pytorch_lightning as pl

import pl_model

def calculate_classification_metrics(
    model: pl.LightningModule,
    dataloader: torch.utils.data.DataLoader,
    label_list: List[str],
    device: Optional[str] = None
) -> None:
    """
    Calculate and display classification metrics including confusion matrix,
    accuracy, precision, recall, and F1 score.
    
    Args:
        model: Trained PyTorch Lightning model
        dataloader: DataLoader containing validation/test data
        label_list: List of class labels
        device: Device to run inference on ('cuda' or 'cpu')
    """
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    model.to(device)
    model.eval()
    
    # Lists to store predictions and true labels
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch in dataloader:
            # Assuming batch contains inputs and labels
            # Modify according to your dataloader structure
            inputs, labels = batch
            inputs = inputs.to(device)
            
            # Get model predictions
            outputs = model(inputs)
            preds = torch.argmax(outputs, dim=1).cpu().numpy()
            
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().numpy())
    
    # Convert lists to numpy arrays
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    
    # Calculate confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    
    # Plot confusion matrix
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=label_list,
                yticklabels=label_list)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.show()
    
    # Print classification report
    print("\nClassification Report:")
    print(classification_report(all_labels, all_preds,
                              target_names=label_list,
                              digits=3))
    
    # Calculate additional metrics
    accuracy = np.sum(all_preds == all_labels) / len(all_labels)
    
    # Calculate per-class metrics
    per_class_metrics = {}
    for i, class_name in enumerate(label_list):
        true_positives = np.sum((all_labels == i) & (all_preds == i))
        false_positives = np.sum((all_labels != i) & (all_preds == i))
        false_negatives = np.sum((all_labels == i) & (all_preds != i))
        
        precision = true_positives / (true_positives + false_positives + 1e-10)
        recall = true_positives / (true_positives + false_negatives + 1e-10)
        f1 = 2 * (precision * recall) / (precision + recall + 1e-10)
        
        per_class_metrics[class_name] = {
            'precision': precision,
            'recall': recall,
            'f1': f1
        }
    
    # Print overall accuracy and per-class metrics
    print(f"\nOverall Accuracy: {accuracy:.3f}")
    print("\nPer-class Metrics:")
    for class_name, metrics in per_class_metrics.items():
        print(f"\n{class_name}:")
        print(f"Precision: {metrics['precision']:.3f}")
        print(f"Recall: {metrics['recall']:.3f}")
        print(f"F1 Score: {metrics['f1']:.3f}")

if __name__ == "__main__":
    valid_csv = r"D:\ICS Project\Project\Output\valiData_Wingbeats.csv"
    label_list = {
        "aegypti": 0, 
        "albopictus": 1, 
        "arabiensis": 2, 
        "gambiae": 3, 
        "quinquefasciatus": 4, 
        "pipiens": 5
    }
    valid_data = AudioDataset(valid_csv, label_list, "validation")
    valid_loader = DataLoader(valid_data, batch_size=4,shuffle=False)
    model = pl_model.MosquitoClassifier.load_from_checkpoint(r"D:\ICS Project\Project\checkpoints\resnet-attention-epoch=16-val_acc=0.86.ckpt")


    calculate_classification_metrics(model, valid_loader, label_list)