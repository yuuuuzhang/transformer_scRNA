# Cell Type Prediction Using Transformer on Single-Cell RNA-seq Data

This repository contains a deep learning project utilizing a Transformer-based model to predict cell types using single-cell RNA-seq (scRNA-seq) data. The model achieves **92.86% accuracy** on a 10-class dataset and demonstrates superior performance when trained on raw count data compared to normalized data.

## Features

- **Transformer-based architecture**: A novel classification model incorporating Transformer layers for enhanced performance.
- **Support for raw count data**: The model outperforms on raw count data, making preprocessing optional.
- **Visualization of latent space**: Visualizes cell type clusters in the latent space using UMAP.
- **Generalizable**: Tested on multiple single-cell datasets (e.g., `paul15`, `moignard15`).

---

## Installation

### Prerequisites
- Python 3.8+
- PyTorch 1.9+
- Additional dependencies:
  ```bash
  pip install numpy matplotlib scanpy umap scikit-learn
  ```

---

## Model Architecture

### Overview
The model comprises three main components:
1. **Encoder**: Compresses high-dimensional input into a latent representation.
2. **Transformer Encoder**: Processes the latent representation to capture dependencies between features.
3. **Classifier**: Outputs predicted cell type probabilities.

### Code Example
```python
class ClassificationModel(nn.Module):
    def __init__(self, input_dim, latent_dim, num_classes, num_heads=2, num_layers=2, dropout=0.1):
        super(ClassificationModel, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, latent_dim)
        )
        encoder_layer = nn.TransformerEncoderLayer(d_model=latent_dim, nhead=num_heads, dim_feedforward=latent_dim * 4, dropout=dropout)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.classifier = nn.Linear(latent_dim, num_classes)

    def forward(self, x):
        latent = self.encoder(x)
        latent = latent.unsqueeze(1).permute(1, 0, 2)
        transformer_output = self.transformer(latent).squeeze(0)
        predicted_labels = self.classifier(transformer_output)
        return transformer_output, predicted_labels
```

---

## Dataset Details

### Dataset I: `paul15`
- **Description**: Contains raw count data with 10 cell types.
- **Preprocessing**:
  - PCA and UMAP for visualization.
  - Labels are encoded into integers.
![paul15](paul15.png)
### Dataset II: `moignard15`
- **Description**: Contains normalized data with 5 cell types.
- **Preprocessing**:
  - Similar steps as Dataset I.
![moignard15](moignard15.png)
---

## Training and Testing

### Training
Define hyperparameters:
```python
input_dim = X_train_tensor.shape[1]
latent_dim = 16
num_classes = len(np.unique(y))
learning_rate = 0.001
batch_size = 32
num_epochs = 20
```
Train the model:
```python
epoch_losses = train_model(model, criterion, optimizer, X_train_tensor, y_train_tensor, num_epochs, batch_size)
plot_loss(epoch_losses)
```

### Testing
Evaluate the model:
```python
test_accuracy, classification_report_str, test_latent, test_predictions = test_model(model, X_test_tensor, y_test)
```

### Results
- **Dataset I**: 92.86% accuracy

- **Dataset II**: 76.75% accuracy

---

## Visualizing Latent Space
UMAP is used to visualize latent space clustering:
```python
latent_2d = umap.UMAP().fit_transform(test_latent)
plt.scatter(latent_2d[:, 0], latent_2d[:, 1], c=predicted_categories, cmap='viridis')
plt.title("Latent Space Visualization")
plt.show()
```
- **Dataset I**:
  
![Dataset I](dataset1.png)

- **Dataset II**: 
![Dataset II](dataset2.png)

---

## Key Observations
- **Performance**: Raw count data yields better results than normalized data.
- **Visualization**: Clear separation of cell types in latent space.

---

