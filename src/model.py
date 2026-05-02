import numpy as np
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


class SimpleLSTM(nn.Module):
    def __init__(self, input_dim=254, hidden_dim=16, num_classes=10):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers=1, batch_first=True)
        self.dropout = nn.Dropout(0.7) # Extremely strong dropout
        self.fc = nn.Linear(hidden_dim, num_classes)
        
    def forward(self, x):
        # x: (B, T, D)
        out, (hn, cn) = self.lstm(x)
        # use the last hidden state with dropout
        out = self.dropout(hn[-1])
        out = self.fc(out)
        return out


def train_lstm(X_train, y_train, X_val, y_val, epochs=200, lr=0.005, weight_decay=1e-2):
    num_classes = len(np.unique(y_train))
    model = SimpleLSTM(hidden_dim=16, num_classes=num_classes)
    criterion = nn.CrossEntropyLoss()
    # Add Strong L2 Regularization (weight_decay)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    
    # Convert to PyTorch tensors
    X_train_t = torch.tensor(X_train, dtype=torch.float32)
    y_train_t = torch.tensor(y_train, dtype=torch.long)
    X_val_t = torch.tensor(X_val, dtype=torch.float32)
    y_val_t = torch.tensor(y_val, dtype=torch.long)
    
    dataset = TensorDataset(X_train_t, y_train_t)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
    
    best_val_acc = 0.0
    best_model_state = None
    patience = 10
    epochs_no_improve = 0
    
    for epoch in range(epochs):
        model.train()
        for batch_x, batch_y in dataloader:
            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            
        # Early Stopping Check
        model.eval()
        with torch.no_grad():
            val_preds = model(X_val_t).argmax(dim=1).numpy()
            val_acc = accuracy_score(y_val, val_preds)
            
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_model_state = model.state_dict()
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1
                
        if epochs_no_improve >= patience:
            break
            
    # Load best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        
    model.eval()
    with torch.no_grad():
        train_preds = model(X_train_t).argmax(dim=1).numpy()
        val_preds = model(X_val_t).argmax(dim=1).numpy()
        
    train_acc = accuracy_score(y_train, train_preds)
    val_acc = accuracy_score(y_val, val_preds)
    
    print("PyTorch LSTM (Ultra-Regularized + Early Stopping)")
    print(f"  Train Acc: {train_acc:.4f}")
    print(f"  Val Acc:   {val_acc:.4f}")
    print("-" * 40)


def load_data():
    X_train = np.load("data/features/X_train.npy")
    y_train = np.load("data/features/y_train.npy")
    X_val = np.load("data/features/X_val.npy")
    y_val = np.load("data/features/y_val.npy")
    return X_train, y_train, X_val, y_val


def evaluate(model, name, X_train, y_train, X_val, y_val):
    model.fit(X_train, y_train)

    train_acc = accuracy_score(y_train, model.predict(X_train))
    val_acc = accuracy_score(y_val, model.predict(X_val))

    print(f"{name}")
    print(f"  Train Acc: {train_acc:.4f}")
    print(f"  Val Acc:   {val_acc:.4f}")
    print("-" * 40)


def main():
    X_train, y_train, X_val, y_val = load_data()
    
    print("Comparing models...\n")
    
    # Train LSTM directly on 3D data: (N, T, 126)
    train_lstm(X_train, y_train, X_val, y_val, epochs=100)

    # Flatten for sklearn models
    N_train, T, F = X_train.shape
    N_val = X_val.shape[0]
    X_train_flat = X_train.reshape(N_train, T * F)
    X_val_flat = X_val.reshape(N_val, T * F)

    # Normalize
    scaler = StandardScaler()
    X_train_flat = scaler.fit_transform(X_train_flat)
    X_val_flat = scaler.transform(X_val_flat)

    # PCA (CRUCIAL)
    pca = PCA(n_components=0.99, random_state=42)
    X_train_flat = pca.fit_transform(X_train_flat)
    X_val_flat = pca.transform(X_val_flat)

    # 1. Linear SVM
    evaluate(LinearSVC(C=1e-5, max_iter=5000), "Linear SVM", X_train_flat, y_train, X_val_flat, y_val)

    # 2. RBF SVM (regularized)
    evaluate(
        SVC(kernel="rbf", C=0.01, gamma="scale"), "RBF SVM", X_train_flat, y_train, X_val_flat, y_val
    )

    # 3. Random Forest
    evaluate(
        RandomForestClassifier(n_estimators=100, max_depth=2, min_samples_leaf=20, random_state=42),
        "Random Forest",
        X_train_flat,
        y_train,
        X_val_flat,
        y_val,
    )

    # 4. MLP (sklearn)
    evaluate(
        MLPClassifier(
            hidden_layer_sizes=(16,), max_iter=500, alpha=20.0, early_stopping=True, validation_fraction=0.15, n_iter_no_change=5, random_state=42
        ),
        "MLP (sklearn)",
        X_train_flat,
        y_train,
        X_val_flat,
        y_val,
    )


if __name__ == "__main__":
    main()
