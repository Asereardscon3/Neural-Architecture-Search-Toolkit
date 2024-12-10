
import torch
import torch.nn as nn
import torch.optim as optim
import optuna
import logging
from typing import Tuple, List

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("NAS-Toolkit")

class SearchableSpace(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, trial: optuna.Trial):
        super(SearchableSpace, self).__init__()
        self.layers = nn.ModuleList()
        
        # Search for number of layers
        num_layers = trial.suggest_int("num_layers", 1, 5)
        current_dim = input_dim
        
        for i in range(num_layers):
            # Search for hidden dimension in each layer
            hidden_dim = trial.suggest_int(f"layer_{i}_dim", 32, 512)
            self.layers.append(nn.Linear(current_dim, hidden_dim))
            self.layers.append(nn.ReLU())
            
            # Optional Dropout search
            if trial.suggest_categorical(f"layer_{i}_dropout", [True, False]):
                dropout_rate = trial.suggest_float(f"layer_{i}_dropout_rate", 0.1, 0.5)
                self.layers.append(nn.Dropout(dropout_rate))
            
            current_dim = hidden_dim
            
        self.output_layer = nn.Linear(current_dim, output_dim)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return self.output_layer(x)

def train_and_evaluate(model, train_loader, val_loader, epochs=5):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(epochs):
        model.train()
        for data, target in train_loader:
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            
    model.eval()
    correct = 0
    with torch.no_grad():
        for data, target in val_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            
    return correct / len(val_loader.dataset)

def objective(trial):
    # This would normally be real data loaders
    # For demonstration, we'll use synthetic data
    input_dim = 100
    output_dim = 10
    
    # Generate synthetic data
    train_data = [(torch.randn(1, input_dim), torch.tensor([random.randint(0, 9)])) for _ in range(100)]
    val_data = [(torch.randn(1, input_dim), torch.tensor([random.randint(0, 9)])) for _ in range(20)]
    
    model = SearchableSpace(input_dim, output_dim, trial)
    accuracy = train_and_evaluate(model, train_data, val_data)
    return accuracy

if __name__ == "__main__":
    import random
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=20)
    
    logger.info(f"Best trial: {study.best_trial.params}")
    logger.info(f"Best accuracy: {study.best_value}")
