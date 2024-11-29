import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
import numpy as np

class SimpleConvNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        return x

class MNISTTrainer:
    def __init__(self, batch_size=64, val_split=0.1, random_seed=140):
        self.batch_size = batch_size
        self.val_split = val_split
        self.random_seed = random_seed
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Set random seeds for reproducibility
        torch.manual_seed(random_seed)
        np.random.seed(random_seed)
        
        self.setup_data()
        self.model = SimpleConvNet().to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters())
        self.criterion = nn.CrossEntropyLoss(reduction='none')  # Use 'none' to get per-sample losses
        
    def setup_data(self):
        # Load and normalize MNIST dataset
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        
        # Load full training set
        full_dataset = datasets.MNIST('data', train=True, download=True, transform=transform)
        
        # Split into train and validation
        val_size = int(len(full_dataset) * self.val_split)
        train_size = len(full_dataset) - val_size
        
        self.train_dataset, self.val_dataset = random_split(
            full_dataset, [train_size, val_size]
        )
        
        # Create data loaders
        self.train_loader = DataLoader(
            self.train_dataset, 
            batch_size=self.batch_size, 
            shuffle=False
        )
        self.val_loader = DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False
        )

    def train_epoch(self):
        self.model.train()
        total_loss = 0
        correct = 0
        
        for batch_idx, (data, target) in enumerate(self.train_loader):
            data, target = data.to(self.device), target.to(self.device)
            self.optimizer.zero_grad()
            
            # Forward pass
            output = self.model(data)
            
            # Compute per-sample losses
            losses = self.criterion(output, target)
            loss = losses.mean()
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            
            # Compute accuracy
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            total_loss += loss.item()
            
        avg_loss = total_loss / len(self.train_loader)
        accuracy = 100. * correct / len(self.train_loader.dataset)
        return avg_loss, accuracy

    def validate(self):
        self.model.eval()
        val_loss = 0
        correct = 0
        
        with torch.no_grad():
            for data, target in self.val_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                val_loss += self.criterion(output, target).mean().item()
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()
        
        val_loss /= len(self.val_loader)
        accuracy = 100. * correct / len(self.val_loader.dataset)
        return val_loss, accuracy

    def train(self, epochs=10):
        train_losses = []
        train_accuracies = []
        val_losses = []
        val_accuracies = []
        
        for epoch in range(epochs):
            # Train
            train_loss, train_acc = self.train_epoch()
            train_losses.append(train_loss)
            train_accuracies.append(train_acc)
            
            # Validate
            val_loss, val_acc = self.validate()
            val_losses.append(val_loss)
            val_accuracies.append(val_acc)
            
            print(f'Epoch {epoch+1}/{epochs}:')
            print(f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%')
            print(f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%\n')
        
        return {
            'train_losses': train_losses,
            'train_accuracies': train_accuracies,
            'val_losses': val_losses,
            'val_accuracies': val_accuracies
        }

# Example usage
if __name__ == "__main__":
    trainer = MNISTTrainer(batch_size=64, val_split=0.1)
    history = trainer.train(epochs=5)