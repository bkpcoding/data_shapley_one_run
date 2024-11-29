import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Optional
import torch.nn.functional as F
from train import MNISTTrainer

class GhostDotProduct:
    """Implements efficient gradient dot product calculation using the ghost dot-product technique."""
    
    def __init__(self, model: nn.Module, criterion: nn.Module):
        self.model = model
        self.criterion = criterion
        self.hooks = []
        self.activation_dict = {}
        self.gradient_dict = {}
        
    def _register_hooks(self):
        """Register forward and backward hooks to capture activations and gradients."""
        def forward_hook(module, input, output):
            self.activation_dict[id(module)] = input[0].detach()
            
        def backward_hook(module, grad_input, grad_output):
            self.gradient_dict[id(module)] = grad_output[0].detach()
            
        # Register hooks for all relevant layers
        for name, module in self.model.named_modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                self.hooks.append(module.register_forward_hook(forward_hook))
                self.hooks.append(module.register_backward_hook(backward_hook))
    
    def _remove_hooks(self):
        """Remove all registered hooks."""
        for hook in self.hooks:
            hook.remove()
        self.hooks = []

    def compute_batch_gradient_dots(self, 
                                batch_data: torch.Tensor, 
                                batch_targets: torch.Tensor,
                                validation_data: torch.Tensor,
                                validation_target: torch.Tensor) -> torch.Tensor:
        """
        Compute gradient dot products between training batch and validation sample.
        Returns a tensor of shape (batch_size,) containing dot products.
        """
        self._register_hooks()
        device = next(self.model.parameters()).device
        batch_size = batch_data.shape[0]
        dot_products = torch.zeros(batch_size, device=device)
        
        # Individual backward passes for each training sample
        batch_grads = []
        for i in range(batch_size):
            self.model.zero_grad()
            output_i = self.model(batch_data[i:i+1])
            loss_i = self.criterion(output_i, batch_targets[i:i+1])
            loss_i.backward(retain_graph=True)
            
            # Store gradients for this sample
            grads_i = {}
            for name, param in self.model.named_parameters():
                if param.grad is not None:
                    grads_i[name] = param.grad.detach().clone()
            batch_grads.append(grads_i)
            self.model.zero_grad()
        
        # Validation sample backward pass
        val_output = self.model(validation_data)
        val_loss = self.criterion(val_output, validation_target)
        val_loss.backward()
        
        # Compute dot products
        for i in range(batch_size):
            dot_product = 0
            for name, param in self.model.named_parameters():
                if param.grad is not None:
                    val_grad = param.grad.detach().flatten()
                    batch_grad = batch_grads[i][name].flatten()
                    dot_product += torch.dot(batch_grad, val_grad)
            dot_products[i] = dot_product
        
        self._remove_hooks()
        self.model.zero_grad()
        return dot_products


class InRunDataShapley:
    """Tracks and computes In-Run Data Shapley values during training."""
    
    def __init__(self, model: nn.Module, criterion: nn.Module):
        self.model = model
        self.criterion = criterion
        self.ghost_dot = GhostDotProduct(model, criterion)
        self.shapley_values: Dict[int, float] = {}  # Maps data index to cumulative Shapley value
        self.iteration_count: Dict[int, int] = {}   # Tracks number of times each data point is used
        
    def update_shapley_values(self,
                            batch_data: torch.Tensor,
                            batch_targets: torch.Tensor,
                            batch_indices: torch.Tensor,
                            validation_data: torch.Tensor,
                            validation_target: torch.Tensor,
                            learning_rate: float):
        """Update Shapley values for the current batch."""
        # Compute gradient dot products
        dot_products = self.ghost_dot.compute_batch_gradient_dots(
            batch_data, batch_targets, validation_data, validation_target
        )
        
        # Update Shapley values
        for idx, dot_product in zip(batch_indices, dot_products):
            idx = idx.item()
            shapley_value = -learning_rate * dot_product.item()  # First-order approximation
            
            if idx not in self.shapley_values:
                self.shapley_values[idx] = 0
                self.iteration_count[idx] = 0
                
            self.shapley_values[idx] += shapley_value
            self.iteration_count[idx] += 1
            
    def get_data_values(self) -> Dict[int, float]:
        """Return the current averaged Shapley values for all seen data points."""
        return {
            idx: value / self.iteration_count[idx]
            for idx, value in self.shapley_values.items()
        }
    
    def get_top_valuable_indices(self, k: int) -> List[int]:
        """Return indices of top-k most valuable data points."""
        values = self.get_data_values()
        return sorted(values.keys(), key=lambda x: values[x], reverse=True)[:k]
    
    def get_negative_value_indices(self) -> List[int]:
        """Return indices of data points with negative values."""
        values = self.get_data_values()
        return [idx for idx, value in values.items() if value < 0]

class ShapleyMNISTTrainer(MNISTTrainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.shapley_tracker = InRunDataShapley(self.model, self.criterion)
        
    def get_visualization_data(self):
        # Get validation example
        val_data, _ = next(iter(self.val_loader))
        validation_example = {
            'image': val_data[0].squeeze().numpy().tolist()
        }
        
        # Get Shapley values
        shapley_values = self.shapley_tracker.get_data_values()
        indices = list(shapley_values.keys())
        values = list(shapley_values.values())
        
        # Sort indices by values
        sorted_indices = [x for _, x in sorted(zip(values, indices), reverse=True)]
        
        # Get top 5 positive examples
        top_positive = []
        for idx in sorted_indices[:5]:
            # Access the original dataset through the Subset's indices
            dataset_idx = self.train_dataset.indices[idx]
            image = self.train_dataset.dataset.data[dataset_idx].numpy() / 255.0
            top_positive.append({
                'image': image.tolist(),
                'value': shapley_values[idx]
            })
        
        # Get top 5 negative examples
        top_negative = []
        for idx in sorted_indices[-5:]:
            # Access the original dataset through the Subset's indices
            dataset_idx = self.train_dataset.indices[idx]
            image = self.train_dataset.dataset.data[dataset_idx].numpy() / 255.0
            top_negative.append({
                'image': image.tolist(),
                'value': shapley_values[idx]
            })
        
        return {
            'validationExample': validation_example,
            'topPositive': top_positive,
            'topNegative': top_negative
        }
    
    def train_epoch(self):
        self.model.train()
        total_loss = 0
        correct = 0
        
        # Get one validation sample for Shapley calculation
        val_data, val_target = next(iter(self.val_loader))
        val_data = val_data[0:1].to(self.device)  # Take just one sample
        val_target = val_target[0:1].to(self.device)
        
        for batch_idx, (data, target) in enumerate(self.train_loader):
            data, target = data.to(self.device), target.to(self.device)
            
            # Regular training step
            self.optimizer.zero_grad()
            output = self.model(data)
            losses = self.criterion(output, target)
            loss = losses.mean()
            loss.backward()
            
            # Update Shapley values before optimizer step
            self.shapley_tracker.update_shapley_values(
                data, target,
                torch.arange(batch_idx * self.batch_size, 
                           min((batch_idx + 1) * self.batch_size, len(self.train_dataset))),
                val_data, val_target,
                self.optimizer.param_groups[0]['lr']
            )
            
            self.optimizer.step()
            
            # Compute accuracy
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            total_loss += loss.item()
            
        avg_loss = total_loss / len(self.train_loader)
        accuracy = 100. * correct / len(self.train_loader.dataset)
        return avg_loss, accuracy

import matplotlib.pyplot as plt
def plot_shapley_analysis(visualization_data):
    """
    Plot validation example alongside top positive and negative contributors.
    
    Args:
        visualization_data: Dictionary containing:
            - validationExample: Dict with 'image' key
            - topPositive: List of dicts with 'image' and 'value' keys
            - topNegative: List of dicts with 'image' and 'value' keys
    """
    # Create figure with gridspec
    fig = plt.figure(figsize=(15, 8))
    gs = fig.add_gridspec(3, 6, hspace=0.3, wspace=0.3)
    
    # Plot validation example (center top)
    ax_val = fig.add_subplot(gs[0, 2:4])
    ax_val.imshow(visualization_data['validationExample']['image'], cmap='gray')
    ax_val.set_title('Validation Example', pad=10)
    ax_val.axis('off')
    
    # Plot top positive examples
    fig.text(0.3, 0.65, 'Top Positive Contributors', fontsize=12, fontweight='bold')
    for i, example in enumerate(visualization_data['topPositive']):
        ax = fig.add_subplot(gs[1, i])
        ax.imshow(example['image'], cmap='gray')
        ax.set_title(f'Value: {example["value"]:.4f}', fontsize=8)
        ax.axis('off')
    
    # Plot top negative examples
    fig.text(0.3, 0.35, 'Top Negative Contributors', fontsize=12, fontweight='bold')
    for i, example in enumerate(visualization_data['topNegative']):
        ax = fig.add_subplot(gs[2, i])
        ax.imshow(example['image'], cmap='gray')
        ax.set_title(f'Value: {example["value"]:.4f}', fontsize=8)
        ax.axis('off')
    
    plt.suptitle('MNIST Shapley Value Analysis', fontsize=14, fontweight='bold', y=0.95)
    plt.savefig("./data_shapley.png")
    
    # Close the figure to free memory
    plt.close(fig)


if __name__ == "__main__":
    trainer = ShapleyMNISTTrainer(batch_size=64, val_split=0.1)
    history = trainer.train(epochs=5)

    # Analyze data values
    shapley_values = trainer.shapley_tracker.get_data_values()
    top_indices = trainer.shapley_tracker.get_top_valuable_indices(k=100)
    negative_indices = trainer.shapley_tracker.get_negative_value_indices()
    vis = trainer.get_visualization_data()
    print(len(vis['validationExample']['image']))
    print(len(vis['topPositive']))
    print(len(vis['topNegative']))
    plot_shapley_analysis(vis)
