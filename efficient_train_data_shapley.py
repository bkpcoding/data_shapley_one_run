import torch
import torch.nn as nn
from typing import Dict, List, Tuple
from train import MNISTTrainer
from opacus.grad_sample import GradSampleModule
import copy


class EfficientGhostDotProduct:
    def __init__(self, model: nn.Module, criterion: nn.Module):
        self.original_model = model  # Wrap with GradSampleModule
        self.criterion = criterion

    def _remove_hooks(self):
        for handle in self.handles:
            handle.remove()
        self.handles = []
        self.activations = {}
        self.output_grads = {}
        
    def _register_hooks(self):
        def save_activation(name):
            def hook(module, inputs):
                if inputs[0].requires_grad:
                    self.activations[name] = inputs[0]
                return inputs
            return hook
            
        def save_output_grad(name):
            def hook(module, grad_input, grad_output):
                self.output_grads[name] = grad_output[0]
            return hook
        
        self.activations.clear()
        self.output_grads.clear()
        
        for name, module in self.model.named_modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                self.handles.append(module.register_forward_pre_hook(save_activation(name)))
                self.handles.append(module.register_backward_hook(save_output_grad(name)))

    def compute_efficient_dot_products(self, 
                                     batch_data: torch.Tensor,
                                     batch_targets: torch.Tensor, 
                                     validation_data: torch.Tensor,
                                     validation_target: torch.Tensor) -> torch.Tensor:
        device = next(self.original_model.parameters()).device
        batch_size = batch_data.shape[0]
        
        # Clone model and wrap with GradSampleModule for Shapley computation
        with torch.no_grad():
            model_clone = copy.deepcopy(self.original_model)
            model_clone = GradSampleModule(model_clone)
            model_clone.to(device)
            
        dot_products = torch.zeros(batch_size, device=device)
        
        # First backward pass for validation sample
        model_clone.zero_grad()
        val_output = model_clone(validation_data)
        val_loss = self.criterion(val_output, validation_target).mean()
        val_loss.backward()
        
        # Store validation gradients
        val_grads = {}
        for name, param in model_clone.named_parameters():
            if hasattr(param, "grad_sample"):
                val_grads[name] = param.grad_sample.clone()
        
        # Second backward pass for batch
        model_clone.zero_grad()
        batch_output = model_clone(batch_data)
        batch_loss = self.criterion(batch_output, batch_targets).mean()
        batch_loss.backward()
        
        # Compute dot products using grad_sample
        for name, param in model_clone.named_parameters():
            if hasattr(param, "grad_sample"):
                batch_grad = param.grad_sample
                val_grad = val_grads[name]
                
                batch_flat = batch_grad.reshape(batch_size, -1)
                val_flat = val_grad.reshape(1, -1)
                
                dot_products += torch.mm(batch_flat, val_flat.t()).squeeze()
        
        # Clean up
        del model_clone
        torch.cuda.empty_cache()
        
        return dot_products

class EfficientInRunDataShapley:
    def __init__(self, model: nn.Module, criterion: nn.Module):
        self.model = model
        self.criterion = criterion
        self.ghost_dot = EfficientGhostDotProduct(model, criterion)
        self.shapley_values: Dict[int, float] = {}
        self.iteration_count: Dict[int, int] = {}
        
    def update_shapley_values(self,
                            batch_data: torch.Tensor,
                            batch_targets: torch.Tensor,
                            batch_indices: torch.Tensor,
                            validation_data: torch.Tensor,
                            validation_target: torch.Tensor,
                            learning_rate: float):
        dot_products = self.ghost_dot.compute_efficient_dot_products(
            batch_data, batch_targets, validation_data, validation_target
        )
        
        for idx, dot_product in zip(batch_indices, dot_products):
            idx = idx.item()
            shapley_value = -learning_rate * dot_product.item()
            
            if idx not in self.shapley_values:
                self.shapley_values[idx] = 0
                self.iteration_count[idx] = 0
                
            self.shapley_values[idx] += shapley_value
            self.iteration_count[idx] += 1
    
    def get_data_values(self) -> Dict[int, float]:
        return {
            # idx: value / self.iteration_count[idx]
            idx: value
            for idx, value in self.shapley_values.items()
        }


class EfficientShapleyMNISTTrainer(MNISTTrainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # self.model = GradSampleModule(self.model)  # Wrap the model
        self.shapley_tracker = EfficientInRunDataShapley(self.model, self.criterion)
        

    def train_epoch(self):
        self.model.train()
        total_loss = 0
        correct = 0
        
        # val_data, val_target = next(iter(self.val_loader))
        val_data, val_target = self.val_loader.dataset[25]

        # val_data = val_data[0:1].to(self.device)
        # val_target = val_target[0:1].to(self.device)
        val_data = val_data.unsqueeze(0).to(self.device)
        val_target = torch.tensor([val_target], device=self.device)
         
        for batch_idx, (data, target) in enumerate(self.train_loader):
            data, target = data.to(self.device), target.to(self.device)
            
            self.optimizer.zero_grad()
            output = self.model(data)
            loss = self.criterion(output, target).mean()
            loss.backward()
            
            # Compute Shapley values before optimizer step
            self.shapley_tracker.update_shapley_values(
                data, target,
                torch.arange(batch_idx * self.batch_size, 
                           min((batch_idx + 1) * self.batch_size, len(self.train_dataset))),
                val_data, val_target,
                self.optimizer.param_groups[0]['lr']
            )
            
            self.optimizer.step()
            
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            total_loss += loss.item()
        
        avg_loss = total_loss / len(self.train_loader)
        accuracy = 100. * correct / len(self.train_loader.dataset)
        return avg_loss, accuracy
       
    def get_visualization_data(self):
        # Get validation example
        # val_data, _ = next(iter(self.val_loader))
        val_data, val_target = self.val_loader.dataset[25]
        val_data = val_data.unsqueeze(0)
        val_target = torch.tensor([val_target])
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


import torch
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter
import time
import argparse
import wandb

def train_model(args):
    # Initialize wandb run
    if args.log:
        run = wandb.init(
            project="shapley-mnist",
            name=f"shapley_mnist_{time.strftime('%Y%m%d-%H%M%S')}",
            config={
                "batch_size": args.batch_size,
                "val_split": args.val_split,
                "epochs": args.epochs,
                "viz_interval": args.viz_interval
            }
        )
    
    # Initialize trainer
    trainer = EfficientShapleyMNISTTrainer(
        batch_size=args.batch_size,
        val_split=args.val_split
    )
    
    # Training loop with metrics tracking
    for epoch in range(args.epochs):
        train_loss, train_acc = trainer.train_epoch()
        val_loss, val_acc = trainer.validate()
        
        # Log metrics to wandb
        if args.log:
            wandb.log({
                "train/loss": train_loss,
                "train/accuracy": train_acc,
                "val/loss": val_loss,
                "val/accuracy": val_acc,
                "epoch": epoch
            })
        
        print(f'Epoch {epoch+1}/{args.epochs}:')
        print(f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%')
        print(f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%\n')
        
        # Generate visualization every few epochs
        if (epoch + 1) % args.viz_interval == 0:
            vis_data = trainer.get_visualization_data()
            fig = plot_shapley_analysis(vis_data)
            # Log the visualization to wandb
            # wandb.log({
            #     "shapley_visualization": wandb.Image(fig),
            #     "epoch": epoch
            # })
            
            # Log Shapley statistics
            values = trainer.shapley_tracker.get_data_values()
            if args.log:
                wandb.log({
                    "shapley_values_histogram": wandb.Histogram(
                        list(values.values())
                    ),
                    "epoch": epoch
                })
    
    # Close wandb run
    if args.log:
        wandb.finish()
    return trainer

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


def plot_shapley_values_by_class(trainer: EfficientShapleyMNISTTrainer):
    """
    Creates visualization of Shapley values distribution across different MNIST classes.
    
    Args:
        trainer: Trained EfficientShapleyMNISTTrainer instance
    """
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    # Get Shapley values and corresponding labels
    shapley_values = trainer.shapley_tracker.get_data_values()
    
    # Create lists to store values and labels
    values = []
    labels = []
    
    # Collect values and labels
    for idx, value in shapley_values.items():
        # Get original dataset index through the Subset's indices
        dataset_idx = trainer.train_dataset.indices[idx]
        # Get label from the original dataset
        label = trainer.train_dataset.dataset.targets[dataset_idx].item()
        values.append(value)
        labels.append(label)
    
    # Convert to numpy arrays
    values = np.array(values)
    labels = np.array(labels)
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 12))
    
    # Box plot
    sns.boxplot(x=labels, y=values, ax=ax1)
    ax1.set_title('Distribution of Shapley Values by Class')
    ax1.set_xlabel('Digit Class')
    ax1.set_ylabel('Shapley Value')
    
    # Calculate and plot mean values
    mean_values = [np.mean(values[labels == i]) for i in range(10)]
    std_values = [np.std(values[labels == i]) for i in range(10)]
    
    # Bar plot with error bars
    ax2.bar(range(10), mean_values, yerr=std_values, capsize=5)
    ax2.set_title('Mean Shapley Values by Class')
    ax2.set_xlabel('Digit Class')
    ax2.set_ylabel('Mean Shapley Value')
    
    # Add value labels on top of bars
    for i, v in enumerate(mean_values):
        ax2.text(i, v + std_values[i], f'{v:.2e}', ha='center', va='bottom')
    
    # Calculate and print additional statistics
    class_counts = [np.sum(labels == i) for i in range(10)]
    print("\nClass Statistics:")
    print("================")
    for digit in range(10):
        print(f"\nDigit {digit}:")
        class_values = values[labels == digit]
        print(f"Count: {class_counts[digit]}")
        print(f"Mean Shapley value: {mean_values[digit]:.2e}")
        print(f"Std dev: {std_values[digit]:.2e}")
        print(f"Min value: {np.min(class_values):.2e}")
        print(f"Max value: {np.max(class_values):.2e}")
    
    plt.tight_layout()
    plt.savefig("shapley_by_class.png")
    plt.close()
    
    return {
        'mean_values': mean_values,
        'std_values': std_values,
        'class_counts': class_counts
    }

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--val_split', type=float, default=0.1)
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--viz_interval', type=int, default=5)
    parser.add_argument('--seed', type=int, default=149)
    parser.add_argument('--log', type=bool, default=False)
    
    args = parser.parse_args()
    
    # Set random seeds
    torch.manual_seed(args.seed)
    
    # Train model
    trainer = train_model(args)
    
    # Final analysis
    print("\nAnalyzing data values...")
    shapley_values = trainer.shapley_tracker.get_data_values()
    stats = plot_shapley_values_by_class(trainer)
    print(f"Stats: ", stats)
    
    # Save final model and Shapley values
    torch.save({
        'model_state': trainer.model.state_dict(),
        'shapley_values': shapley_values,
    }, 'mnist_shapley_final.pt')
    
    print("Training complete. Model and Shapley values saved.")