"""
Ensemble Model Architecture for Brain Tumor Classification
Combines Swin Transformer, DeiT, and ConvNeXt
Shared between FL Server and Clients
"""

import torch
import torch.nn as nn

# Import timm for pre-trained models
try:
    import timm
except ImportError:
    timm = None
    print("Warning: timm is not installed. Install it with `pip install timm` to use the ensemble model.")


class EnsembleModel(nn.Module):
    """
    Ensemble of multiple pre-trained models from timm library.
    Averages logits from each sub-model for final prediction.
    
    Models used:
    - Swin Transformer Small: Hierarchical transformer (local + global attention)
    - DeiT Base Distilled: Data-efficient Vision Transformer
    - ConvNeXt Small: Modern convolutional architecture
    """
    def __init__(self, model_names, num_classes, pretrained=True, device='cuda'):
        super(EnsembleModel, self).__init__()
        if timm is None:
            raise ImportError("timm is required to create ensemble models. Install with: pip install timm")
        
        self.device = device
        self.model_names = model_names
        self.submodels = nn.ModuleList()
        
        for name in model_names:
            try:
                # Create model with final classification head for num_classes
                m = timm.create_model(name, pretrained=pretrained, num_classes=num_classes)
                self.submodels.append(m)
                print(f"  ✓ Loaded {name}")
            except Exception as ex:
                raise RuntimeError(f"Failed to create timm model '{name}': {ex}")

    def forward(self, x):
        """
        Forward pass through all sub-models and average their logits.
        
        Args:
            x: Input tensor of shape (batch_size, 3, 224, 224)
        
        Returns:
            Averaged logits of shape (batch_size, num_classes)
        """
        logits = []
        for m in self.submodels:
            logits.append(m(x))
        
        # Stack: (n_models, batch_size, num_classes)
        stacked = torch.stack(logits, dim=0)
        # Average across models: (batch_size, num_classes)
        avg_logits = torch.mean(stacked, dim=0)
        return avg_logits


def build_model(num_classes=4, pretrained=True, device='cuda'):
    """
    Builds the ensemble model with predefined list of models.
    
    Args:
        num_classes: Number of output classes (default: 4 for brain tumor types)
        pretrained: Whether to use ImageNet pre-trained weights (default: True)
        device: Device to use ('cuda' or 'cpu')
    
    Returns:
        EnsembleModel instance
    """
    model_names = [
        "swin_small_patch4_window7_224",    # Swin Transformer
        "deit_base_distilled_patch16_224",  # DeiT
        "convnext_small"                     # ConvNeXt
    ]
    
    print("Building EnsembleModel with:", model_names)
    model = EnsembleModel(
        model_names=model_names, 
        num_classes=num_classes, 
        pretrained=pretrained, 
        device=device
    )
    return model
