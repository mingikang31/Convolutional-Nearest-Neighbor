"""Main File for VIT model"""

import argparse 
from pathlib import Path 
import os

# Datasets 
from dataset import ImageNet, CIFAR10, CIFAR100
# This uses the same updated Train_Eval function from the previous step
from train_eval import Train_Eval

# Models
from vit import ViT

# Utilities 
from utils import write_to_file, set_seed

import wandb

def args_parser():
    # This function remains unchanged
    parser = argparse.ArgumentParser(description="ViT training and evaluation", add_help=False) 

    parser.add_argument("--experiment_name", type=str, default="ViT_Experiment", help="Name of the experiment for logging purposes")

    # Model Arguments
    parser.add_argument("--layer", type=str, default="Attention", choices=["Attention", "ConvNN", "ConvNNAttention", "Conv1d", "Conv1dAttention"], help="Layer to use for training and evaluation")
    
    parser.add_argument("--patch_size", type=int, default=16, help="Patch size for Attention Models")
    parser.add_argument("--num_layers", type=int, default=8, help="Number of layers in the model")   
    parser.add_argument("--num_heads", type=int, default=4, help="Number of heads for Attention Models")
    parser.add_argument("--d_model", type=int, default=512, help="Dimensionality of the model for Attention Models")
    
    parser.add_argument("--dropout", type=float, default=0.1, help="Dropout rate for the model")
    parser.add_argument("--attention_dropout", type=float, default=0.1, help="Dropout rate for the model")
    
    # Additional Layer Arguments for ConvNN
    parser.add_argument("--K", type=int, default=9, help="K-nearest neighbor for ConvNN Layer")
    parser.add_argument("--kernel_size", type=int, default=3, help="Kernel size for Conv1d Layer")
    parser.add_argument("--num_samples", type=int, default=0, help="Number of samples for ConvNN Layer, -1 for all samples")
    parser.add_argument("--magnitude_type", type=str, default="similarity", choices=["similarity", "distance"], help="Magnitude type for ConvNN Models")
    
    # Arguments for Data 
    parser.add_argument("--dataset", type=str, default="cifar10", choices=["cifar10", "cifar100", 'imagenet'], help="Dataset to use for training and evaluation")
    parser.add_argument("--data_path", type=str, default="./Data", help="Path to the dataset")
        
    # Training Arguments
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size for training and evaluation")
    parser.add_argument("--num_epochs", type=int, default=100, help="Number of epochs for training")
    parser.add_argument("--use_amp", action="store_true", help="Use mixed precision training")
    parser.set_defaults(use_amp=False)
    parser.add_argument("--clip_grad_norm", type=float, default=None, help="Gradient clipping value")
    
    # Loss Function Arguments
    parser.add_argument("--criterion", type=str, default="CrossEntropy", choices=["CrossEntropy", "MSE"], help="Loss function to use for training")
    
    # Optimizer Arguments 
    parser.add_argument('--optimizer', type=str, default='adamw', choices=['adam', 'sgd', 'adamw'], help='Default Optimizer: adamw')
    parser.add_argument('--momentum', type=float, default=0.9, help='Momentum for SGD optimizer')
    parser.add_argument('--weight_decay', type=float, default=1e-6, help='Weight decay for optimizer')
    
    # Learning Rate Arguments
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate for the optimizer")
    parser.add_argument('--lr_step', type=int, default=20, help='Step size for learning rate scheduler')
    parser.add_argument('--lr_gamma', type=float, default=0.1, help='Gamma for learning rate scheduler')
    parser.add_argument('--scheduler', type=str, default='step', choices=['step', 'cosine', 'plateau'], help='Learning rate scheduler')
    
    # Device Arguments
    parser.add_argument("--device", type=str, default="cuda", choices=["cpu", "cuda", "mps"], help="Device to use for training and evaluation")
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument("--num_workers", type=int, default=1, help="Number of workers for Dataloader")

    # Output Arguments 
    parser.add_argument("--output_dir", type=str, default="./Output/VIT/VIT_Attention", help="Directory to save the output files")
    
    return parser
    
def main(args):
    
    # --- 1. Initialize wandb as early as possible ---
    wandb.init(
        project="ConvNN ViT", # A specific project name for ViT experiments
        name=args.experiment_name,
        config=vars(args) # Log all command-line arguments
    )
    
    # Check if the output directory exists, if not create it
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    # This flag might be needed by your dataset loader
    args.resize = True

    # --- 2. Load Dataset and Update wandb Config ---
    if args.dataset == "cifar10":
        dataset = CIFAR10(args)
        # Update config with values determined after arg parsing (e.g., from the dataset)
        wandb.config.update({"num_classes": dataset.num_classes, "img_size": dataset.img_size})
    elif args.dataset == "cifar100":
        dataset = CIFAR100(args)
        wandb.config.update({"num_classes": dataset.num_classes, "img_size": dataset.img_size})
    elif args.dataset == "imagenet":
        dataset = ImageNet(args)
        wandb.config.update({"num_classes": dataset.num_classes, "img_size": dataset.img_size})
    else:
        raise ValueError("Dataset not supported")
    
    # --- 3. Initialize Model and Log Parameters ---
    model = ViT(args)
    print(f"Model: {model.name}")
    
    # Parameters
    total_params, trainable_params = model.parameter_count()
    print(f"Total Parameters: {total_params}")
    print(f"Trainable Parameters: {trainable_params}")
    # Log calculated parameters to the wandb summary for easy access
    wandb.summary['total_params'] = total_params
    wandb.summary['trainable_params'] = trainable_params
    
    # Set the seed for reproducibility
    if args.seed != 0:
        set_seed(args.seed)
    
    # --- 4. Run Training and Capture Final Results ---
    summary_results = Train_Eval(args, 
                                 model, 
                                 dataset.train_loader, 
                                 dataset.test_loader
                                )
    
    # Log the final, best metrics to the wandb summary
    wandb.summary.update(summary_results)
    
    # --- 5. Store Local Artifacts ---
    print(f"Saving artifacts to {args.output_dir}")
    write_to_file(os.path.join(args.output_dir, "args.txt"), str(wandb.config))
    write_to_file(os.path.join(args.output_dir, "model.txt"), str(model))
    # Write the summary dictionary to the results file
    write_to_file(os.path.join(args.output_dir, "summary_results.txt"), str(summary_results))

    # --- 6. IMPORTANT: Finish the wandb run ---
    wandb.finish()


if __name__ == '__main__': 
    parser = argparse.ArgumentParser(description="ViT training and evaluation", parents=[args_parser()])
    args = parser.parse_args()

    main(args)