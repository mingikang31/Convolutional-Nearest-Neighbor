# In allconvnet_main.py

"""Main File for the project"""

import argparse 
from pathlib import Path
import os 

# Datasets 
from dataset import ImageNet, CIFAR10, CIFAR100
# *** NOTE: Ensure your updated train_eval.py is saved ***
from train_eval import Train_Eval 

# Models 
from allconvnet import AllConvNet 

# Utilities 
from utils import write_to_file, set_seed

import wandb

# The args_parser() and check_args() functions remain unchanged
def args_parser():
    parser = argparse.ArgumentParser(description="Convolutional Nearest Neighbor training and evaluation", add_help=False) 

    parser.add_argument("--experiment_name", type=str, default="ConvNN_Experiment", help="Name of the experiment for logging purposes")
    
    # Model Arguments
    parser.add_argument("--layer", type=str, default="ConvNN", choices=["Conv2d", "ConvNN", "ConvNN_Attn", "Attention", "Conv2d/ConvNN", "Conv2d/ConvNN_Attn", "Attention/ConvNN", "Attention/ConvNN_Attn", "Conv2d/Attention"], help="Type of Convolution or Attention layer to use")
    parser.add_argument("--num_layers", type=int, default=5, help="Number of layers.")   
    parser.add_argument("--channels", nargs='+', type=int, default=[32, 64, 128, 256, 512], help="Channel sizes for each layer.")
    
    # Additional Layer Arguments
    parser.add_argument("--K", type=int, default=9, help="K-nearest neighbor for ConvNN")
    parser.add_argument("--kernel_size", type=int, default=3, help="Kernel Size for Conv2d")        
    parser.add_argument("--sampling", type=str, default=None, choices=["All", "Random", "Spatial"], help="Sampling method for ConvNN Models")
    parser.add_argument("--num_samples", type=int, default=0, help="Number of samples for ConvNN Models")
    parser.add_argument("--num_heads", type=int, default=4, help="Number of heads for Attention Models")
    parser.add_argument("--shuffle_pattern", type=str, default="BA", choices=["BA", "NA"], help="Shuffle pattern: BA (Before & After) or NA (No Shuffle)")
    parser.add_argument("--shuffle_scale", type=int, default=2, help="Shuffle scale for ConvNN Models")
    parser.add_argument("--magnitude_type", type=str, default="similarity", choices=["similarity", "distance"], help="Magnitude type for ConvNN Models")
    parser.add_argument("--location_channels", action="store_true", help="Use location channels for ConvNN Models")
    parser.set_defaults(location_channels=False)
    
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
    
    # Output Arguments 
    parser.add_argument("--output_dir", type=str, default="./Output/Simple/ConvNN", help="Directory to save the output files")
    
    return parser

def check_args(args):
    # This function remains unchanged
    print("Checking arguments based on the model...")    
    assert args.layer in ["Conv2d", "ConvNN", "ConvNN_Attn", "Attention", "Conv2d/ConvNN", "Conv2d/ConvNN_Attn", "Attention/ConvNN", "Attention/ConvNN_Attn", "Conv2d/Attention"], f"Model {args.layer} not supported"
    assert args.dataset in ["cifar10", "cifar100", 'imagenet'], f"Dataset {args.dataset} not supported"
    assert args.criterion in ["CrossEntropy", "MSE"], f"Criterion {args.criterion} not supported"
    assert args.optimizer in ['adam', 'sgd', 'adamw'], f"Optimizer {args.optimizer} not supported"
    assert args.scheduler in ['step', 'cosine', 'plateau'], f"Scheduler {args.scheduler} not supported"
    assert args.num_layers == len(args.channels), f"Number of layers {args.num_layers} does not match the number of channels {len(args.channels)}"
    if args.sampling == "All":
        args.num_samples = 0
    if args.num_samples == 0:
        args.sampling = "All"
    args.resize = False
    return args
    
    
def main(args):

    args = check_args(args)
    
    # Initialize wandb as early as possible
    wandb.init(
        project="ConvNN: All ConvNet",
        name=args.experiment_name,
        config=vars(args) # Log all command-line arguments
    )
    
    # Check if the output directory exists, if not create it
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    
    # Dataset 
    if args.dataset == "cifar10":
        dataset = CIFAR10(args)
        # These are now part of the wandb config, no need to re-assign to args
        wandb.config.update({"num_classes": dataset.num_classes, "img_size": dataset.img_size})
    elif args.dataset == "cifar100":
        dataset = CIFAR100(args)
        wandb.config.update({"num_classes": dataset.num_classes, "img_size": dataset.img_size})
    elif args.dataset == "imagenet":
        dataset = ImageNet(args)
        wandb.config.update({"num_classes": dataset.num_classes, "img_size": dataset.img_size})
    else:
        raise ValueError("Dataset not supported")

    # Model 
    model = AllConvNet(args)
    print(f"Model: {model.name}")
    
    # Parameters - Log these directly to wandb summary
    total_params, trainable_params = model.parameter_count()
    print(f"Total Parameters: {total_params}")
    print(f"Trainable Parameters: {trainable_params}")
    wandb.summary['total_params'] = total_params
    wandb.summary['trainable_params'] = trainable_params
    
    # Set the seed for reproducibility
    if args.seed != 0:
        set_seed(args.seed)
    
    # Training Modules 
    # Capture the returned dictionary of summary metrics
    summary_results = Train_Eval(args, 
                                 model, 
                                 dataset.train_loader, 
                                 dataset.test_loader
                                )
    
    # Log the final, best metrics to the wandb summary
    wandb.summary.update(summary_results)

    # Storing Results in output directory remains possible
    print(f"Saving artifacts to {args.output_dir}")
    write_to_file(os.path.join(args.output_dir, "args.txt"), str(wandb.config))
    write_to_file(os.path.join(args.output_-dir, "model.txt"), str(model))
    # Write the summary dictionary to the results file
    write_to_file(os.path.join(args.output_dir, "summary_results.txt"), str(summary_results))

    # *** IMPORTANT: Finish the wandb run ***
    wandb.finish()


if __name__ == '__main__': 
    parser = argparse.ArgumentParser(description="Convolutional Nearest Neighbor training and evaluation", parents=[args_parser()])
    args = parser.parse_args()

    main(args)