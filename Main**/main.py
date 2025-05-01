"""Main File for the project"""

import argparse 
from pathlib import Path
import os 

# Datasets 
from dataset import ImageNet, CIFAR10, CIFAR100, MNIST
from train_eval import Train_Eval

# Models 
from models import ClassificationModel 

# Utilities 
from utils import write_to_file, set_seed


def args_parser():
    parser = argparse.ArgumentParser(description="Convolutional Nearest Neighbor training and evaluation", add_help=False) 
    
    # Model Arguments
    parser.add_argument("--model", type=str, default="ConvNN", choices=["Conv2d", "ConvNN", "ConvNN_Attn", "Attention", "Conv2d/ConvNN", "Conv2d/ConvNN_Attn", "Attention/ConvNN", "Attention/ConvNN_Attn", "Conv2d/Attention"], help="Model to use for training and evaluation")
    parser.add_argument("--num_layers", type=int, default=4, help="Number of layers in the model")   
    parser.add_argument("--hidden_dim", type=int, default=16, help="Hidden dimension for the model")
    
    # Additional Layer Arguments
    parser.add_argument("--k_kernel", type=int, default=9, help="Kernel Size for Conv2d or K for ConvNN (K = number of nearest neighbors)")
    parser.add_argument("--sampling", type=str, default="All", choices=["All", "Random", "Spatial"], help="Sampling method for ConvNN Models")
    parser.add_argument("--num_samples", type=str, default="64", help="Number of samples for ConvNN Models")
    parser.add_argument("--num_heads", type=int, default=4, help="Number of heads for Attention Models")
    parser.add_argument("--shuffle_pattern", type=str, default="BA", choices=["BA", "NA"], help="Shuffle pattern: BA (Before & After) or NA (No Shuffle)")
    parser.add_argument("--shuffle_scale", type=int, default=2, help="Shuffle scale for ConvNN Models")
    parser.add_argument("--magnitude_type", type=str, default="similarity", choices=["similarity", "distance"], help="Magnitude type for ConvNN Models")
    parser.add_argument("--location_channels", action="store_true", help="Use location channels for ConvNN Models")
    parser.set_defaults(location_channels=False)
    
    

    
    # Arguments for Data 
    parser.add_argument("--dataset", type=str, default="cifar10", choices=["cifar10", "cifar100", "MNIST"], help="Dataset to use for training and evaluation")
    parser.add_argument("--data_path", type=str, default="./Data", help="Path to the dataset")
        
    # Training Arguments
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size for training and evaluation")
    parser.add_argument("--num_epochs", type=int, default=100, help="Number of epochs for training")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate for the optimizer")
    parser.add_argument("--criterion", type=str, default="CrossEntropy", choices=["CrossEntropy", "MSE"], help="Loss function to use for training")
    parser.add_argument('--opt', type=str, default='adamw', choices=['adam', 'sgd', 'adamw'], help='Default Optimizer: adamw')
    
    parser.add_argument("--device", type=str, default="cuda", choices=["cpu", "cuda", "mps"], help="Device to use for training and evaluation")
    parser.add_argument('--seed', default=0, type=int)
    
    # Output Arguments 
    parser.add_argument("--output_dir", type=str, default="./Output/ConvNN", help="Directory to save the output files")
    
    return parser
### Come back to this later 
def check_args(args):
    # Check the arguments based on the model 
    print("Checking arguments based on the model...")    
    
    if args.sampling == "All": # only for Conv2d_NN, Conv2d_NN_Attn
        args.num_samples = "all"
        
    
    return args
    
    
def main(args):

    # args = check_args(args)
    
    # Check if the output directory exists, if not create it
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    
    # Dataset 
    if args.dataset == "cifar10":
        dataset = CIFAR10(args)
        args.num_classes = dataset.num_classes 
        args.img_size = dataset.img_size 
    elif args.dataset == "cifar100":
        dataset = CIFAR100(args)
        args.num_classes = dataset.num_classes 
        args.img_size = dataset.img_size 
    elif args.dataset == "MNIST":
        dataset = MNIST(args)
        args.num_classes = dataset.num_classes 
        args.img_size = dataset.img_size 
    else:
        raise ValueError("Dataset not supported")
    
    
    # Model 
    model = ClassificationModel(args)
    print(f"Model: {model.name}")
    
    # Parameters
    total_params, trainable_params = model.parameter_count()
    print(f"Total Parameters: {total_params}")
    print(f"Trainable Parameters: {trainable_params}")
    args.total_params = total_params
    args.trainable_params = trainable_params
    
    # Set the seed for reproducibility
    set_seed(args.seed)
    
    
    # Training Modules 
    train_eval_results = Train_Eval(args, 
                                model, 
                                dataset.train_loader, 
                                dataset.test_loader
                                )
    
    # Storing Results in output directory 
    write_to_file(os.path.join(args.output_dir, "args.txt"), args)
    write_to_file(os.path.join(args.output_dir, "model.txt"), model)
    write_to_file(os.path.join(args.output_dir, "train_eval_results.txt"), train_eval_results)

if __name__ == '__main__': 
    parser = argparse.ArgumentParser(description="Convolutional Nearest Neighbor training and evaluation", parents=[args_parser()])
    args = parser.parse_args()

    main(args)
    
