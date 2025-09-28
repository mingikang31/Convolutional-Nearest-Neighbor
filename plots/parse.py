import pandas as pd
import re
import os

def parse_training_results(file_path):
    """Parse training results from text file into a pandas DataFrame"""
    
    # Read the file
    with open(file_path, 'r') as f:
        content = f.read()
    
    # Initialize lists to store data
    epochs = []
    times = []
    train_losses = []
    train_acc_top1 = []
    train_acc_top5 = []
    test_losses = []
    test_acc_top1 = []
    test_acc_top5 = []
    
    # Regular expression to match each epoch line
    pattern = r'\[Epoch (\d+)\] Time: ([\d.]+)s.*?Train.*?Loss: ([\d.]+).*?Top1: ([\d.]+)%.*?Top5: ([\d.]+)%.*?Test.*?Loss: ([\d.]+).*?Top1: ([\d.]+)%.*?Top5: ([\d.]+)%'
    
    # Find all matches
    matches = re.findall(pattern, content)
    
    for match in matches:
        epochs.append(int(match[0]))
        times.append(float(match[1]))
        train_losses.append(float(match[2]))
        train_acc_top1.append(float(match[3]))
        train_acc_top5.append(float(match[4]))
        test_losses.append(float(match[5]))
        test_acc_top1.append(float(match[6]))
        test_acc_top5.append(float(match[7]))
    
    # Create DataFrame
    df = pd.DataFrame({
        'epoch': epochs,
        'time': times,
        'train_loss': train_losses,
        'train_accuracy_top1': train_acc_top1,
        'train_accuracy_top5': train_acc_top5,
        'test_loss': test_losses,
        'test_accuracy_top1': test_acc_top1,
        'test_accuracy_top5': test_acc_top5
    })
    
    return df


def parse_params_gflops(file_dir):
    """Parse training results from text file into a pandas DataFrame"""

    train_path = os.path.join(file_dir, "train_eval_results.txt")
    args_path = os.path.join(file_dir, "args.txt")
    
    # Read the file
    with open(train_path, 'r') as f:
        train_content = f.readline()

    with open(args_path, 'r') as f:
        args_content = f.readlines()

    total_params = args_content[-2].strip().split(": ")[1]
    trainable_params = args_content[-1].strip().split(": ")[1]

    gflops = re.search(r'GFLOPs: ([\d.]+)', train_content).group(1)

    return {
        "total_params": int(total_params),
        "trainable_params": int(trainable_params),
        "gflops": float(gflops)
    }



if __name__ == "__main__":
    pass