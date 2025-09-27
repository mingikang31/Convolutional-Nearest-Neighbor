import pandas as pd
import matplotlib.pyplot as plt

def parse_results(file_path):


if __name__ == "__main__":
    file_path = 'Output/Sep_24_Branching_NoSplit/vgg_1e-5_cos/CIFAR10/LocCol_LocCol_Branch/ConvBranch_K9_r000_s42'

    args_path = file_path + '/args.txt'
    model_path = 

    
    df = parse_results(file_path)
    
    # Plotting
    plt.figure(figsize=(10, 6))
    for method in df['Method'].unique():
        subset = df[df['Method'] == method]
        plt.plot(subset['Parameter'], subset['Metric1'], marker='o', label=method)
    
    plt.title('Performance Comparison')
    plt.xlabel('Parameter')
    plt.ylabel('Metric1')
    plt.legend()
    plt.grid(True)
    plt.savefig('performance_comparison.png')
    plt.show()