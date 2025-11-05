# 1. Loss's for different LRs 
library(tidyverse)
library(zoo)  # For rollmean function

setwd("/Users/mingikang/Developer/Convolutional-Nearest-Neighbor/Final_Output/loss_test/csv/lr")

# Smoothing window size (adjust this for more/less smoothing)
SMOOTH_WINDOW <- 5  # Try values like 5, 10, 15, 20

datasets = c("CIFAR10", "CIFAR100")
lrs = c("1e-3", "1e-4", "1e-5", "5e-4", "5e-5")

for (dataset in datasets) {
  for (lr in lrs){
    path <- paste0("VGG11-", dataset, "_lr_", lr, ".csv") 
    print(path)
    
    # Loss <-> Accuracy Top1 
    plot_title = paste0(dataset, " Loss for lr = ", lr)
    
    save_path <- paste0("../../plots/Loss Plots/MA 5/VGG11-", dataset, "_lr_", lr, "_plot", ".png")  
    
    data <- read.csv(path)
    
    # Reorder the df
    df_long <- data %>%
      pivot_longer(
        cols = -epoch,
        names_to = c("Model", "Type", ".value"),
        names_sep = "_"
      ) %>%
      # Apply moving average smoothing
      group_by(Model, Type) %>%
      mutate(
        #Loss_smooth = rollmean(Loss, k = SMOOTH_WINDOW, fill = NA, align = "center") # Previous Implementation
        Loss_smooth = rollapply(Loss, width=SMOOTH_WINDOW, FUN = mean, align="right", fill=NA, partial=TRUE)
        # Loss_smooth = rollapply(Accuracy, width=SMOOTH_WINDOW, FUN = mean, align="right", fill=NA, partial=TRUE)
      ) %>%
      ungroup()
    
    # Create a single combined plot
    combined_plot <- df_long %>%
      # Use Loss_smooth instead of Loss
      ggplot(aes(x = epoch, y = Loss_smooth, color = Model, linetype = Type)) +
      geom_line(linewidth = 1.0) +
      
      scale_linetype_manual(
        name = "Loss Type", 
        values = c("Train" = "dotted", "Test" = "solid")
      ) +
      
      # --- Labels and Titles ---
      labs(
        title = plot_title,
        subtitle = "Comparison of Conv2d and Branching ConvNN",
        x = "Epochs",
        y = "Loss",
        color = "Model"
      ) +
      
      # --- Theme and Styling ---
      theme_bw(base_size = 10) +
      theme(
        legend.position = "right",
        plot.title = element_text(face = "bold", size = 23),
        plot.subtitle = element_text(size = 18),
        legend.title = element_text(face = "bold")
      )
    
    # Save the combined plot to a file
    ggsave(
      save_path,
      plot = combined_plot,
      width = 8,
      height = 5,
      units = "in",
      dpi = 300,
      bg = "white"
    )
    
  }
}