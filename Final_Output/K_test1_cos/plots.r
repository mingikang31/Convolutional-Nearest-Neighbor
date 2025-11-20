library(tidyverse)
# The stringr package (part of tidyverse) is used for str_detect()
library(stringr)

# --- Data Reading ---
# Set your working directory and read the data using the robust read_csv()
setwd("/Users/mingikang/Developer/Convolutional-Nearest-Neighbor/Final_Output/K_test1_cos")
df <- read_csv("csv/VGG11-CIFAR100_k_test.csv")

branch_df = df %>% 
  filter(Model == "Branch")

baseline_df = df %>% 
  filter(Model == "Baseline")


# --- Step 1: Create the Plot (No processing needed!) ---
# We map the new columns directly.
k_plot <- ggplot(branch_df, aes(x = K, 
                             y = BestAcc,                # Use 'BestAcc' (or 'Last5AvgAcc') for the y-axis
                             color = factor(KS),         # Use factor(KS) for discrete colors
                             linetype = Model              # Use 'Model' for linetype
)) +
  geom_line(linewidth = 1) + # Draw the lines
  geom_hline(
    data = baseline_df, 
    aes(yintercept = BestAcc, 
        color = factor(KS), 
        linetype = Model), 
    linewidth = 1.0
  ) + 
  
  # --- Customize Scales and Legends ---
  scale_color_brewer(
    palette = "Set1",
    name = "Kernel Size" # Sets the title for the color legend
  ) +
  scale_linetype_manual(
    name = "Model Type", # Sets the title for the linetype legend
    # IMPORTANT: Update these values to match the strings in your 'Model' column
    values = c("Baseline" = "dotted", "Branch" = "solid") 
  ) +
  
  # --- Customize Axes and Labels ---
  scale_x_continuous(breaks = 1:12) + # Ensure integer ticks for K
  
  # (Optional) Adjust Y-axis breaks based on your new data's range
  # scale_y_continuous(breaks = seq(50, 60, by = 2)) + 
  
  labs(
    x = "K (Number of Neighbors)",
    y = "Best Accuracy (%)" # Updated to reflect 'BestAcc'
  ) +
  
  # --- Apply a Theme ---
  theme_bw(base_size = 12) +
  theme(
    legend.position = "right",
    plot.title = element_text(face = "bold", size = 23),
    plot.subtitle = element_text(size = 18),
    legend.title = element_text(face = "bold") # Make legend titles bold
  )

# Step 2: Display the plot
print(k_plot)

# Step 3: (Optional) Save the plot
# ggsave(
#   "path/to/your/new_plot.png",
#   plot = k_plot,
#   width = 10,
#   height = 6,
#   units = "in",
#   dpi = 300,
#   bg = "white"
# )