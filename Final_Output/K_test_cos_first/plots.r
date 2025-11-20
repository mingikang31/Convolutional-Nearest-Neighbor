
library(tidyverse)
# The stringr package (part of tidyverse) is used for str_detect()
library(stringr) 

# --- Data Reading ---
# Set your working directory and read the data using the robust read_csv()
setwd("/Users/mingikang/Developer/Convolutional-Nearest-Neighbor/Final_Output/K_test")
df <- read_csv("csv/VGG11-CIFAR100_k_test_summary.csv")

# --- Step 1: Reshape and Process Data ---
# Reshape the data and create new columns for Kernel Size and Model Type
df_processed <- df %>%
  pivot_longer(
    cols = -K,
    names_to = "Metric",
    values_to = "Value"
  ) %>%
  mutate(
    # Create a 'Kernel_Size' column by extracting the number from the 'Metric' string
    Kernel_Size = case_when(
      str_detect(Metric, "Ks1") ~ "1",
      str_detect(Metric, "Ks2") ~ "2",
      str_detect(Metric, "Ks3") ~ "3"
    ),
    # Create a 'Model_Type' column based on whether '_K' is in the 'Metric' string
    Model_Type = case_when(
      str_detect(Metric, "_K") ~ "Branching ConvNN",
      TRUE                     ~ "Standard Conv2d" # 'TRUE' acts as an else condition
    )
  )

# --- Step 2: Create the Plot with New Aesthetics ---
# Map 'color' to Kernel_Size and 'linetype' to Model_Type
k_plot <- ggplot(df_processed, aes(x = K, y = Value, color = Kernel_Size, linetype = Model_Type)) +
  geom_line(linewidth = 1.2) + # Draw the lines
  
  # --- Customize Scales and Legends ---
  scale_color_brewer(
    palette = "Set1",
    name = "Kernel Size" # Sets the title for the color legend
  ) + 
  scale_linetype_manual(
    name = "Model Type", # Sets the title for the linetype legend
    values = c("Standard Conv2d" = "dotted", "Branching ConvNN" = "solid")
  ) +
  
  # --- Customize Axes and Labels ---
  scale_x_continuous(breaks = 1:12) + # Ensure integer ticks for K
  #scale_y_continuous(breaks = seq(50, 82.25, 2.5)) + # Ensure integer ticks for K
  
  labs(
    # title = "Model Accuracy by Kernel Size and Type",
    # subtitle = "Comparison of Conv2d and Branching ConvNN",
    x = "K (Number of Neighbors)",
    y = "Top-1 Accuracy (%)"
  ) +
  
  # --- Apply a Theme ---
  theme_bw(base_size = 12) +
  theme(
    legend.position = "right",
    plot.title = element_text(face = "bold", size = 23),
    plot.subtitle = element_text(size=18),
    legend.title = element_text(face = "bold") # Make legend titles bold
  )

# Step 3: Display the plot
print(k_plot)


save_path = "/Users/mingikang/Developer/Convolutional-Nearest-Neighbor/"

# # Step 4: Save the plot to a file
# ggsave(
#   save_path,
#   plot = k_plot,
#   width = 10,
#   height = 6,
#   units = "in",
#   dpi = 300,
#   bg = "white"
# )
