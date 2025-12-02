library(tidyverse)
library(ggplot2)
library(tidyr)
library(scales) # For the percent axis

# Set working directory
setwd("/Users/mingikang/Developer/Convolutional-Nearest-Neighbor/Final_Output/K_test")

# Read data
data <- read_csv("csv/VGG11-CIFAR10_k_test.csv")

# --- 1. Robust Data Wrangling (FIXED) ---
df_long <- data %>%
  pivot_longer(
    cols = -epoch,
    names_to = "column_name",
    values_to = "value"
  ) %>%
  filter(epoch == 150) %>%
  
  # FIX 1: Use the correct regex that matches "K0", "K1", etc. and "KS" or "Ks"
  extract(
    column_name,
    into = c("Model", "Ks_raw", "K_val", "Type", "Metric"),
    regex = "(\\w+)_([Kk][Ss]\\d)_K(\\d+)_(Test|Train)_(Accuracy|Loss)",
    remove = FALSE
  ) %>%
  
  # FIX 2: Create normalized Ks column and correctly parse K
  mutate(
    K = as.numeric(K_val),
    Ks = toupper(Ks_raw) # Normalize to "KS1", "KS2"
  ) %>%
  
  filter(!is.na(Model), !is.na(Type), !is.na(Metric)) %>%
  select(-column_name, -Ks_raw, -K_val) %>% # Clean up
  pivot_wider(names_from = Metric, values_from = value)


# --- 2. Filter Data for Plotting (FIXED LOGIC) ---
df_test_all <- df_long %>%
  filter(Type == "Test", !is.na(Accuracy)) %>%
  filter(Ks != "KS1") # Keep your filter to remove kernel size 1

# FIX 3: Baselines are ALL models where K is 0
df_baselines <- df_test_all %>%
  filter(K == 0) # This will get Conv2d AND ConvNN models

# FIX 4: K-Test models are ALL models where K is greater than 0
df_k_test <- df_test_all %>%
  filter(K > 0) # This will get the Branching models


# --- 3. Create the Plot ---
combined_plot <- ggplot() +
  
  # Add horizontal lines for baselines (Conv2d and ConvNN)
  geom_hline(
    data = df_baselines,
    aes(yintercept = Accuracy, color = Ks, linetype = Model),
    linewidth = 1.1 
  ) +
  
  # Add lines for K-Test models (Branching)
  geom_line(
    data = df_k_test,
    aes(x = K, y = Accuracy, color = Ks, linetype = Model),
    linewidth = 1.2 
  ) +
  
  # Add points for K-Test models
  geom_point(
    data = df_k_test,
    aes(x = K, y = Accuracy, color = Ks),
    size = 2.5
  ) +
  
  # --- Customize Scales (FIXED) ---
  
  # Customize colors (Removed KS1)
  scale_color_manual(
    name = "Kernel Size",
    values = c("KS2" = "#1F78B4", "KS3" = "#33A02C"),
    labels = c("KS2" = "2 (2x2)", "KS3" = "3 (3x3)")
  ) +
  
  # Customize linetype (Includes all 3 models)
  scale_linetype_manual(
    name = "Model Type",
    values = c("Conv2d" = "dotted", "ConvNN" = "dashed", "Branching" = "solid"),
    labels = c("Conv2d" = "Standard Conv2d", "ConvNN" = "ConvNN (Baseline)", "Branching" = "Branching ConvNN")
  ) +
  
  # Customize x-axis
  scale_x_continuous(
    breaks = 1:12
  ) +
  # 
  # # Customize y-axis
  # scale_y_continuous(
  #   limits = c(70, 82.5),
  #   breaks = seq(70, 82.5, 2.5),
  #   labels = scales::percent_format(scale = 1, accuracy = 0.1)
  # ) +
  
  # --- Labels and Titles ---
  labs(
    title = "Model Accuracy by Kernel Size and Type",
    subtitle = "Comparison of Conv2d, ConvNN, and Branching ConvNN",
    x = "K (Number of Neighbors)",
    y = "Top-1 Accuracy (%)"
  ) +
  
  # --- Theme ---
  theme_bw(base_size = 14) + 
  theme(
    plot.title = element_text(face = "bold", size = 18),
    plot.subtitle = element_text(size = 14, margin = margin(b = 10)),
    legend.position = "right",
    legend.title = element_text(face = "bold"),
    legend.box = "vertical",
    legend.spacing.y = unit(0.5, 'cm'), 
    panel.grid.minor = element_blank(),
    panel.border = element_rect(colour = "black", linewidth = 1) 
  ) +
  
  # Customize legend appearance
  guides(
    color = guide_legend(order = 2, title = "Kernel Size"), 
    linetype = guide_legend(order = 1, title = "Model Type")
  )

# Display the plot
print(combined_plot)

# # Save the plot
# ggsave(
#   "plots/VGG11_CIFAR10_k_test_all_models.png", # New filename
#   plot = combined_plot,
#   width = 10,
#   height = 6,
#   units = "in",
#   dpi = 300,
#   bg = "white"
# )