library(tidyverse)
library(stringr)
library(patchwork)

# ==========================================
# PART 1: CIFAR-10 (Top Plot)
# ==========================================
setwd("/Users/mingikang/Developer/Convolutional-Nearest-Neighbor/Final_Output/K_test_pt4---/")
df_c10 <- read_csv("csv/VGG11-CIFAR10_k_test.csv")

# Prepare Data
branch_c10 = df_c10 %>% 
  filter(Model == "Branch") %>%
  mutate(Model = "Branching")

baseline_c10 = df_c10 %>% 
  filter(Model == "Baseline") %>%
  mutate(Model = "Conv2d")

# Create Plot 1 (CIFAR-10)
k_plot_c10 <- ggplot(branch_c10, aes(x = K, y = BestAcc, color = factor(KS), linetype = Model)) +
  geom_line(linewidth = 1) + 
  geom_hline(data = baseline_c10, aes(yintercept = BestAcc, color = factor(KS), linetype = Model), linewidth = 1.0) + 
  
  # Scales
  scale_color_brewer(palette = "Set1", name = "Conv2d Kernel Size") +
  scale_linetype_manual(name = "Model Type", values = c("Conv2d" = "dotted", "Branching" = "solid")) +
  scale_x_continuous(breaks = seq(1, 12, by = 1), limits = c(1, 12)) + 
  scale_y_continuous(breaks = seq(0, 100, by = 5)) + 
  
  # Labels & Theme
  labs(
    title = "CIFAR-10",       # <--- Added Title
    x = NULL,                 # <--- Removed X Label (redundant for top plot)
    y = "Top-1 Accuracy (%)"
  ) +
  theme_bw(base_size = 12) +
  theme(
    plot.title = element_text(face = "bold", size = 12),
    legend.title = element_text(face = "bold", size = 12),
    legend.text = element_text(size = 12)
  )

# ==========================================
# PART 2: CIFAR-100 (Bottom Plot)
# ==========================================
# Note: We read the new file, but keep variable names distinct to avoid confusion
df_c100 <- read_csv("csv/VGG11-CIFAR100_k_test.csv")

# Prepare Data
branch_c100 = df_c100 %>% 
  filter(Model == "Branch") %>%
  mutate(Model = "Branching")

baseline_c100 = df_c100 %>% 
  filter(Model == "Baseline") %>%
  mutate(Model = "Conv2d")

# Create Plot 2 (CIFAR-100)
k_plot_c100 <- ggplot(branch_c100, aes(x = K, y = BestAcc, color = factor(KS), linetype = Model)) +
  geom_line(linewidth = 1) + 
  geom_hline(data = baseline_c100, aes(yintercept = BestAcc, color = factor(KS), linetype = Model), linewidth = 1.0) + 
  
  # Scales (MUST BE IDENTICAL to top plot for legend merging)
  scale_color_brewer(palette = "Set1", name = "Conv2d Kernel Size") +
  scale_linetype_manual(name = "Model Type", values = c("Conv2d" = "dotted", "Branching" = "solid")) +
  scale_x_continuous(breaks = seq(1, 12, by = 1), limits = c(1, 12)) + 
  scale_y_continuous(breaks = seq(0, 100, by = 5)) + 
  # Labels & Theme
  labs(
    title = "CIFAR-100",      # <--- Added Title
    x = "K (Number of Neighbors)", # Keep X Label here
    y = "Top-1 Accuracy (%)"
  ) +
  theme_bw(base_size = 12) +
  theme(
    plot.title = element_text(face = "bold", size = 12),
    legend.title = element_text(face = "bold", size = 12),
    legend.text = element_text(size = 12)
  )

# ==========================================
# PART 3: Combine with Patchwork
# ==========================================

combined_plot = k_plot_c10 / k_plot_c100

final_plot = combined_plot + 
  plot_layout(guides = "collect") &   # Add A/B tags
  theme(legend.position = "bottom")     # Apply bottom legend to the WHOLE structure

print(final_plot)

# Save
ggsave(
  "/Users/mingikang/Developer/Convolutional-Nearest-Neighbor/Final_Plots/VGG_combined_k_plot.png",
  plot = final_plot,
  width = 8,
  height = 6, # Made height taller to accommodate 2 plots
  units = "in",
  dpi = 500,
  bg = "white"
)