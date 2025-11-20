library(tidyverse)
library(zoo)  # For rollapply
library(patchwork)

# Set working directory
setwd("/Users/mingikang/Developer/Convolutional-Nearest-Neighbor/Final_Output/loss_test_cos_first---/csv/lr")

# Constants
SMOOTH_WINDOW <- 10

# ==========================================
# PART 1: CIFAR-10 (Top Plot)
# ==========================================
# Read Data
data_c10 <- read.csv("VGG11-CIFAR10_lr_1e-4.csv")

# Process Data
df_c10 <- data_c10 %>%
  pivot_longer(
    cols = -epoch,
    names_to = c("Model", "Type", ".value"),
    names_sep = "_"
  ) %>%
  group_by(Model, Type) %>%
  mutate(
    Loss_smooth = rollapply(Loss, width=SMOOTH_WINDOW, FUN = mean, align="right", fill=NA, partial=TRUE)
  ) %>%
  ungroup()

# Create Plot
plot_c10 <- df_c10 %>%
  ggplot(aes(x = epoch, y = Loss_smooth, color = Model, linetype = Type)) +
  geom_line(linewidth = 1.0) +
  
  # Scales
  scale_linetype_manual(
    name = "Loss Type", 
    values = c("Train" = "dotted", "Test" = "solid")
  ) +
  scale_color_manual(
    name = "Model",
    values = c(
      "Conv2d" = "#4DAF4A", # Set1 Blue
      "ConvNN" = "#377EB8",  # Set1 Green
      "Branching" = "#E41A1C"
    )
  ) +  
  # Labels
  labs(
    title = "CIFAR-10",        # Added Title
    x = NULL,                  # Removed X Label for top plot
    y = "Loss",
    color = "Model"
  ) +
  
  # Theme
  theme_bw(base_size = 12) +
  theme(
    plot.title = element_text(face = "bold", size = 12),
    legend.title = element_text(face = "bold", size = 12),
    legend.text = element_text(size = 12)
  )

# ==========================================
# PART 2: CIFAR-100 (Bottom Plot)
# ==========================================
# Read Data
data_c100 <- read.csv("VGG11-CIFAR100_lr_1e-4.csv")

# Process Data
df_c100 <- data_c100 %>%
  pivot_longer(
    cols = -epoch,
    names_to = c("Model", "Type", ".value"),
    names_sep = "_"
  ) %>%
  group_by(Model, Type) %>%
  mutate(
    Loss_smooth = rollapply(Loss, width=SMOOTH_WINDOW, FUN = mean, align="right", fill=NA, partial=TRUE)
  ) %>%
  ungroup()

# Create Plot
plot_c100 <- df_c100 %>%
  ggplot(aes(x = epoch, y = Loss_smooth, color = Model, linetype = Type)) +
  geom_line(linewidth = 1.0) +
  
  # Scales
  scale_linetype_manual(
    name = "Loss Type", 
    values = c("Train" = "dotted", "Test" = "solid")
  ) +
  scale_color_manual(
    name = "Model",
    values = c(
      "Conv2d" = "#4DAF4A", # Set1 Blue
      "ConvNN"    = "#377EB8", # Set1 Green
      "Branching" = "#E41A1C"
    )
  ) +
  # Labels
  labs(
    title = "CIFAR-100",             # Added Title
    x = "Epochs",                    # Keep X Label here
    y = "Loss",
    color = "Model"
  ) +
  
  # Theme
  theme_bw(base_size = 12) +
  theme(
    plot.title = element_text(face = "bold", size = 12),
    legend.title = element_text(face = "bold", size = 12),
    legend.text = element_text(size = 12)
  )

# ==========================================
# PART 3: Combine and Save
# ==========================================

combined_plot = plot_c10 / plot_c100

final_plot = combined_plot + 
  plot_layout(guides = "collect") & 
  theme(legend.position = "bottom")     # Apply bottom legend to ALL

print(final_plot)


ggsave(
  "/Users/mingikang/Developer/Convolutional-Nearest-Neighbor/Final_Plots/VGG_combined_loss_1e-4.png",
  plot = final_plot,
  width = 8,
  height = 8 , # Made height taller to accommodate 2 plots
  units = "in",
  dpi = 500,
  bg = "white"
)
