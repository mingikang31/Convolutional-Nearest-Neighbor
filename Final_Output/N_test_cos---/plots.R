library(tidyverse)
library(patchwork)

# --- Data Reading ---
setwd("/Users/mingikang/Developer/Convolutional-Nearest-Neighbor/Final_Output/N_test_cos---/")
df <- read_csv("csv/N_test_cos_results.csv")

df = df %>%
  filter(N != 32)

# ==============================================================================
# HELPER: Define Colors to Match Your Image
# ==============================================================================
# We map the "Human Readable" label to the specific hex codes
custom_colors <- c(
  "Spatial (r)"       = "#E41A1C",  # Red
  "Random (r^2)"      = "#377EB8",  # Blue (Assuming 'color' in CSV is Random)
  "All Samples"       = "black",    # Black
  "Conv2d (baseline)" = "#4DAF4A"   # Green
)

# ==============================================================================
# FUNCTION: Prepare Data and Plot
# This function avoids repeating code for CIFAR10 and CIFAR100
# ==============================================================================
create_n_plot <- function(data, dataset_name, plot_title, show_x_label = TRUE) {

  # 1. Filter and Create Legend Labels
  #    This is the most important step. We map the raw data to the Legend Names.
  plot_data <- data %>%
    filter(Dataset == dataset_name) %>%
    mutate(Legend_Label = case_when(
      Model == "Convolution" ~ "Conv2d (baseline)",
      Model == "Branching" & Type == "all" ~ "All Samples",
      Model == "Branching" & Type == "spatial" ~ "Spatial (r)",
      Model == "Branching" & Type == "random" ~ "Random (r^2)"
    )) %>%
    filter(!is.na(Legend_Label)) # Remove anything else (like ConvNN All/Spatial)

  # 2. Separate Dynamic Lines (Changes with N) vs Horizontal Lines (Constant)
  lines_df <- plot_data %>%
    filter(Legend_Label %in% c("Spatial (r)", "Random (r^2)"))

  hlines_df <- plot_data %>%
    filter(Legend_Label %in% c("All Samples", "Conv2d (baseline)")) %>%
    group_by(Legend_Label) %>% slice(1) %>% ungroup()

  # 3. Create Plot
  p <- ggplot() +
    # Dynamic Lines (Spatial, Random)
    geom_line(
      data = lines_df,
      aes(x = N, y = BestAcc, color = Legend_Label),
      linewidth = 1.0
    ) +
    # Horizontal Lines (All, Baseline)
    geom_segment(
      data = hlines_df,
      aes(
        x = 4,               # Start at 4
        xend = 30,           # End at 30
        y = BestAcc,         # Y height
        yend = BestAcc,      # Same Y height (keeps it flat)
        color = Legend_Label
      ),
      linewidth = 1.0
    ) +
    # Scales
    scale_color_manual(name = "Sampling Method", values = custom_colors) +
    scale_x_continuous(breaks = seq(4, 30, 2)) +
    scale_y_continuous(breaks = seq(0, 100, )) + # Tick every 0.5%
    # Labels
    labs(
      title = plot_title,
      y = "Top-1 Accuracy (%)",
      x = NULL
    ) +
    theme_bw(base_size = 12) +
    theme(
      plot.title = element_text(face = "bold", size = 12),
      # legend.title = element_text(face = "bold", size = 12),
      # legend.text = element_text(size = 12),
      legend.position = "none"
    )

  return(p)
}

# ==============================================================================
# GENERATE PLOTS
# ==============================================================================

# 1. Generate CIFAR-10 (Top Plot)
p_c10 <- create_n_plot(df, "VGG11-CIFAR10", "CIFAR-10", show_x_label = FALSE)

# 2. Generate CIFAR-100 (Bottom Plot)
p_c100 <- create_n_plot(df, "VGG11-CIFAR100", "CIFAR-100", show_x_label = TRUE)

# 3. Combine
final_plot <- (p_c10 / p_c100)
# +
#   plot_layout(guides = "collect") &
#   theme(legend.position = "bottom")

# Print
print(final_plot)

# Save
ggsave(
  "/Users/mingikang/Developer/Convolutional-Nearest-Neighbor/Final_Plots/VGG_combined_n_plot.png",
  plot = final_plot,
  width = 8,
  height = 5,
  units = "in",
  dpi = 500,
  bg = "white"
)





### GFLOPS CODE

library(tidyverse)
library(patchwork)


### 1. Accuracy vs. N ###
# --- Data Reading ---
setwd("/Users/mingikang/Developer/Convolutional-Nearest-Neighbor/Final_Output/N_test_cos---/")
df <- read_csv("csv/N_test_cos_results.csv")

df = df %>%
  filter(N != 32)

fmt_mflops <- function(x) {
  # Multiply by 1000 (1 G = 1000 M) and format as integer with "M"
  # Example: 0.031 -> "31M"
  # sprintf("%.0fK", x * 1000)
  return(x)
}
                 
# ==============================================================================
# HELPER: Define Colors to Match Your Image
# ==============================================================================
# We map the "Human Readable" label to the specific hex codes
custom_colors <- c(
  "Spatial"       = "#E41A1C",  # Red
  "Random"      = "#377EB8",  # Blue (Assuming 'color' in CSV is Random)
  "All Samples"       = "black",    # Black
  "Conv2d (baseline)" = "#4DAF4A"   # Green
)

# ==============================================================================
# FUNCTION: Prepare Data and Plot
# This function avoids repeating code for CIFAR10 and CIFAR100
# ==============================================================================
create_n_plot <- function(data, dataset_name, plot_title, show_x_label = TRUE) {

  # 1. Filter and Create Legend Labels
  #    This is the most important step. We map the raw data to the Legend Names.
  plot_data <- data %>%
    filter(Dataset == dataset_name) %>%
    mutate(Legend_Label = case_when(
      Model == "Convolution" ~ "Conv2d (baseline)",
      Model == "Branching" & Type == "all" ~ "All Samples",
      Model == "Branching" & Type == "spatial" ~ "Spatial",
      Model == "Branching" & Type == "random" ~ "Random"
    )) %>%
    filter(!is.na(Legend_Label)) # Remove anything else (like ConvNN All/Spatial)

  # 2. Separate Dynamic Lines (Changes with N) vs Horizontal Lines (Constant)
  lines_df <- plot_data %>%
    filter(Legend_Label %in% c("Spatial", "Random"))

  hlines_df <- plot_data %>%
    filter(Legend_Label %in% c("All Samples", "Conv2d (baseline)")) %>%
    group_by(Legend_Label) %>% slice(1) %>% ungroup()

  # 3. Create Plot
  p <- ggplot() +
    # Dynamic Lines (Spatial, Random)
    geom_line(
      data = lines_df,
      aes(x = N,
          y = ifelse(Legend_Label == "Random", GFLOPs + 0.0013, GFLOPs),
          color = Legend_Label),
      linewidth = 1.0,

    ) +
    # Horizontal Lines (All, Baseline)
    geom_segment(
      data = hlines_df,
      aes(
        x = 4,               # Start at 4
        xend = 30,           # End at 30
        y = GFLOPs,         # Y height
        yend = GFLOPs,      # Same Y height (keeps it flat)
        color = Legend_Label
      ),
      linewidth = 1.0
    ) +
    
    scale_color_manual(name = "Sampling Method", values = custom_colors) +
    scale_x_continuous(breaks = seq(4, 30, 2)) +
    scale_y_continuous(labels = fmt_mflops) + # Tick every 0.5%
    # Labels
    labs(
      y = "GFLOPS",
      x = "r (Number of Candidates)"
    ) +
    theme_bw(base_size = 12) +
    theme(    legend.position = "bottom",
              legend.title = element_text(face = "bold", size = 12),
              legend.text = element_text(size = 14))

  return(p)
}

# ==============================================================================
# GENERATE PLOTS
# ==============================================================================

# 1. Generate CIFAR-10 (Top Plot)
p_c10 <- create_n_plot(df, "VGG11-CIFAR10", "CIFAR-10", show_x_label = FALSE)

# p_c100 <- create_n_plot(df, "VGG11-CIFAR100", "CIFAR-100", show_x_label = TRUE)
#
# # 3. Combine
# final_plot <- (p_c10 / p_c100) +
#   plot_layout(guides = "collect") &
#   theme(legend.position = "bottom")
#
# # Print
print(p_c10)
ggsave(
  "/Users/mingikang/Developer/Convolutional-Nearest-Neighbor/Final_Plots/VGG_combined_n_gflop_plot.png",
  plot = p_c10,
  width = 8,
  height = 2.5,
  units = "in",
  dpi = 500,
  bg = "white"
)

