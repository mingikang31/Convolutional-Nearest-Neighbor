# library(ggplot2)
# library(dplyr)
# library(readr)
# 
# # Read the CSV file
# data <- read_csv("K_N_tests.csv")
# 
# # Clean and prepare the data
# data_clean <- data %>%
#   # Remove rows with missing values in key columns
#   filter(!is.na(Model), !is.na(K), !is.na(`Top1 Overall`)) %>%
#   # Convert Top1 Overall from percentage string to numeric
#   mutate(
#     Top1_Overall_numeric = as.numeric(gsub("%", "", `Top1 Overall`)),
#     # Create a combined factor for model and coordination
#     Model_Coord = paste0(Model, " (", ifelse(Coord == "Yes", "with coord", "without coord"), ")"),
#     # Convert Coord to a factor for better plotting
#     Coord_factor = factor(Coord, levels = c("No", "Yes"), labels = c("Without Coord", "With Coord"))
#   )
# 
# # Create the main plot
# p1 <- ggplot(data_clean, aes(x = K, y = Top1_Overall_numeric, color = Model_Coord)) +
#   geom_line(size = 1, alpha = 0.8) +
#   geom_point(size = 2, alpha = 0.9) +
#   labs(
#     title = "Model Performance vs Number of Ks",
#     subtitle = "Top1 Overall Accuracy across different models and coordination settings",
#     x = "Number of Ks (K)",
#     y = "Top1 Overall Accuracy (%)",
#     color = "Model & Coordination"
#   ) +
#   theme_minimal() +
#   theme(
#     plot.title = element_text(size = 14, face = "bold"),
#     plot.subtitle = element_text(size = 12),
#     legend.position = "bottom",
#     legend.text = element_text(size = 9),
#     legend.title = element_text(size = 10, face = "bold"),
#     axis.title = element_text(size = 11),
#     axis.text = element_text(size = 10)
#   ) +
#   guides(color = guide_legend(ncol = 2)) +
#   scale_x_continuous(breaks = 1:10) +
#   scale_y_continuous(limits = c(min(data_clean$Top1_Overall_numeric) - 2, 
#                                 max(data_clean$Top1_Overall_numeric) + 2))
# 
# # Alternative plot with faceting by coordination
# p2 <- ggplot(data_clean, aes(x = K, y = Top1_Overall_numeric, color = Model)) +
#   geom_line(size = 1, alpha = 0.8) +
#   geom_point(size = 2, alpha = 0.9) +
#   facet_wrap(~ Coord_factor, ncol = 2) +
#   labs(
#     title = "Model Performance vs Number of Ks (Faceted by Coordination)",
#     subtitle = "Top1 Overall Accuracy comparison between with/without coordination",
#     x = "Number of Ks (K)",
#     y = "Top1 Overall Accuracy (%)",
#     color = "Model"
#   ) +
#   theme_minimal() +
#   theme(
#     plot.title = element_text(size = 14, face = "bold"),
#     plot.subtitle = element_text(size = 12),
#     legend.position = "bottom",
#     legend.text = element_text(size = 9),
#     legend.title = element_text(size = 10, face = "bold"),
#     axis.title = element_text(size = 11),
#     axis.text = element_text(size = 10),
#     strip.text = element_text(size = 11, face = "bold")
#   ) +
#   scale_x_continuous(breaks = 1:10) +
#   scale_y_continuous(limits = c(min(data_clean$Top1_Overall_numeric) - 2, 
#                                 max(data_clean$Top1_Overall_numeric) + 2))
# 
# # Alternative plot with separate lines for coord vs no coord
# p3 <- ggplot(data_clean, aes(x = K, y = Top1_Overall_numeric, color = Model, linetype = Coord_factor)) +
#   geom_line(size = 1, alpha = 0.8) +
#   geom_point(size = 2, alpha = 0.9) +
#   labs(
#     title = "Model Performance vs Number of Ks",
#     subtitle = "Top1 Overall Accuracy with coordination shown by line type",
#     x = "Number of Ks (K)",
#     y = "Top1 Overall Accuracy (%)",
#     color = "Model",
#     linetype = "Coordination"
#   ) +
#   theme_minimal() +
#   theme(
#     plot.title = element_text(size = 14, face = "bold"),
#     plot.subtitle = element_text(size = 12),
#     legend.position = "bottom",
#     legend.text = element_text(size = 9),
#     legend.title = element_text(size = 10, face = "bold"),
#     axis.title = element_text(size = 11),
#     axis.text = element_text(size = 10)
#   ) +
#   scale_x_continuous(breaks = 1:10) +
#   scale_y_continuous(limits = c(min(data_clean$Top1_Overall_numeric) - 2, 
#                                 max(data_clean$Top1_Overall_numeric) + 2))
# 
# # Display the plots
# print("Plot 1: All model-coordination combinations as separate lines")
# print(p1)
# 
# print("Plot 2: Faceted by coordination (with/without)")
# print(p2)
# 
# print("Plot 3: Models colored, coordination as line type")
# print(p3)
# 
# # Summary statistics
# print("Summary by Model and Coordination:")
# summary_stats <- data_clean %>%
#   group_by(Model, Coord_factor) %>%
#   summarise(
#     Mean_Accuracy = round(mean(Top1_Overall_numeric), 2),
#     Max_Accuracy = round(max(Top1_Overall_numeric), 2),
#     Min_Accuracy = round(min(Top1_Overall_numeric), 2),
#     Best_K = K[which.max(Top1_Overall_numeric)],
#     .groups = 'drop'
#   )
# print(summary_stats)
# 
# 
# 
# 



library(ggplot2)
library(dplyr)
library(readr)
setwd("/Users/mingikang/Developer/Convolutional-Nearest-Neighbor/Output/Final_results/ACM")
# Read the CSV file
data <- read_csv("K_N_tests.csv")

# Clean and prepare the data
data_clean <- data %>%
  # Remove rows with missing values in key columns
  filter(!is.na(Model), !is.na(K), !is.na(`Top1 Overall`)) %>%
  # Convert percentage strings to numeric and prepare all metrics
  mutate(
    Top1_Overall_numeric = as.numeric(gsub("%", "", `Top1 Overall`)),
    Top1_at_50_numeric = as.numeric(gsub("%", "", `Top1 at 50 Epoch`)),
    Top1_Epoch_numeric = as.numeric(`Top1 Epoch`),
    # Create a combined factor for model and coordination
    Model_Coord = paste0(Model, " (", ifelse(Coord == "Yes", "with coord", "without coord"), ")"),
    # Convert Coord to a factor for better plotting
    Coord_factor = factor(Coord, levels = c("No", "Yes"), labels = c("Without Coord", "With Coord"))
  )

# Create the main plot
p1 <- ggplot(data_clean, aes(x = K, y = Top1_Overall_numeric, color = Model_Coord)) +
  geom_line(size = 1, alpha = 0.8) +
  geom_point(size = 2, alpha = 0.9) +
  labs(
    title = "Model Performance vs Number of Ks",
    subtitle = "Top1 Overall Accuracy across different models and coordination settings",
    x = "Number of Ks (K)",
    y = "Top1 Overall Accuracy (%)",
    color = "Model & Coordination"
  ) +
  theme_minimal() +
  theme(
    plot.title = element_text(size = 14, face = "bold"),
    plot.subtitle = element_text(size = 12),
    legend.position = "bottom",
    legend.text = element_text(size = 9),
    legend.title = element_text(size = 10, face = "bold"),
    axis.title = element_text(size = 11),
    axis.text = element_text(size = 10)
  ) +
  guides(color = guide_legend(ncol = 2)) +
  scale_x_continuous(breaks = 1:10) +
  scale_y_continuous(limits = c(min(data_clean$Top1_Overall_numeric) - 2, 
                                max(data_clean$Top1_Overall_numeric) + 2))

# Alternative plot with faceting by coordination
p2 <- ggplot(data_clean, aes(x = K, y = Top1_Overall_numeric, color = Model)) +
  geom_line(size = 1, alpha = 0.8) +
  geom_point(size = 2, alpha = 0.9) +
  facet_wrap(~ Coord_factor, ncol = 2) +
  labs(
    title = "Model Performance vs Number of Ks (Faceted by Coordination)",
    subtitle = "Top1 Overall Accuracy comparison between with/without coordination",
    x = "Number of Ks (K)",
    y = "Top1 Overall Accuracy (%)",
    color = "Model"
  ) +
  theme_minimal() +
  theme(
    plot.title = element_text(size = 14, face = "bold"),
    plot.subtitle = element_text(size = 12),
    legend.position = "bottom",
    legend.text = element_text(size = 9),
    legend.title = element_text(size = 10, face = "bold"),
    axis.title = element_text(size = 11),
    axis.text = element_text(size = 10),
    strip.text = element_text(size = 11, face = "bold")
  ) +
  scale_x_continuous(breaks = 1:10) +
  scale_y_continuous(limits = c(min(data_clean$Top1_Overall_numeric) - 2, 
                                max(data_clean$Top1_Overall_numeric) + 2))

# Alternative plot with separate lines for coord vs no coord
p3 <- ggplot(data_clean, aes(x = K, y = Top1_Overall_numeric, color = Model, linetype = Coord_factor)) +
  geom_line(size = 1, alpha = 0.8) +
  geom_point(size = 2, alpha = 0.9) +
  labs(
    title = "Model Performance vs Number of Ks",
    subtitle = "Top1 Overall Accuracy with coordination shown by line type",
    x = "Number of Ks (K)",
    y = "Top1 Overall Accuracy (%)",
    color = "Model",
    linetype = "Coordination"
  ) +
  theme_minimal() +
  theme(
    plot.title = element_text(size = 14, face = "bold"),
    plot.subtitle = element_text(size = 12),
    legend.position = "bottom",
    legend.text = element_text(size = 9),
    legend.title = element_text(size = 10, face = "bold"),
    axis.title = element_text(size = 11),
    axis.text = element_text(size = 10)
  ) +
  scale_x_continuous(breaks = 1:10) +
  scale_y_continuous(limits = c(min(data_clean$Top1_Overall_numeric) - 2, 
                                max(data_clean$Top1_Overall_numeric) + 2))

# Display the plots
print("=== TOP1 OVERALL ACCURACY PLOTS ===")
print("Plot 1: All model-coordination combinations as separate lines")
print(p1)

print("Plot 2: Faceted by coordination (with/without)")
print(p2)

print("Plot 3: Models colored, coordination as line type")
print(p3)

# ========================================
# PLOTS FOR TOP1 AT 50 EPOCH
# ========================================

p4 <- ggplot(data_clean, aes(x = K, y = Top1_at_50_numeric, color = Model_Coord)) +
  geom_line(size = 1, alpha = 0.8) +
  geom_point(size = 2, alpha = 0.9) +
  labs(
    title = "Model Performance at Epoch 50 vs Number of Ks",
    subtitle = "Top1 Accuracy at 50 Epochs across different models and coordination settings",
    x = "Number of Ks (K)",
    y = "Top1 Accuracy at 50 Epochs (%)",
    color = "Model & Coordination"
  ) +
  theme_minimal() +
  theme(
    plot.title = element_text(size = 14, face = "bold"),
    plot.subtitle = element_text(size = 12),
    legend.position = "bottom",
    legend.text = element_text(size = 9),
    legend.title = element_text(size = 10, face = "bold"),
    axis.title = element_text(size = 11),
    axis.text = element_text(size = 10)
  ) +
  guides(color = guide_legend(ncol = 2)) +
  scale_x_continuous(breaks = 1:10) +
  scale_y_continuous(limits = c(min(data_clean$Top1_at_50_numeric, na.rm = TRUE) - 2, 
                                max(data_clean$Top1_at_50_numeric, na.rm = TRUE) + 2))

p5 <- ggplot(data_clean, aes(x = K, y = Top1_at_50_numeric, color = Model)) +
  geom_line(size = 1, alpha = 0.8) +
  geom_point(size = 2, alpha = 0.9) +
  facet_wrap(~ Coord_factor, ncol = 2) +
  labs(
    title = "Model Performance at Epoch 50 vs Number of Ks (Faceted by Coordination)",
    subtitle = "Top1 Accuracy at 50 Epochs comparison between with/without coordination",
    x = "Number of Ks (K)",
    y = "Top1 Accuracy at 50 Epochs (%)",
    color = "Model"
  ) +
  theme_minimal() +
  theme(
    plot.title = element_text(size = 14, face = "bold"),
    plot.subtitle = element_text(size = 12),
    legend.position = "bottom",
    legend.text = element_text(size = 9),
    legend.title = element_text(size = 10, face = "bold"),
    axis.title = element_text(size = 11),
    axis.text = element_text(size = 10),
    strip.text = element_text(size = 11, face = "bold")
  ) +
  scale_x_continuous(breaks = 1:10) +
  scale_y_continuous(limits = c(min(data_clean$Top1_at_50_numeric, na.rm = TRUE) - 2, 
                                max(data_clean$Top1_at_50_numeric, na.rm = TRUE) + 2))

p6 <- ggplot(data_clean, aes(x = K, y = Top1_at_50_numeric, color = Model, linetype = Coord_factor)) +
  geom_line(size = 1, alpha = 0.8) +
  geom_point(size = 2, alpha = 0.9) +
  labs(
    title = "Model Performance at Epoch 50 vs Number of Ks",
    subtitle = "Top1 Accuracy at 50 Epochs with coordination shown by line type",
    x = "Number of Ks (K)",
    y = "Top1 Accuracy at 50 Epochs (%)",
    color = "Model",
    linetype = "Coordination"
  ) +
  theme_minimal() +
  theme(
    plot.title = element_text(size = 14, face = "bold"),
    plot.subtitle = element_text(size = 12),
    legend.position = "bottom",
    legend.text = element_text(size = 9),
    legend.title = element_text(size = 10, face = "bold"),
    axis.title = element_text(size = 11),
    axis.text = element_text(size = 10)
  ) +
  scale_x_continuous(breaks = 1:10) +
  scale_y_continuous(limits = c(min(data_clean$Top1_at_50_numeric, na.rm = TRUE) - 2, 
                                max(data_clean$Top1_at_50_numeric, na.rm = TRUE) + 2))

# ========================================
# PLOTS FOR TOP1 EPOCH (Best Epoch Number)
# ========================================

p7 <- ggplot(data_clean, aes(x = K, y = Top1_Epoch_numeric, color = Model_Coord)) +
  geom_line(size = 1, alpha = 0.8) +
  geom_point(size = 2, alpha = 0.9) +
  labs(
    title = "Best Epoch Number vs Number of Ks",
    subtitle = "Epoch where best Top1 accuracy was achieved across different models and coordination settings",
    x = "Number of Ks (K)",
    y = "Best Epoch Number",
    color = "Model & Coordination"
  ) +
  theme_minimal() +
  theme(
    plot.title = element_text(size = 14, face = "bold"),
    plot.subtitle = element_text(size = 12),
    legend.position = "bottom",
    legend.text = element_text(size = 9),
    legend.title = element_text(size = 10, face = "bold"),
    axis.title = element_text(size = 11),
    axis.text = element_text(size = 10)
  ) +
  guides(color = guide_legend(ncol = 2)) +
  scale_x_continuous(breaks = 1:10) +
  scale_y_continuous(limits = c(min(data_clean$Top1_Epoch_numeric, na.rm = TRUE) - 3, 
                                max(data_clean$Top1_Epoch_numeric, na.rm = TRUE) + 3))

p8 <- ggplot(data_clean, aes(x = K, y = Top1_Epoch_numeric, color = Model)) +
  geom_line(size = 1, alpha = 0.8) +
  geom_point(size = 2, alpha = 0.9) +
  facet_wrap(~ Coord_factor, ncol = 2) +
  labs(
    title = "Best Epoch Number vs Number of Ks (Faceted by Coordination)",
    subtitle = "Epoch where best Top1 accuracy was achieved, comparison between with/without coordination",
    x = "Number of Ks (K)",
    y = "Best Epoch Number",
    color = "Model"
  ) +
  theme_minimal() +
  theme(
    plot.title = element_text(size = 14, face = "bold"),
    plot.subtitle = element_text(size = 12),
    legend.position = "bottom",
    legend.text = element_text(size = 9),
    legend.title = element_text(size = 10, face = "bold"),
    axis.title = element_text(size = 11),
    axis.text = element_text(size = 10),
    strip.text = element_text(size = 11, face = "bold")
  ) +
  scale_x_continuous(breaks = 1:10) +
  scale_y_continuous(limits = c(min(data_clean$Top1_Epoch_numeric, na.rm = TRUE) - 3, 
                                max(data_clean$Top1_Epoch_numeric, na.rm = TRUE) + 3))

p9 <- ggplot(data_clean, aes(x = K, y = Top1_Epoch_numeric, color = Model, linetype = Coord_factor)) +
  geom_line(size = 1, alpha = 0.8) +
  geom_point(size = 2, alpha = 0.9) +
  labs(
    title = "Best Epoch Number vs Number of Ks",
    subtitle = "Epoch where best Top1 accuracy was achieved, coordination shown by line type",
    x = "Number of Ks (K)",
    y = "Best Epoch Number",
    color = "Model",
    linetype = "Coordination"
  ) +
  theme_minimal() +
  theme(
    plot.title = element_text(size = 14, face = "bold"),
    plot.subtitle = element_text(size = 12),
    legend.position = "bottom",
    legend.text = element_text(size = 9),
    legend.title = element_text(size = 10, face = "bold"),
    axis.title = element_text(size = 11),
    axis.text = element_text(size = 10)
  ) +
  scale_x_continuous(breaks = 1:10) +
  scale_y_continuous(limits = c(min(data_clean$Top1_Epoch_numeric, na.rm = TRUE) - 3, 
                                max(data_clean$Top1_Epoch_numeric, na.rm = TRUE) + 3))

print("=== TOP1 AT 50 EPOCH PLOTS ===")
print("Plot 4: All model-coordination combinations as separate lines")
print(p4)

print("Plot 5: Faceted by coordination (with/without)")
print(p5)

print("Plot 6: Models colored, coordination as line type")
print(p6)

print("=== BEST EPOCH NUMBER PLOTS ===")
print("Plot 7: All model-coordination combinations as separate lines")
print(p7)

print("Plot 8: Faceted by coordination (with/without)")
print(p8)

print("Plot 9: Models colored, coordination as line type")
print(p9)

# Summary statistics for all metrics
print("=== SUMMARY STATISTICS ===")

print("Summary by Model and Coordination - Overall Accuracy:")
summary_overall <- data_clean %>%
  group_by(Model, Coord_factor) %>%
  summarise(
    Mean_Accuracy = round(mean(Top1_Overall_numeric), 2),
    Max_Accuracy = round(max(Top1_Overall_numeric), 2),
    Min_Accuracy = round(min(Top1_Overall_numeric), 2),
    Best_K = K[which.max(Top1_Overall_numeric)],
    .groups = 'drop'
  )
print(summary_overall)

print("Summary by Model and Coordination - Accuracy at 50 Epochs:")
summary_at_50 <- data_clean %>%
  group_by(Model, Coord_factor) %>%
  summarise(
    Mean_Accuracy = round(mean(Top1_at_50_numeric, na.rm = TRUE), 2),
    Max_Accuracy = round(max(Top1_at_50_numeric, na.rm = TRUE), 2),
    Min_Accuracy = round(min(Top1_at_50_numeric, na.rm = TRUE), 2),
    Best_K = K[which.max(Top1_at_50_numeric)],
    .groups = 'drop'
  )
print(summary_at_50)

print("Summary by Model and Coordination - Best Epoch Numbers:")
summary_epochs <- data_clean %>%
  group_by(Model, Coord_factor) %>%
  summarise(
    Mean_Epoch = round(mean(Top1_Epoch_numeric, na.rm = TRUE), 1),
    Max_Epoch = max(Top1_Epoch_numeric, na.rm = TRUE),
    Min_Epoch = min(Top1_Epoch_numeric, na.rm = TRUE),
    Median_Epoch = median(Top1_Epoch_numeric, na.rm = TRUE),
    .groups = 'drop'
  )
print(summary_epochs)