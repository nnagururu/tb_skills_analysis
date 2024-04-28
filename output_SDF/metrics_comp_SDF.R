library(tidyverse)
library(gtsummary)
library(flextable)
library(ggplot2)
library(reshape2)

#### Table 1 for both gen and agg ####
gen_data <- read.csv('gen_metrics.csv', check.names = FALSE)
gen_table<- gen_data %>% select(`Procedure Duration (s)`, `Bone Voxels Removed`, `Non-Bone Voxels Removed`, `%Time with 6mm Burr (%)`, `%Time with 4mm Burr (%)`,
                             `Number of Strokes`, expert) %>% rename(Experience = expert) %>% 
                              mutate(Experience = factor(Experience, levels = c(0, 1), labels = c("Novice", "Expert")))

gen_table %>%
  tbl_summary(
    by = Experience,
    statistic = list(all_continuous() ~ "{median} ({p25}, {p75})",
                     all_categorical() ~ "{n} ({p})"),
    digits = list(
      all_continuous() ~ 2,
      `Bone Voxels Removed` ~ 0,
      `Non-Bone Voxels Removed` ~ 0,
      `Number of Strokes` ~ 0
    ),
    missing_text = "(Missing)"
  ) %>%
  add_overall %>%
  add_p(pvalue_fun = ~style_pvalue(.x, digits = 2)) %>%
  modify_header(label ~ "**Variable**") %>%
  modify_spanning_header(c("stat_1", "stat_2") ~ "**Expertise**") %>%
  as_flex_table() %>%
  save_as_docx(path = "gen_metrics_SDF.docx")


# Iterate over each bucket and create separate tables
all_buckets_data <- list()

num_buckets <- 5  # Define the number of buckets
for (i in 0:(num_buckets - 1)) {
  # bucket_data <- read.csv("stroke_metrics.csv", check.names = FALSE)
  bucket_data <- read.csv(paste0("bucket_", i, "_metrics.csv"), check.names = FALSE)
  bucket_table <- bucket_data %>%
    select(`Number of Strokes in Bucket`, `Avg Voxels Removed per Stroke`, `Avg Stroke Length (m)`, `Avg Stroke Velocity (m/s)`, 
           `Avg Stroke Acceleration (m/s^2)`, `Avg Stroke Jerk (m/s^3)`, `Avg Stroke Curvature`, `Avg Stroke Force (N)`,
           `Avg Stroke Drill Angle wrt Camera (degrees)`, `Avg Stroke Drill Angle wrt Bone (degrees)`, expert) %>% 
    mutate(
      `Avg Stroke Length (mm)` = `Avg Stroke Length (m)` * 1000,
      `Avg Stroke Velocity (mm/s)` = `Avg Stroke Velocity (m/s)` * 1000,
      `Avg Stroke Acceleration (mm/s^2)` = `Avg Stroke Acceleration (m/s^2)` * 1000,
      `Avg Stroke Jerk (mm/s^3)` = `Avg Stroke Jerk (m/s^3)` * 1000
    ) %>%
    select(
      -`Avg Stroke Length (m)`,
      -`Avg Stroke Velocity (m/s)`,
      -`Avg Stroke Acceleration (m/s^2)`,
      -`Avg Stroke Jerk (m/s^3)`
    ) %>%
    rename(Experience = expert) %>%
    mutate(Experience = factor(Experience, levels = c(0, 1), labels = c("Novice", "Expert")))
  
  temp_data <- bucket_table %>%
    mutate(Bucket = factor(paste("Bucket", i), levels = paste("Bucket", 0:(num_buckets - 1))))
  all_buckets_data[[i + 1]] <- temp_data
  
  bucket_table %>%
    tbl_summary(
      by = Experience,
      statistic = list(all_continuous() ~ "{median} ({p25}, {p75})",
                       all_categorical() ~ "{n} ({p})"),
      # digits = list(
      #   all_continuous() ~ 2,
      #   all_numeric() ~ 0
      # ),
      missing_text = "(Missing)"
    ) %>%
    add_overall %>%
    add_p(pvalue_fun = ~style_pvalue(.x, digits = 2)) %>%
    modify_header(label ~ "**Variable**") %>%
    modify_spanning_header(c("stat_1", "stat_2") ~ "**Expertise**") %>%
    as_flex_table() %>%
    save_as_docx(path = paste0("bucket_", i, "_metrics_SDF.docx"))
    # save_as_docx(pathC = "overall_stroke_metrics_SDF.docx")
  
}


#### Box plots over buckets ####


# Load and prepare data from all buckets
all_buckets_data <- list()

for (i in 0:(num_buckets - 1)) {
  file_path <- paste0("bucket_", i, "_metrics.csv")
    temp_data <- read.csv(file_path, check.names = FALSE) %>%
      mutate(Bucket = factor(paste("Bucket", i), levels = paste("Bucket", 0:(num_buckets - 1))))
    all_buckets_data[[i + 1]] <- temp_data
}

# Combine all data into one dataframe
combined_data <- bind_rows(all_buckets_data)

# Create a function to generate box plots for each metric
plot_metric_over_time <- function(metric_name) {
  p <- ggplot(combined_data, aes(x = Bucket, y = get(metric_name))) +
    geom_boxplot() +
    labs(title = paste("Boxplot of", metric_name, "Over Time"),
         x = "Bucket (Ordinal Time Intervals)",
         y = metric_name) +
    theme_minimal()
  
  print(p)
}

# Metrics to plot
metrics_list <- c("Avg Voxels Removed per Stroke", "Avg Stroke Length (mm)",
                  "Avg Stroke Velocity (mm/s)", "Avg Stroke Acceleration (mm/s^2)",
                  "Avg Stroke Jerk (mm/s^3)", "Avg Stroke Curvature", 
                  "Avg Stroke Force (N)", "Avg Stroke Drill Angle wrt Camera (degrees)",
                  "Avg Stroke Drill Angle wrt Bone (degrees)")

# Generate a plot for each metric
lapply(metrics_list, plot_metric_over_time)

