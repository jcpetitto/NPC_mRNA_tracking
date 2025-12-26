#  conda activate pyr_yeast_env

#  open R @ CL after
library(reticulate)
library(dplyr)
library(tidyverse)

use_condaenv("yeast_env", required = TRUE)

# json_dir <- "/Users/jctourtellotte/UMass Medical School Dropbox/Jocelyn Tourtellotte/11_Grunwald_Shared_Projects_Folder/Lab_Jocelyn/07_Yeast/src_yeast_pipeline/tracking_experiments/registration/"
# setwd("/home/jocelyn.tourtellotte-umw/src_yeast_pipeline/") <- this is where the packages are found relative to (and json_dir same way...)
# TODO Make this path part of the config
json_dir <- '/home/jocelyn.tourtellotte-umw/yeast_output/dual_label_experiments/registration/'

extract_reg_data <- function(json_dir){
  json_files <- list.files(path = json_dir, pattern = "\\.json$", full.names = TRUE)

  all_df_fov <- list()
  all_df_ne_label <- list()
  all_df_ne_subset <- list()

  for (current_json_path in json_files) {
    cat("Processing:", current_json_path, "\n")

    # depends on FoV_reg_json_to_df from utils.img_registration 
    python_command <- sprintf("from utils.img_registration import FoV_reg_json_to_df\njson_path = r'%s' # Use a raw string literal in Python for the path\ndf_fov, df_ne_label, df_ne_subset = FoV_reg_json_to_df(json_path)", current_json_path)
    tryCatch({
      py_run_string(python_command)
      all_df_fov[[current_json_path]] <- py$df_fov
      all_df_ne_label[[current_json_path]] <- py$df_ne_label
      all_df_ne_subset[[current_json_path]] <- py$df_ne_subset
    })
  }
    cat("\nProcessing complete!\n")
  return(list(
    fov = all_df_fov,
    ne_label = all_df_ne_label,
    ne_subset = all_df_ne_subset
  ))
}

# only return the summary statistics
reg_summ_stat_fn <- function(target_df, group_vec, col_vec, fn_list){
  target_df %>% 
    group_by(across(all_of(group_vec))) %>%
    summarise(across(
      .cols = col_vec,
      .fns = fn_list,
      .names = "{.col}_{.fn}"
    )) %>%
      ungroup()
}

# return target_df with summary stats added as columns
reg_incl_summ_stat_fn <- function(target_df, group_vec, col_vec, fn_list, prefix=""){
  target_df %>%
    group_by(across(all_of(group_vec))) %>%
    mutate(across(
      .cols = col_vec,
      .fns = fn_list,
      .names = paste0(prefix, "_{.fn}_{.col}")
      # or glue::glue("{prefix}_{.fn}_{.col}")
    )) %>%
      ungroup()
}

all_reg_data_list <- extract_reg_data(json_dir)

# registration data for full FoVs
fov_list <- all_reg_data_list$fov
combined_fov_df <- dplyr::bind_rows(fov_list, .id = "source_file")

# registration data for ne_labels by FoV with subsetting
ne_subset_list <- all_reg_data_list$ne_subset
combined_ne_subset_df <- dplyr::bind_rows(ne_subset_list, .id = "source_file")

var_vec <- c(angle, scale, x_shift, y_shift)
stat_list <- list(mean = mean, sd = std)

FoV_ne_label_stats <- reg_incl_summ_stat_fn(combined_ne_subset_df, c(FoV_id, ne_label), var_vec, stat_list, "ne_label")

# FoV_ne_label_stats_mean <- combined_ne_subset_df %>% group_by(FoV_id, ne_label) %>% summarise(scale_mean = mean(scale), angle_mean = mean(angle), x_shift_mean = mean(x_shift), y_shift_mean = mean(y_shift))

# FoV_ne_label_stats_std <- combined_ne_subset_df %>% group_by(FoV_id, ne_label) %>% summarise(scale_std = sd(scale), angle_std = sd(angle), x_shift_std = sd(x_shift), y_shift_std = sd(y_shift))

# combined_ne_subset_df <- left_join(combined_ne_subset_df, FoV_ne_label_stats_std, by = c('FoV_id','ne_label'))
# combined_ne_subset_df <- left_join(combined_ne_subset_df, FoV_ne_label_stats_mean, by = c('FoV_id','ne_label'))

combined_ne_subset_df <- combined_ne_subset_df %>% relocate(source_file, .after = last_col()) %>% relocate(strain, .before=FoV_id) %>% relocate(c(date,batch), .after=strain) %>% relocate(aqsettings, .before=source_file)

# TODO Make this path part of the config
write_csv(combined_ne_subset_df, "/home/jocelyn.tourtellotte-umw/yeast_output/registration/reg_results_combined_ne_subset.csv")


FoV_ne_label_stats_mean <- combined_fov_df %>% group_by(FoV_id) %>% summarise(scale_mean = mean(scale), angle_mean = mean(angle), x_shift_mean = mean(x_shift), y_shift_mean = mean(y_shift))

FoV_ne_label_stats_std <- combined_fov_df %>% group_by(FoV_id) %>% summarise(scale_std = sd(scale), angle_std = sd(angle), x_shift_std = sd(x_shift), y_shift_std = sd(y_shift))

combined_fov_df <- left_join(combined_fov_df, FoV_ne_label_stats_std, by = c('FoV_id'))
combined_fov_df <- left_join(combined_fov_df, FoV_ne_label_stats_mean, by = c('FoV_id'))

combined_fov_df <- combined_fov_df %>% relocate(source_file, .after = last_col()) %>% relocate(strain, .before=FoV_id) %>% relocate(c(date,batch), .after=strain) %>% relocate(aqsettings, .before=source_file)


# TODO Make this path part of the config
write_csv(combined_fov_df, "/home/jocelyn.tourtellotte-umw/yeast_output/registration/reg_results_combined_fov.csv")


# angle_combined_ne_subset_df <- combined_ne_subset_df %>% select("FoV_id", "ne_label", "angle", starts_with("angle") & !ends_with("2std"))
# scale_combined_ne_subset_df <- combined_ne_subset_df %>% select("FoV_id", "ne_label", "scale", starts_with("scale") & !ends_with("2std"))
# x_shift_combined_ne_subset_df <- combined_ne_subset_df %>% select("FoV_id", "ne_label", "x_shift", starts_with("x_shift") & !ends_with("2std"))
# y_shift_combined_ne_subset_df <- combined_ne_subset_df %>% select("FoV_id", "ne_label", "y_shift", starts_with("y_shift") & !ends_with("2std"))

# angle_ne_subset_not2std_df <- angle_combined_ne_subset_df %>% filter(!between(angle, -2*angle_std + angle_mean, 2*angle_std + angle_mean))

# scale_ne_subset_not2std_df <- scale_combined_ne_subset_df %>% filter(!between(scale, -2*scale_std + scale_mean, 2*scale_std + scale_mean))

# x_shift_ne_subset_not2std_df <- x_shift_combined_ne_subset_df %>% filter(!between(x_shift, -2*x_shift_std + x_shift_mean, 2*x_shift_std + x_shift_mean))

# y_shift_ne_subset_not2std_df <- y_shift_combined_ne_subset_df %>% filter(!between(y_shift, -2*y_shift_std + y_shift_mean, 2*y_shift_std + y_shift_mean))

# ne_subset_not2std_df <- left_join(angle_ne_subset_not2std_df, scale_ne_subset_not2std_df, by=c("FoV_id", "ne_label")) %>% left_join(., x_shift_ne_subset_not2std_df, by=c("FoV_id", "ne_label")) %>% left_join(., y_shift_ne_subset_not2std_df, by=c("FoV_id", "ne_label"))