library(tidyverse)
library(dplyr)
library(jsonlite)
library(stringr)

# Source external script
# TODO: more robust way to point to this ??
# source("r_utility_fns.R")

# Registration

# Registration statistics should not vary dramatically between longest segment and merged segments because the cropboxes should (on average?) be approximately the same


# Directory containing registration JSON files for longest NE segment
longest_seg_json_dir <- "/home/jocelyn.tourtellotte-umw/yeast_output/dual_label_experiments/no_merge/registration"
# Directory containing registration JSON files for merged NE segments
merged_json_dir <- "/home/jocelyn.tourtellotte-umw/yeast_output/dual_label_experiments/merged/registration"

# registration results as a list of 3 tibbles, one per level: FoV, ne, and sliced ne
longest_seg_reg_data <- import_reg_json_by_dir(longest_seg_json_dir, TRUE, suffix_to_remove = "reg_results_")
merged_seg_reg_data <- import_reg_json_by_dir(merged_json_dir, TRUE, suffix_to_remove = "reg_results_")

# save to csv: 3 per list <- 1 per level
reg_list_to_csvs(longest_seg_reg_data, "/home/jocelyn.tourtellotte-umw/yeast_output/dual_label_experiments/no_merge/registration/")
reg_list_to_csvs(merged_seg_reg_data, "/home/jocelyn.tourtellotte-umw/yeast_output/dual_label_experiments/merged/registration/")