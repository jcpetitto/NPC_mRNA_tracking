library(jsonlite)
library(tidyverse)

# Source external script ####
source("./r_utility_fns.R")

# source('/home/jocelyn.tourtellotte-umw/src_yeast_pipeline/R_analysis/r_utility_fns.R') 

# Directory containing calculations for distances between two labels modeling NE
# based on the longest segments only
longest_seg_dist_dir <- "/home/jocelyn.tourtellotte-umw/yeast_output/dual_label_experiments/merged/distances/"
# based on merged NE segments and closed with a bezier curve
merged_dist_dir <- "/home/jocelyn.tourtellotte-umw/yeast_output/dual_label_experiments/no_merge/distances/"

dist_join_by <- join_by(strain, FoV_id)

longest_seg_dist_data <- import_dual_dist_json_by_dir(longest_seg_dist_dir)
longest_seg_dist_data <- left_join(all_experiment_key, longest_seg_dist_data, by = dist_join_by, relationship = "many-to-many")
merged_dist_data <- import_dual_dist_json_by_dir(merged_dist_dir)
merged_dist_data <- left_join(all_experiment_key, merged_dist_data, by = dist_join_by, relationship = "many-to-many")

write_csv(longest_seg_dist_data, "/home/jocelyn.tourtellotte-umw/yeast_output/dual_label_experiments/no_merge/distances/longest_seg_dist_data.csv")
write_csv(merged_seg_reg_data, "/home/jocelyn.tourtellotte-umw/yeast_output/dual_label_experiments/merged/distances/merged_seg_reg_data.csv")