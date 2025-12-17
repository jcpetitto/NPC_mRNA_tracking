# Source external script
# source("./r_utility_fns.R")

# Reference Table from Directory ---- 
# TODO get paths from the config json
# make DF that associates: strain, date, FoV_id
# write that DF to a csv
# source('/home/jocelyn.tourtellotte-umw/src_yeast_pipeline/R_analysis/r_utility_fns.R') 

# TODO Test the edits (addl_info column added)
build_experiment_key <- function(root_dir, output_dir, file_name_prefix="dual_label_exper", names_for_levels = c("strain", "date", "addl_info","FoV_id")){
    all_experiment_key <- get_subdirs_three_levels(start_path = "/pi/david.grunwald-umw/data/yeast_data/npc_dual_label/", c("strain", "date", "FoV_id")) %>%
            separate(
                col = date,
                into = c("date","addl_info"),
                sep = " ",
                extra = "merge",
                remove = TRUE,
                fill = "right"
                ) %>%
        mutate(FoV_id = str_remove(FoV_id, "FoV_")) %>%
        dplyr::select(strain, date, addl_info, FoV_id)

    write_csv(all_experiment_key, "/home/jocelyn.tourtellotte-umw/yeast_output/dual_label_experiments/ _table.csv")
}



# loading after downloading from the cluster; can skip this for end-to-end work on the cluster
# all_experiment_key <- read_csv("/Users/jctourtellotte/UMass Medical School Dropbox/Jocelyn Tourtellotte/11_Grunwald_Shared_Projects_Folder/Lab_Jocelyn/07_Yeast/yeast_output/dual_label_experiments/dual_label_exper_table.csv")
# all_experiment_key <- read_csv("/home/jocelyn.tourtellotte-umw/yeast_output/dual_label_experiments/dual_label_exper_table.csv")