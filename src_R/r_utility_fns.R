# Note: load required packages in the main script (the one from which this script is source)

NUP_COLOR_PALETTE = c("#128A6A", "#176E9C", "#C1872B", "#652D90")
MRNA_COLOR_PALETTE = c("#3b528b", "#21918C", "#440154")
SPATIAL_COLOR_PALETTE = c("#26A9E0", "#FFF000")
SIG_BG_COLOR_PALETTE = c("#0D0887", "#9C179E", "#F7941D")
BG_CORRECTED_COLOR_PALETTE = c("#FBB61A", "#BC3754", "#781C6D")







#  Statistics ####

# example value re: col_to_summ <- c(scale_diff, angle_diff, shift_x_diff, shift_y_diff)
calc_summ_stats <- function(data, col_to_summ)
    data %>%
    summarize(
        n = n(),
        across(
            .cols = {{ col_to_summ }},
            .fns = list(
                mean = mean,
                sd = sd,
                median = median,
                IQR = ~IQR(.x, na.rm = TRUE)
                ),
        .names = "{.col}_{.fn}"
        ))

# Plotting ####
## grid of histograms ####
create_histogram_plot_grid <- function(data, plot_var, group_vars, x_limit_method = "minmax", num_cols = 3) {
    # Validate input
    stopifnot(is.data.frame(data))
    stopifnot(plot_var %in% names(data))
    stopifnot(all(group_vars %in% names(data)))

    create_histogram <- function(df, bin_width, plot_title, x_var_col, x_lims, y_lims) {
        mean_col_name <- paste0(x_var_col, "_mean")
        sd_col_name <- paste0(x_var_col, "_sd")

        mean_val <- df[[mean_col_name]][1]
        sd_val <- df[[sd_col_name]][1]

        ggplot(df, aes(x = .data[[x_var_col]])) +
        geom_histogram(binwidth = bin_width, fill = "cornflowerblue", color = "cornflowerblue") +
        # stat_function(
        #     fun = dnorm, 
        #     args = list(mean = mean_val, sd = sd_val), 
        #     linewidth = 1, 
        #     color = "red"
        # ) +
        coord_cartesian(xlim = x_lims, ylim = y_lims, expand = FALSE) + 
        labs(title = plot_title, x = x_var_col, y = "Count") +
        scale_y_continuous(expand = expansion(mult = c(0, 0.05))) +
        theme_minimal(base_size = 10)
    }

    # Calculate global axis limits
    plot_vector <- data[[plot_var]]

    if (x_limit_method == "3sd") {
        mean_val <- mean(plot_vector, na.rm = TRUE)
        sd_val <- sd(plot_vector, na.rm = TRUE)
        x_limits <- c(mean_val - 3 * sd_val, mean_val + 3 * sd_val)
    } else {
        x_limits <- c(min(plot_vector, na.rm = TRUE), max(plot_vector, na.rm = TRUE))
    }

    global_bin_width <- 3.49 * sd(plot_vector, na.rm = TRUE) * (length(plot_vector)^(-1/3))

    max_y_count <- data %>%
        mutate(bin = cut_width(.data[[plot_var]], width = global_bin_width, boundary = 0)) %>%
        count(across(all_of(group_vars)), bin) %>%
        pull(n) %>%
        max(na.rm = TRUE)

    y_limits <- c(0, max_y_count)

    plot_list <- data %>%
        group_by(across(all_of(group_vars))) %>%
        nest() %>%
        ungroup() %>%
        mutate(bin_w = map_dbl(data, ~ 3.49 * sd(.[[plot_var]], na.rm = TRUE) * (length(.[[plot_var]])^(-1/3)))) %>%
        mutate(plot_title = do.call(paste, c(across(all_of(group_vars)), sep = " | "))) %>%
        mutate(plot = pmap(
            .l = list(data, bin_w, plot_title),
            .f = create_histogram,
            x_var_col = plot_var,
            x_lims = x_limits,
            y_lims = y_limits
        )) %>%
        pull(plot)

    return(wrap_plots(plot_list, ncol = num_cols))
}

# Loading CSV Data ####

# experiment_key_df is expected to include the join_by variables
load_results_csv <- function(path, strains_to_include, experiment_key_df, join_by_list = c("strain","FoV_id")){
    
    vars_to_join_by <- join_by(!!!join_by_list)
    new_df <- read_csv(path) %>%
                dplyr::filter(strain %in% strains_to_include) %>%
                select(!addl_info)
    
    # TODO: add check for join_by variables in both dataframes
    new_df <- left_join(experiment_key_df, new_df, by = vars_to_join_by, relationship = "many-to-many")
    return(new_df)
}

# strains_to_include - filter for this when importing the csv

prepare_reg_data_for_plotting <- function(data, grouping_vars, summary_vars, strains_to_include, title_cols) {

    summary_stats <- data %>%
        group_by(across(all_of(grouping_vars))) %>%
        calc_summ_stats(col_to_summ = all_of(summary_vars)) %>%
        ungroup() %>%
        rename(ne_count = n)

    data_to_plot <- data %>%
        left_join(summary_stats, by = grouping_vars) %>%
        # Use the new argument to create the 'exper' column programmatically
        unite("exper", all_of(title_cols), sep = " | ", remove = FALSE)

    return(data_to_plot)
}

# generate_analysis_outputs <- function(data, plot_var, group_vars, strains_to_include, seg_method_name, granularity_name) {

# data_to_plot <- prepare_reg_data_for_plotting(
#     data = data,
#     grouping_vars = group_vars,
#     summary_vars = plot_var, # Note: we only summarize the one variable of interest
#     strains_to_include = strains_to_include
#   )

#   # 2. Create the summary table
#   summary_table <- data_to_plot %>%
#     # Select only the distinct summary rows
#     distinct(across(all_of(c(group_vars, "ne_count"))), .keep_all = TRUE) %>%
#     # Select relevant columns for the table
#     select(all_of(group_vars), ne_count, starts_with(plot_var)) %>%
#     kbl(
#       table.attr = "class='summary-table'",
#       booktabs = TRUE,
#       caption = glue("Summary: {plot_var} ({seg_method_name} at {granularity_name})")
#     ) %>%
#     kable_paper("hover", fixed_thead = T) %>%
#     kable_styling(full_width = FALSE)

#   # 3. Create the histogram plot grid
#   plot_grid <- create_histogram_plot_grid(
#     data = data_to_plot,
#     plot_var = plot_var,
#     group_vars = group_vars
#   )

#   # 4. Return both objects in a named list
#   return(list(table = summary_table, plot = plot_grid))
# }



# For Cluster ####
# can be used locally, but easier on the HPC when all experiments are in play
## Unpacking: JSONS -> tidy_df #####

# TODO see if/how these can be integrated into one function with a switch re: type or some other form of flexibility
## String Handling
split_source_file <- function(data, src_into = c("mode", "strain", "addl_info")){
    cleaned_data <- data %>%
        mutate(source_file = str_replace(source_file, "(_[A-Z0-9]+)_\\1", "\\1")
    )
    separated_data <- cleaned_data %>%
        separate(
            col = source_file,
            into = {{ src_into }},
            sep = "_",
            remove = TRUE,
            extra = "merge",
            fill = "right"
        ) %>%
        relocate(strain, .before = everything())
    return(separated_data)
}

# Grab three levels of subdirectories
#     variable_names - list of variable names for the new column headings
#                                         given first subdirectory first
get_subdirs_three_levels <- function(start_path, variable_names) {
    # --- Validate Input ---
    if (!dir.exists(start_path)) {
        warning("The specified path does not exist or is not a directory.")
        return(tibble()) # Return an empty tibble
    }

    # --- Find and Process Directories ---
    # Recursively list all directories with full paths
    all_dirs <- list.dirs(path = start_path, recursive = TRUE, full.names = TRUE)

    # Create a tidy data frame and process the paths
    tibble(full_path = all_dirs) %>%
        # Remove the starting path itself from the list
        filter(full_path != start_path) %>%
        # Create a relative path by removing the starting path prefix
        mutate(
            relative_path = str_remove(full_path, fixed(paste0(start_path, "/")))
        ) %>%
        # Calculate the depth by counting the number of directory separators
        mutate(
            depth = str_count(relative_path, "/") + 1
        ) %>%
        # Keep only the directories that are exactly 3 levels deep
        filter(depth == 3) %>%
        # Separate the relative_path into three new columns
        separate(
            col = relative_path,
            into = variable_names,
            sep = "/",
            remove = TRUE # We don't need the relative_path column anymore
        )
}

### Registration ####

import_reg_json_by_dir <- function(dir_path, has_two_modes = FALSE, suffix_to_remove="reg_results_"){
    json_files <- list.files(
        path = dir_path,
        pattern = "\\.json$",
        full.names = TRUE
    )

    FoV_registration_data <- json_files %>%
        set_names(basename(.)) %>% 
        map_dfr(unpack_FoV_reg, .id = "source_file") %>%
        mutate(source_file = str_remove(source_file, suffix_to_remove)) %>%
        mutate(source_file = str_remove(source_file, ".json"))

    ne_label_registration_data <- json_files %>%
        set_names(basename(.)) %>%
        map_dfr(unpack_ne_label_reg, .id = "source_file") %>%
        mutate(source_file = str_remove(source_file, suffix_to_remove)) %>%
        mutate(source_file = str_remove(source_file, ".json"))
    slice_registration_data <- json_files %>%
        set_names(basename(.)) %>%
        map_dfr(unpack_slice_reg, .id = "source_file") %>%
        mutate(source_file = str_remove(source_file, suffix_to_remove)) %>%
        mutate(source_file = str_remove(source_file, ".json"))

        if(has_two_modes){
            FoV_registration_data <- split_source_file(FoV_registration_data, c("mode", "strain", "addl_info"))
            ne_label_registration_data <- split_source_file(ne_label_registration_data, c("mode", "strain", "addl_info"))
            slice_registration_data <- split_source_file(slice_registration_data, c("mode", "strain", "addl_info"))
        }

    return(list(FoV_reg = FoV_registration_data, ne_label_reg = ne_label_registration_data, ne_slice_reg = slice_registration_data))        
}

unpack_FoV_reg <- function(file_path) {
    if (!file.exists(file_path)) {
        cat("❌ Error: File not found at path:", file_path, "\n")
        return(NULL)
    }

    data <- fromJSON(file_path)

    tidy_df <- map_dfr(data, .id = "FoV_id", ~ {
    # The FoV data is at the top level of the .x object
        tibble(
            scale = .x$scale,
            angle = .x$angle,
            shift_y = .x$shift_vector[1],
            shift_x = .x$shift_vector[2]
        )
    })

    return(tidy_df)
}

unpack_ne_label_reg <- function(file_path) {
    if (!file.exists(file_path)) {
        cat("❌ Error: File not found at path:", file_path, "\n")
        return(NULL)
    }

    data <- fromJSON(file_path)

    tidy_df <- map_dfr(data, .id = "FoV_id", ~ {
        # Filter .x to *only* keep the ne_label sub-lists.
        # We do this by checking which elements are themselves lists.
        ne_label_lists <- .x[sapply(.x, is.list)]
        
        map_dfr(ne_label_lists, .id = "ne_label_id", ~ {
            # The ne_label reg data is at the top level of *this* object
            tibble(
                scale = .x$scale,
                angle = .x$angle,
                shift_y = .x$shift_vector[1],
                shift_x = .x$shift_vector[2]
            )
        })
    })

    return(tidy_df)
}

unpack_slice_reg <- function(file_path) {
    if (!file.exists(file_path)) {
        cat("❌ Error: File not found at path:", file_path, "\n")
        return(NULL)
    }

    data <- fromJSON(file_path)

    tidy_df <- map_dfr(data, .id = "FoV_id", ~ {
        # Filter .x to *only* keep the ne_label sub-lists
        ne_label_lists <- .x[sapply(.x, is.list)]
        
        map_dfr(ne_label_lists, .id = "ne_label_id", ~ {
            # Filter .x (now ne_label content) to *only* keep the slice sub-lists
            slice_lists <- .x[sapply(.x, is.list)]
            
            map_dfr(slice_lists, .id = "slice_id", ~ {
                # .x is now the "slice_01" content
                tibble(
                    scale = .x$scale,
                    angle = .x$angle,
                    shift_vector = list(.x$shift_vector) # Create list-column
                )
            })
        })
    }) %>%
        unnest_wider(shift_vector, names_sep = "_") %>% # This will now work
        rename(shift_y = shift_vector_1, shift_x = shift_vector_2)

    return(tidy_df)
}

reg_list_to_csvs <- function(reg_list, output_dir_path, midfix = "dual_label"){
    list_names <- names(merged_seg_reg_data)
    walk2(reg_list, list_names, \(.x, .y) write_csv(.x, paste0(output_dir_path, paste(.y,"_",midfix,".csv"))))
}


### Dual Distance #### 

import_dual_dist_json_by_dir <- function(dir_path, suffix_to_remove="dual_dist_result_"){
    json_files <- list.files(
        path = dir_path,
        pattern = "\\.json$",
        full.names = TRUE
    )

    ne_label_dist_data <- json_files %>%
        set_names(basename(.)) %>% 
        map_dfr(dual_dist_from_json_tidy, .id = "source_file") %>%
        mutate(source_file = str_remove(source_file, suffix_to_remove)) %>%
        mutate(source_file = str_remove(source_file, ".json"))
    ne_label_dist_data <- split_source_file(ne_label_dist_data, src_into = c("strain", "addl_info"))

    return(ne_label_dist_data)
}

dual_dist_from_json_tidy <- function(file_path) {
    if (file.exists(file_path)) {
        cat("✅ File found successfully!\n\n")

        data <- fromJSON(file_path)

        # The tidyverse approach to replace the nested for-loops
        tidy_df <- map_dfr(data, ~ {
                # .x is the content for each FoV_id
                map_dfr(.x, ~ {
                # .x is now the content for each ch1_ne_label
                # The innermost map creates a tibble from the distance vector
                map_dfr(.x, ~ tibble(dist = .x), .id = "ch2_ne_label")
            }, .id = "ch1_ne_label") # .id captures the name of the list element as a new column
        }, .id = "FoV_id") %>% # .id captures the top-level name (FoV_id)
            group_by(FoV_id, ch1_ne_label, ch2_ne_label) %>%
            mutate(dist_idx = row_number() - 1) %>% # TODO minamally important -> convert to int. Index correlates to position if this were in python
            ungroup()
        return(tidy_df)

    } else {
        # If the file is not found, this error message will print.
        cat("❌ Error: File NOT FOUND at the path you provided.\n\n")
        cat("Please double-check the following path for any typos or errors:\n")
        print(file_path)
        return(NULL) # Return NULL or an empty tibble on error
    }
}

