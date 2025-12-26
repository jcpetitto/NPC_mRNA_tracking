#!/bin/bash

# This script recursively finds and renames TIFF files based on a prefix list.
# It navigates a 'root/strain/batch' directory structure.
# Within each 'batch' folder, it looks for 'FoV_NNNN' subfolders.
# Inside each 'FoV_NNNN' folder, it finds files matching a list of prefixes
# and renames them to '[prefix][FoV_number].tiff'.

# --- Configuration ---
# Define the list of valid file prefixes to look for.
# Add or remove prefixes from this list as needed.
PREFIX_LIST=(
    "BF1red"
    "BF1green"
    "BF2red"
    "BF2green"
    "RNAgreen"
    "NEred"
)
# --- End of Configuration ---


# --- Script Input ---
# The script takes the root directory (containing strain folders) as an argument.
ROOT_DIR="$1"


# --- Input Validation ---
if [ -z "$ROOT_DIR" ]; then
    echo "Error: No root directory specified."
    echo "Usage: $0 /path/to/your/root_directory"
    exit 1
fi

if [ ! -d "$ROOT_DIR" ]; then
    echo "Error: Root directory '$ROOT_DIR' not found."
    exit 1
fi

echo "Starting recursive file renaming in root directory: $ROOT_DIR"
echo "=================================================================="

# --- Main Logic ---
# Loop through each 'strain' folder in the root directory.
for strain_path in "$ROOT_DIR"/*/; do
    if [ -d "$strain_path" ]; then
        echo "Scanning Strain: $(basename "$strain_path")"

        # Loop through each 'batch' folder within the strain folder.
        for batch_path in "$strain_path"/*/; do
            if [ -d "$batch_path" ]; then
                echo "  Scanning Batch: $(basename "$batch_path")"

                # Look for 'FoV_' folders within the batch folder.
                for fov_dir in "$batch_path"FoV_*; do
                    if [ -d "$fov_dir" ]; then
                        dir_name=$(basename "$fov_dir")
                        echo "    Processing FoV: $dir_name"

                        # Extract the correct number from the directory name (e.g., "0123")
                        correct_padded_number=${dir_name#FoV_}

                        # Verify that we extracted a valid number from the folder name.
                        if ! [[ "$correct_padded_number" =~ ^[0-9]+$ ]]; then
                            echo "      -> Skipping, could not extract number from folder name."
                            continue
                        fi

                        # Loop through each defined prefix from the configuration list.
                        for prefix in "${PREFIX_LIST[@]}"; do
                            # Use 'shopt -s nullglob' to prevent errors if no files match the pattern.
                            shopt -s nullglob
                            # Find all .tiff files that start with the current prefix.
                            files_to_check=("$fov_dir"/"$prefix"*.tiff)
                            shopt -u nullglob

                            # If no files were found for this prefix, just move to the next prefix.
                            if [ ${#files_to_check[@]} -eq 0 ]; then
                                continue
                            fi

                            # Construct the single, correct target filename for this prefix and FoV.
                            new_filename="${prefix}${correct_padded_number}.tiff"
                            new_filepath="$fov_dir/$new_filename"

                            # Process all files found for the current prefix.
                            for old_filepath in "${files_to_check[@]}"; do
                                old_filename=$(basename "$old_filepath")

                                # Only rename if the current filename is not already the correct one.
                                # This prevents unnecessary operations and handles cases with multiple matching files.
                                if [ "$old_filename" != "$new_filename" ]; then
                                    echo "      -> Prefix '$prefix': Renaming '$old_filename' to '$new_filename'"

                                    # --- RENAME COMMAND ---
                                    # To perform a "dry run" (see what would change without doing it),
                                    # comment out the 'mv' line below by adding a '#' at the start.
                                    mv "$old_filepath" "$new_filepath"

                                    if [ $? -eq 0 ]; then
                                        echo "        -> Success."
                                    else
                                        echo "        -> ERROR: Failed to rename."
                                    fi
                                fi
                            done
                        done
                    fi
                done
            fi
        done
    fi
done

echo "=================================================================="
echo "Recursive file renaming process complete."