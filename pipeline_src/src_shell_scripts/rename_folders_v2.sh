#!/bin/bash

# --- Configuration ---
# Set the desired width for the zero-padded number (e.g., 4 for FoV_0001).
PADDING_WIDTH=4
# --- End of Configuration ---


# --- Script Input ---
# The script now takes the root directory (containing strain folders) as an argument.
ROOT_DIR="$1"


# --- Input Validation ---
# Check if a root directory was provided on the command line.
if [ -z "$ROOT_DIR" ]; then
    echo "Error: No root directory specified."
    echo "Usage: $0 /path/to/your/root_directory"
    exit 1
fi

# Check if the provided root directory actually exists.
if [ ! -d "$ROOT_DIR" ]; then
    echo "Error: Root directory '$ROOT_DIR' not found."
    exit 1
fi

echo "Starting scan in root directory: $ROOT_DIR"
echo "================================================="


# --- Main Logic ---
# Loop through each item in the root directory (these are the 'strain' folders).
# The `*/` ensures we only process directories.
for strain_path in "$ROOT_DIR"/*/; do
    if [ -d "$strain_path" ]; then
        echo "Processing Strain: $(basename "$strain_path")"

        # Now, loop through each sub-item in the strain folder (the 'batch' folders).
        for batch_path in "$strain_path"/*/; do
            if [ -d "$batch_path" ]; then
                echo "  Processing Batch: $(basename "$batch_path")"

                # Check if any 'cell *' folders exist in the current batch folder.
                # Using 'shopt -s nullglob' makes the pattern expand to nothing if no matches are found.
                shopt -s nullglob
                cell_folders=("$batch_path"cell\ *)
                shopt -u nullglob

                # If the array of cell folders is empty, skip to the next batch.
                if [ ${#cell_folders[@]} -eq 0 ]; then
                    echo "    -> No 'cell *' folders found. Skipping."
                    continue
                fi

                # Loop through all directories starting with "cell " inside the batch folder.
                for old_path in "${cell_folders[@]}"; do
                    old_name=$(basename "$old_path")
                    # Extract the number from the folder name.
                    number=$(echo "$old_name" | sed 's/cell //')

                    # Verify that we extracted a valid number.
                    if [[ "$number" =~ ^[0-9]+$ ]]; then
                        # Format the new name with the "FoV_" prefix and zero-padding.
                        new_name=$(printf "FoV_%0*d" "$PADDING_WIDTH" "$number")
                        new_path="$batch_path$new_name"

                        echo "    - Renaming '$old_name' to '$new_name'"

                        # The 'mv' command performs the rename.
                        mv "$old_path" "$new_path"

                        if [ $? -eq 0 ]; then
                            echo "      -> Success."
                        else
                            echo "      -> ERROR: Failed to rename."
                        fi
                    else
                      echo "    - Skipping '$old_name' (does not match 'cell [number]' format)."
                    fi
                done
            fi
        done
    fi
done

echo "================================================="
echo "Renaming process complete."