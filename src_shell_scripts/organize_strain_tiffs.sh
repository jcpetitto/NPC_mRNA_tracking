#!/bin/sh

# A script to rename TIFF files with zero-padded numbers and group them into FoV folders.
# This version is written for maximum portability to work with /bin/sh.

# --- EDIT THIS LIST ---
# Define the specific strain folders you want to process.
STRAINS_TO_PROCESS="BMY1409 BMY1410 BMY1914 BMY1915"

# --- VALIDATE COMMAND-LINE INPUT ---
if [ -z "$1" ]; then
    echo "Error: Please provide the path to the folder that contains your strain folders."
    echo "Usage: ./organize_tiffs.sh /path/to/your/data"
    exit 1
fi

CONTAINING_FOLDER="$1"

if [ ! -d "$CONTAINING_FOLDER" ]; then
    echo "Error: The containing folder '$CONTAINING_FOLDER' was not found."
    exit 1
fi

echo "Starting file organization in: $CONTAINING_FOLDER"
echo "Targeting strains: $STRAINS_TO_PROCESS"

# --- PROCESS THE SPECIFIED FOLDERS ---
for strain_name in $STRAINS_TO_PROCESS; do
    strain_dir_path="${CONTAINING_FOLDER}/${strain_name}"

    if [ ! -d "$strain_dir_path" ]; then
        echo "--> WARNING: Strain folder '$strain_dir_path' not found. Skipping."
        continue
    fi

    echo "-> Processing strain: $strain_name"

    for date_dir in "$strain_dir_path"/*; do
        if [ ! -d "$date_dir" ]; then
            echo "  -> No date sub-directories found in '$strain_name'."
            continue
        fi

        echo "  -> Processing date folder: ${date_dir}"

        (
            cd "$date_dir" || exit

            if ! ls *.tiff 1>/dev/null 2>&1; then
                echo "    - No .tiff files found to process."
                continue
            fi

            echo "    - Renaming files..."
            for file in *.tiff; do
                # Use portable sed to extract the prefix and number
                num_str=$(echo "$file" | sed -E 's/.*[^0-9]([0-9]+)\.tiff$/\1/')
                prefix=$(echo "$file" | sed -E "s/([0-9]+)\\.tiff$//")
                
                # Check if sed was successful in extracting a number
                if [ -n "$num_str" ]; then
                    padded_num=$(printf "%04d" "$num_str")
                    new_name="${prefix}${padded_num}.tiff"
                    
                    if [ "$file" != "$new_name" ]; then
                        echo "Original: $file -> New: $new_name"
                        mv "$file" "$new_name"
                    fi
                fi
            done

            echo "    - Grouping files into FoV folders..."
            unique_nums=$(ls *.tiff 2>/dev/null | grep -oE '[0-9]{4}\.tiff$' | sed 's/\.tiff//' | sort -u)

            if [ -z "$unique_nums" ]; then
                echo "    - No correctly formatted .tiff files to group."
                continue
            fi

            for num in $unique_nums; do
                fov_folder="FoV_${num}"
                echo "Creating folder and moving files for: ${fov_folder}" 
                mkdir -p "$fov_folder"
                mv *"${num}".tiff "$fov_folder/"
            done
        )
    done
done

echo "Organization complete."