#!/usr/bin/env python3
# Prompts and Revisions by Jocelyn Tourtellotte, PhD
# Code by Gemini Pro 2.5
# for use with yeast image processing pipeline
#   built by the Grunwald Lab, UMass Chan
# 2025-07-11

"""
Scans a directory of yeast imaging data and creates a JSON index file 
FOR EACH EXPERIMENT within a strain folder.

Assumed Directory Structure:
ROOT_DIRECTORY/
├── BMY1234/
│   ├── 2023-10-26_Experiment1/  <-- Experiment sub-folder
│   │   ├── {cell_dir_prefix}{id_number}/
│   │   └── ...
│   └── 2023-10-27_Experiment2/
│       └── ...
└── BMY5678/
    └── ...

Running with --output_dir /home/user/json_output will produce:
/home/user/json_output/
├── BMY1234_2023-10-26_Experiment1.json
├── BMY1234_2023-10-27_Experiment2.json
└── ...
"""
import argparse
import json
import re
from pathlib import Path

def create_path_dict_entry(cell_dir: Path, exp_dir: Path) -> dict:
    """Creates a single dictionary entry for one 'cell' folder."""
    
    match = re.search(r'\d+', cell_dir.name)
    if not match:
        print(f"    Warning: Could not find digits in folder name: {cell_dir.name}. Skipping.")
        return None
    digits = match.group(0)

    path_dict_entry = {
        'FoV_id': digits,
        'FoV_collection_path': str(exp_dir),
        'imgs': {
            'fn_reg_npc1':  f'{cell_dir.name}/BF1red{digits}.tiff',
            'fn_reg_rnp1':  f'{cell_dir.name}/BF1green{digits}.tiff',
            'fn_reg_npc2':  f'{cell_dir.name}/BF2red{digits}.tiff',
            'fn_reg_rnp2':  f'{cell_dir.name}/BF2green{digits}.tiff',
            'fn_track_rnp': f'{cell_dir.name}/RNAgreen{digits}.tiff',
            'fn_track_npc': f'{cell_dir.name}/NEred{digits}.tiff'
        }
    }
    return path_dict_entry

def process_experiment(exp_dir: Path, output_dir: Path, cell_dir_prefix: str="FoV_"):
    """
    Gathers all cell data for a single experiment and saves it to a JSON file.
    """
    experiment_data = []
    print(f"  -> Processing experiment: {exp_dir.name}")
    
    for cell_dir in sorted(exp_dir.iterdir()):
        if cell_dir.is_dir() and cell_dir.name.startswith(cell_dir_prefix):
            entry = create_path_dict_entry(cell_dir, exp_dir)
            if entry:
                experiment_data.append(entry)
    
    if not experiment_data:
        print(f"    No '{cell_dir_prefix}' folders found in {exp_dir.name}. No JSON file will be created.")
        return

    # Create the output filename, e.g., "BMY820_2023-10-26.json"
    strain_name = exp_dir.parent.name
    output_filename = f"{strain_name}_{exp_dir.name}.json"
    output_path = output_dir / output_filename
    
    # Write the data for this experiment to its own JSON file
    with open(output_path, 'w') as f:
        json.dump(experiment_data, f, indent=2)
        
    print(f"    ✅ Successfully wrote {len(experiment_data)} entries to {output_path}")

def main():
    """Main function to parse arguments and drive the script."""
    parser = argparse.ArgumentParser(
        description="Generate JSON index files for yeast imaging data strains.",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument(
        "root_dir",
        type=Path,
        help="The root directory containing the BMY* strain folders."
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        required=True, # This makes the argument mandatory
        help="Directory where the output .json files will be saved. Must be writable."
    )
    parser.add_argument(
        "--strains",
        nargs="+",
        default=None,
        help="Optional: A space-separated list of specific strain folders to process."
    )
    
    args = parser.parse_args()

    if not args.root_dir.is_dir():
        print(f"Error: Root directory not found at '{args.root_dir}'")
        return
    
    # Create the output directory if it doesn't exist
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Determine which strain folders to process
    if args.strains:
        strain_dirs_to_process = [args.root_dir / s for s in args.strains]
    else:
        strain_dirs_to_process = sorted([d for d in args.root_dir.iterdir() if d.is_dir() and d.name.startswith("BMY")])

    if not strain_dirs_to_process:
        print("No strain directories found to process.")
        return

    # Process each selected strain
    for strain_dir in strain_dirs_to_process:
        if not strain_dir.is_dir():
            print(f"\nWarning: Specified strain folder does not exist: {strain_dir.name}")
            continue
        
        print(f"\nScanning strain: {strain_dir.name}")
        
        # Iterate through experiment subdirectories
        for exp_dir in sorted(strain_dir.iterdir()):
            if exp_dir.is_dir():
                process_experiment(exp_dir, args.output_dir)

if __name__ == "__main__":
    main()