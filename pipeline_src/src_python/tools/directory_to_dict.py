#!/usr/bin/env python3
#  built by the Jocelyn Tourtellotte, PhD @ Grunwald Lab, UMass Chan
# 2025-07-11
# for use with yeast image processing pipeline

"""
Scans a directory of yeast imaging data and creates a JSON index file 
FOR EACH EXPERIMENT within a strain folder.

Assumed Directory Structure:
ROOT_DIRECTORY/
├── BMY1234/
│   ├── 2023-10-26_Experiment1/  <-- Experiment sub-folder
│   │   ├── {experiment_dir_prefix}{id_number}/
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
import os
from pathlib import Path

def create_path_dict_entry(experiment_dir, exp_dir) -> dict:
    """Creates a single dictionary entry for one 'experiment' folder."""
    
    match = re.search(r'\d+', experiment_dir.name)
    if not match:
        print(f"    Warning: Could not find digits in folder name: {experiment_dir.name}. Skipping.")
        return None
    digits = match.group(0)

# TODO Make the suffixes etc part of the config for easier set-up re: channels per experiment
    path_dict_entry = {
        'FoV_id': digits,
        'FoV_collection_path': str(exp_dir),
        'imgs': {
            'fn_ch1_reg1':  f'{experiment_dir.name}/BF1red{digits}.tiff',
            'fn_ch2_reg1':  f'{experiment_dir.name}/BF1green{digits}.tiff',
            'fn_ch1_reg2':  f'{experiment_dir.name}/BF2red{digits}.tiff',
            'fn_ch2_reg2':  f'{experiment_dir.name}/BF2green{digits}.tiff',
            'fn_track_ch1': f'{experiment_dir.name}/NEred{digits}.tiff',
            'fn_track_ch2': f'{experiment_dir.name}/RNAgreen{digits}.tiff'
        }
    }
    return path_dict_entry

def process_experiment(exp_dir, experiment_dir_prefix:str="FoV_"):
    """
    Gathers all FoV data for a SINGLE experiment folder.
    'exp_dir' is the path to one specific experiment (e.g., .../BMY1234/2023-10-26_Experiment1)
    """
    experiment_data = {}
    print(f"  -> Processing experiment: {exp_dir.name}")
    
    for fov_dir in sorted(exp_dir.iterdir()):
        if fov_dir.is_dir() and fov_dir.name.startswith(experiment_dir_prefix):
            # The 'experiment_dir' in the original function was actually the fov_dir
            entry = create_path_dict_entry(fov_dir, exp_dir)
            if entry:
                # This is the robust method from the last response. It's correct!
                fov_id = entry.pop('FoV_id') 
                experiment_data[fov_id] = entry
                
    if not experiment_data:
        print(f"    Warning: No '{experiment_dir_prefix}' folders found in {exp_dir.name}.")

    return experiment_data

def process_strains(strain_dirs_to_process, output_dir = None):
    """
    Loops through strain and experiment folders, builds a dictionary of all experiments.
    """
    all_experiments = {}
    for strain_dir in strain_dirs_to_process:
        if not strain_dir.is_dir():
            print(f"\nWarning: Specified strain folder does not exist: {strain_dir.name}")
            continue
        
        print(f"\nScanning strain: {strain_dir.name}")
        
        # Iterate through experiment subdirectories within this strain
        for exp_dir in sorted(strain_dir.iterdir()):
            if exp_dir.is_dir():
                # Get the data for this one experiment by calling the specialist
                experiment_data = process_experiment(exp_dir)

                # If the pipeline is running (output_dir is None), build the main dictionary
                if output_dir is None:
                    if experiment_data: # Ensure it's not empty
                        # Create the unique key, e.g., "BMY1234_2023-10-26_Experiment1"
                        experiment_key = f"{strain_dir.name}_{exp_dir.name}"
                        all_experiments[experiment_key] = experiment_data
                # If running in standalone mode to write files
                else:
                    if experiment_data:
                        # Create the output filename
                        os.makedirs(output_dir, exist_ok=True)
                        output_filename = f"{strain_dir.name}_{exp_dir.name}.json"
                        output_path = os.path.join(output_dir, output_filename)
                        
                        with open(output_path, 'w') as f:
                            json.dump(experiment_data, f, indent=2)
                        print(f"    ✅ Successfully wrote {len(experiment_data)} entries to {output_path}")

    return all_experiments



def confirm_strains(root_dir, strain_list = None):
    if strain_list is not None:
        strain_dirs_to_process = [Path(root_dir) / Path(s) for s in strain_list]
        return strain_dirs_to_process
    else:
        strain_dirs_to_process = sorted([d for d in Path(root_dir).iterdir() if d.is_dir() and d.name.startswith("BMY")])
        return strain_dirs_to_process

    if not strain_dirs_to_process:
        print("No strain directories found to process.")
        return

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
    
    

    # Determine which strain folders to process
    # if args.strains:
    #     strain_dirs_to_process = [args.root_dir / s for s in args.strains]
    # else:
    #     strain_dirs_to_process = sorted([d for d in args.root_dir.iterdir() if d.is_dir() and d.name.startswith("BMY")])

    # if not strain_dirs_to_process:
    #     print("No strain directories found to process.")
    #     return
    strain_dirs_to_process = confirm_strains(args.root_dir, args.strains)

    # Process each selected strain
    process_strains(strain_dirs_to_process, args.output_dir)

    # for strain_dir in strain_dirs_to_process:
    #     if not strain_dir.is_dir():
    #         print(f"\nWarning: Specified strain folder does not exist: {strain_dir.name}")
    #         continue
        
    #     print(f"\nScanning strain: {strain_dir.name}")
        
    #     # Iterate through experiment subdirectories
    #     for exp_dir in sorted(strain_dir.iterdir()):
    #         if exp_dir.is_dir():
    #             process_experiment(exp_dir, args.output_dir)

if __name__ == "__main__":
    main()