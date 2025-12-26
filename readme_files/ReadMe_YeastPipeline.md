


# Directory Structure
All FoVs associated with a given experiment are contained within the same folder.

Results files are produced either by experiment and/or by FoVs

Example of a shell script for the batch renaming of folders can be found in the `src_shell_scripts` folder.
For example, "cell 101" will be renamed "FoV_0101".


# Channel labels
channel 1 - labels NPC/NE
channel 2 - " " for dual label OR labels mRNA for tracking





# Flow
Python -> JSON (file type selected because easy to load back into the pipeline) # TODO impliment this
JSON -> R script -> csv (aggregated data)
                OR -> data analysis (via R script)
csv -> can be pulled in Quarto document for data analyis