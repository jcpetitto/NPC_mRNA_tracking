# New script: analyze_debug_data.py
import torch
import numpy as np
import matplotlib.pyplot as plt

try:
    data = torch.load("debug_theseus_inputs.pt")
    # --- UPDATED FAILING INDEX ---
    failing_index = 47 
    
    initial_guess = data["initial_guess"]
    intensities = data["intensities"]
    
    print("--- ANALYSIS OF BATCH ELEMENT 47 ---")
    
    print("\nInitial Guess:")
    guess_47 = initial_guess[failing_index]
    print(guess_47)
    if torch.isnan(guess_47).any() or torch.isinf(guess_47).any():
        print("\n!! WARNING: NaN or Inf detected in initial guess. !!")

    print("\nIntensity Profile:")
    intensities_47 = intensities[failing_index]
    print(intensities_47)
    if torch.isnan(intensities_47).any() or torch.isinf(intensities_47).any():
        print("\n!! WARNING: NaN or Inf detected in intensity profile. !!")
    if torch.all(intensities_47 == 0):
        print("\n!! WARNING: Intensity profile is all zeros. !!")

    # Plot the problematic data
    plt.figure(figsize=(8, 6))
    plt.title(f"Intensity Profile for Failing Element #{failing_index}")
    plt.plot(intensities_47.numpy(), marker='o')
    plt.xlabel("Sample Index along Normal Line")
    plt.ylabel("Pixel Intensity")
    plt.grid(True)
    plt.savefig("failing_intensity_profile.png")
    print("\nSUCCESS: Saved a plot of the failing intensity profile to 'failing_intensity_profile.png'")

except FileNotFoundError:
    print("ERROR: 'debug_theseus_inputs.pt' not found. Did the main script run and save the file?")
except IndexError:
    print(f"ERROR: The batch size was less than {failing_index + 1}, so the failing element couldn't be analyzed.")