def calculate_optimal_drift_bins(n_frames, frame_duration_sec, 
                                 typical_drift_timescale_sec=20.0):
    """
    Calculate optimal number of drift bins based on acquisition parameters.
    
    Args:
        n_frames: Total number of frames
        frame_duration_sec: Duration of each frame in seconds
        typical_drift_timescale_sec: Characteristic drift time (default 20s)
        
    Returns:
        n_bins: Optimal number of bins
        
    Reference:
        Mlodzianoski et al. Nat Methods 2011 8(12):1027-1036
        "Sample drift correction in 3D fluorescence photoactivation 
        localization microscopy"
        
        Drift bins should span >10-30 sec to capture stage mechanics
        correlation time while avoiding statistical noise.
    """
    total_duration_sec = n_frames * frame_duration_sec
    
    # Each bin should span at least the drift timescale
    n_bins = int(total_duration_sec / typical_drift_timescale_sec)
    
    # Clamp to reasonable range
    n_bins = np.clip(n_bins, 3, 20)
    
    return n_bins


    rad = (dxi**2 + dyi**2)**(3/2) / abs(dxi*ddyi - dyi*ddxi)
   # Reject if radius < some physical minimum (e.g., 50nm)