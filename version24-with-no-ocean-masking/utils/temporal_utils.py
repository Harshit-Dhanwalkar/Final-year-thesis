import pandas as pd

def get_time_coord(ds):
    """Standardize dataset to use 'time' coordinate (rename 'valid_time' if needed)"""
    # Check if 'time' already exists
    if 'time' in ds.coords:
        return ds, 'time'
    # Check for 'valid_time' and rename it
    if 'valid_time' in ds.coords:
        ds = ds.rename({'valid_time': 'time'})
        return ds, 'time'
    # If neither exists, try to find a time-like coordinate
    time_candidates = [coord for coord in ds.coords if 'time' in coord.lower()]
    if time_candidates:
        time_coord = time_candidates[0]
        if time_coord != 'time':
            ds = ds.rename({time_coord: 'time'})
        return ds, 'time'
    raise ValueError("No time coordinate found in dataset. Available coords: " + str(list(ds.coords)))

def get_time_window(track_df, hours_before=24, hours_after=6):
    """Get time window relative to landfall (or track midpoint if no landfall)"""
    track_times = pd.to_datetime(track_df['time'])

    # Try to use landfall if available
    if 'landfall_time' in track_df.columns:
        landfall_rows = track_df[track_df['landfall_time'].notna()]
        if not landfall_rows.empty:
            ref_time = pd.to_datetime(landfall_rows.iloc[0]['landfall_time'])
        else:
            ref_time = track_times.median()
    else:
        ref_time = track_times.median()

    t_start = ref_time - pd.Timedelta(hours=hours_before)
    t_end = ref_time + pd.Timedelta(hours=hours_after)
    return t_start, t_end, ref_time

def validate_time_bounds(t_start, t_end, ds_times):
    """Validate and adjust time window to dataset bounds"""
    ds_min = pd.to_datetime(ds_times.min())
    ds_max = pd.to_datetime(ds_times.max())

    # Adjust to dataset bounds
    t_start_adj = max(t_start, ds_min)
    t_end_adj = min(t_end, ds_max)

    # Check if window is valid
    is_valid = t_start_adj < t_end_adj
    return t_start_adj, t_end_adj, is_valid
