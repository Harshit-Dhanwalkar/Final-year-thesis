import pandas as pd

def standardize_time_coord(ds):
    """Standardize dataset to use 'time' coordinate (rename 'valid_time' if needed)"""
    if 'valid_time' in ds.coords and 'time' not in ds.coords:
        ds = ds.rename({'valid_time': 'time'})

    return ds


def get_time_window(track_df, hours_before=24, hours_after=6):
    """Get time window relative to landfall (or track midpoint if no landfall)
    Args:
        track_df: DataFrame with track points and optional 'landfall_time'
        hours_before: Hours before reference time
        hours_after: Hours after reference time
    Returns:
        tuple: (t_start, t_end, reference_time)
    """
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
    """
    Validate and adjust time window to dataset bounds.

    Args:
        t_start, t_end: Time window bounds (pd.Timestamp)
        ds_times: Available times in dataset (pd.DatetimeIndex)

    Returns:
        tuple: (t_start_adjusted, t_end_adjusted, is_valid)
    """
    ds_min = pd.to_datetime(ds_times.min())
    ds_max = pd.to_datetime(ds_times.max())

    # Adjust to dataset bounds
    t_start_adj = max(t_start, ds_min)
    t_end_adj = min(t_end, ds_max)

    # Check if window is valid
    is_valid = t_start_adj < t_end_adj

    return t_start_adj, t_end_adj, is_valid
