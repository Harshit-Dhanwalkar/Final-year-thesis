import pandas as pd
import os
import glob
import re

def analyze_md_lifecycles(root_dir="."):
    files = glob.glob(os.path.join(root_dir, "output_*", "MD_tracks_*.csv"))

    if not files:
        print("No track files found.")
        return

    track_list = []
    for f in files:
        year_match = re.search(r"output_(\d{4})", f)
        year = int(year_match.group(1)) if year_match else 0

        try:
            df_year = pd.read_csv(f)
            df_year['time'] = pd.to_datetime(df_year['time'], errors='coerce')
            for tid, group in df_year.groupby('track_id'):
                group = group.sort_values('time').dropna(subset=['time'])
                if group.empty: continue

                # Metrics
                duration = (group['time'].max() - group['time'].min()).total_seconds() / 3600
                avg_speed = group['speed_kmh'].mean()

                # Inland Penetration
                land_points = group[group['is_land'] == True]
                if not land_points.empty:
                    penetration = land_points['distance_km'].max() - land_points['distance_km'].min()
                else:
                    penetration = 0

                origin_is_land = bool(group.iloc[0]['is_land'])
                made_landfall = group['is_land'].any() if not origin_is_land else True
                track_list.append({
                    'year': year,
                    'duration': duration,
                    'avg_speed': avg_speed,
                    'penetration': penetration,
                    'is_long_lived': duration >= 96,
                    'origin': 'Land' if origin_is_land else 'Ocean',
                    'landfall': made_landfall
                })
        except Exception as e:
            print(f"Error in {f}: {e}")

    df = pd.DataFrame(track_list)
    yearly_counts = df.groupby('year').size()

    def get_stats(subset):
        return {
            'N': len(subset),
            'mean_dur': subset['duration'].mean(),
            'std_dur': subset['duration'].std(),
            'mean_speed': subset['avg_speed'].mean(),
            'mean_pen': subset['penetration'].mean(),
            'boe_perc': (len(subset[subset['origin'] == 'Ocean']) / len(subset)) * 100 if len(subset) > 0 else 0
        }

    all_stats = get_stats(df)
    short_stats = get_stats(df[df['is_long_lived'] == False])
    long_stats = get_stats(df[df['is_long_lived'] == True])

    print(f"Confirmed Events (N): {all_stats['N']}")
    print(f"Annual Range: {yearly_counts.min()} to {yearly_counts.max()} per year")
    print(f"Ocean Originated (Landfall Candidates): {len(df[df['origin'] == 'Ocean'])}")
    print(f"Long-lived Count: {long_stats['N']} ({long_stats['N']/all_stats['N']:.1%})")
    print(f"Penetration Difference: {long_stats['mean_pen'] - short_stats['mean_pen']:.1f} km")

    row_fmt = "{:<30} | {:<10} | {:<12} | {:<10}"
    print(row_fmt.format("Statistic", "All MDs", "Short-lived", "Long-lived"))
    print("-" * 75)
    for label, key in [("Count", "N"), ("Mean Dur", "mean_dur"), ("Std Dur", "std_dur"), 
                       ("Inland Pen", "mean_pen"), ("Trans Speed", "mean_speed"), ("BOE %", "boe_perc")]:
        print(row_fmt.format(label, 
                             f"{all_stats[key]:.1f}", 
                             f"{short_stats[key]:.1f}", 
                             f"{long_stats[key]:.1f}"))

if __name__ == "__main__":
    analyze_md_lifecycles()
