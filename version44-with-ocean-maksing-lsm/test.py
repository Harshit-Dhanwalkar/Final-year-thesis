import os
import glob
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def generate_thesis_data():
    all_tracks = []
    # 1. Load data from all years
    dirs = sorted(glob.glob("output_*"))
    if not dirs:
        print("No output_* directories found. Please run this in your project root.")
        return

    for d in dirs:
        year_str = d.split("_")[-1]
        csv_path = os.path.join(d, f"MD_tracks_{year_str}.csv")
        if not os.path.exists(csv_path): continue
            
        df = pd.read_csv(csv_path)
        df['time'] = pd.to_datetime(df['time'])
        
        for tid, group in df.groupby('track_id'):
            group = group.sort_values('time')
            duration = (group['time'].max() - group['time'].min()).total_seconds() / 3600
            is_long = duration >= 96
            
            # Extract soil moisture - ensure it's a valid float
            sm = np.nan
            if 'swvl1' in group.columns:
                val = group['swvl1'].iloc[0]
                if pd.notnull(val):
                    sm = float(val)
            
            all_tracks.append({
                'year': int(year_str),
                'duration': duration,
                'is_long_lived': is_long,
                'soil_moisture': sm,
                'is_landfall': group['is_land'].any()
            })
            
    df_stats = pd.DataFrame(all_tracks)
    os.makedirs("Overall_years", exist_ok=True)

    # --- CALCULATE STATISTICS ---
    yearly = df_stats.groupby('year').agg(
        total=('year', 'count'),
        long=('is_long_lived', 'sum'),
        mean_sm=('soil_moisture', 'mean')
    )
    yearly['fraction'] = (yearly['long'] / yearly['total']) * 100
    
    print(f"Total MDs: {len(df_stats)}")
    print(f"Landfalling MDs: {df_stats['is_landfall'].sum()}")
    print(f"Long-lived fraction range: {yearly['fraction'].min():.1f}% to {yearly['fraction'].max():.1f}%")
    
    top_3 = yearly.sort_values('fraction', ascending=False).head(3).index.tolist()
    dry_2 = yearly.sort_values('mean_sm', ascending=True).head(2).index.tolist()
    print(f"Top BOE Years: {top_3}")
    print(f"Dry/Short-lived Years: {dry_2}")

    # --- PLOT 1: ANNUAL TRENDS ---
    fig, ax = plt.subplots(figsize=(11, 6))
    
    # Add Grid for better readability
    ax.grid(True, linestyle='--', alpha=0.5, zorder=0)
    
    # Total Count as Bars
    ax.bar(yearly.index, yearly['total'], color='#AED6F1', label='Total MDs', zorder=3)
    
    # Long-lived Count as a Line (Same Axis)
    ax.plot(yearly.index, yearly['long'], color='#E74C3C', marker='o', 
            linewidth=2, label='Long-lived MDs (≥96h)', zorder=4)
    
    ax.set_ylabel('Number of Events', fontsize=12)
    ax.set_xlabel('Year', fontsize=12)
    ax.set_xticks(yearly.index)
    plt.xticks(rotation=45) # Rotates years for clarity
    
    # Set limit slightly higher than max count for legend room
    ax.set_ylim(0, yearly['total'].max() + 2)
    
    ax.legend(loc='upper left', frameon=True)
    plt.title('Annual Frequency of Total and Long-lived MDs (2001–2025)', fontsize=14)
    plt.tight_layout()
    plt.savefig('Overall_years/annual_count_trend.png', dpi=300)

    # --- PLOT 2: DURATION HISTOGRAM ---
    plt.figure(figsize=(9, 6))
    plt.grid(axis='y', linestyle='--', alpha=0.7, zorder=0)
    plt.hist([df_stats[~df_stats['is_long_lived']]['duration'], 
              df_stats[df_stats['is_long_lived']]['duration']], 
             bins=np.arange(0, df_stats['duration'].max()+24, 24), stacked=True, 
             color=['#3498DB', '#C0392B'], label=['Short-lived (<96h)', 'Long-lived (≥96h)'], 
             zorder=3, edgecolor='white')

    plt.axvline(96, color='black', linestyle='--', linewidth=2, label='96h Threshold')
    plt.xlabel('Duration (Hours)', fontsize=12)
    plt.ylabel('Frequency', fontsize=12)
    plt.legend()
    plt.title('Distribution of MD Lifecycles', fontsize=14)
    plt.savefig('Overall_years/duration_histogram.png', dpi=300)

    # --- PLOT 3: SOIL MOISTURE CORRELATION ---
    corr_data = yearly.dropna(subset=['mean_sm', 'fraction'])
    if len(corr_data) > 1:
        plt.figure(figsize=(8, 6))
        plt.grid(True, linestyle=':', alpha=0.6)
        x, y = corr_data['mean_sm'], corr_data['fraction']
        plt.scatter(x, y, color='darkgreen', edgecolors='k', s=60, label='Yearly Data', zorder=3)
        try:
            z = np.polyfit(x, y, 1)
            p = np.poly1d(z)
            plt.plot(x, p(x), "r--", alpha=0.8, label=f'Linear Trend (R²={np.corrcoef(x,y)[0,1]**2:.2f})')
        except:
            pass
        plt.xlabel('Mean Antecedent Soil Moisture ($m^3/m^3$)', fontsize=12)
        plt.ylabel('Long-lived Fraction (%)', fontsize=12)
        plt.legend()
        plt.title('Soil Moisture vs. Longevity Correlation', fontsize=14)
        plt.savefig('Overall_years/soil_moisture_longevity_corr.png', dpi=300)

if __name__ == "__main__":
    generate_thesis_data()
