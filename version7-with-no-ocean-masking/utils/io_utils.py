import os
import pandas as pd

# File code abbreviations
FILE_CODES = {
    'md_detection_tracking': 'md',
    'antecedent_analysis': 'an',
    'correlation_analysis': 'ca',
    'soilmoiture_analysis': 'sm',
    'speed_analysis': 'sa',
    'heatflux_analysis': 'hf',
    'before_after_24hr': 'ba',
    'azimuthal_hovmoller': 'ho'
}

def std_print(level=None, file_code=None, message="", value=None):
    """
    Args:
        level: 'INFO', 'WARN', 'ERROR', 'DEBUG', 'SAVED', 'EXPORT'
        file_code: 'md', 'an', 'ca', 'sm', 'sa', 'hf', 'ba', 'ho'
        message: Message text
        value: Optional value to display
        unit: Optional unit for value
    """
    CLR = {
        'INFO':   '\033[92m', # Green
        'SAVED':  '\033[94m', # Blue
        'EXPORT': '\033[96m', # Cyan
        'WARN':   '\033[93m', # Yellow
        'DEBUG':  '\033[91m', # Red
        'ERROR':  '\033[91m', # Red
        'RESET':  '\033[0m'   # Reset to default
    }

    spacing = max(0, 50 - len(message))
    # CASE 1: No level provided - Indent output by 11 characters
    if level is None:
        indent = " " * 11
        if value is not None:
            # unit_str = f" {unit}" if unit else ""
            print(f"{indent} {message}{' ':>{spacing}}       {value}")
        else:
            print(f"{indent} {message}")
        return

    # CASE 2: Level provided - Standardized format
    level_upper = str(level).upper()
    color = CLR.get(level_upper, CLR['RESET'])
    level_str = level_upper[:8]
    f_code = file_code if file_code else "??"
    tag = f"{color}{level_str:<6} {f_code}{CLR['RESET']}"

    if value is not None:
        # Format with value
        print(f" >  [{tag}] {message}{' ':>{spacing}}   {value}")
    else:
        # Format without value
        print(f" >  [{tag}] {message}")

def setup_output_dir(base_dir, *subdirs):
    """ Create output directory structure. 
    Args:
        base_dir: Base output directory
        *subdirs: Subdirectories to create
    Returns:
        str: Full path to deepest subdirectory
    """
    path = os.path.join(base_dir, *subdirs) if subdirs else base_dir
    os.makedirs(path, exist_ok=True)

    std_print('INFO', 'io', f'Output directory ready', path)
    return path

def export_results_to_csv(df, output_dir, filename, file_code='io'):
    """Export DataFrame to CSV with standardized message"""
    os.makedirs(output_dir, exist_ok=True)
    filepath = os.path.join(output_dir, f"{filename}.csv")

    df.to_csv(filepath, index=False)

    std_print('SAVED', file_code, f'Results exported', filepath)

    return filepath
