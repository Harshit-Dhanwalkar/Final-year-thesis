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

# def std_print(level, file_code, message, value=None, unit=None):
#     """
#     Args:
#         level: 'INFO', 'WARN', 'ERROR', 'DEBUG', 'SAVE'
#         file_code: 'md', 'an', 'ca', 'sm', 'sa', 'hf', 'ba', 'ho'
#         message: Message text
#         value: Optional value to display
#         unit: Optional unit for value
#     Example:
#         standardized_print('INFO', 'md', 'Tracks detected', 6, 'tracks')
#         standardized_print('SAVE', 'an', 'Results exported', '/path/file.csv')
#         standardized_print('WARN', 'ca', 'Insufficient data')
#     """
#     # Truncate level to 8 chars to fit format
#     level_str = level[:8]
#
#     if value is not None:
#         # Format with value
#         spacing = max(0, 50 - len(message))
#         unit_str = f" {unit}" if unit else ""
#         print(f" >  [{level_str} {file_code}] {message}{' ':>{spacing}}:          {value}{unit_str}")
#     else:
#         # Format without value
#         print(f" >  [{level_str} {file_code}] {message}")
def std_print(level=None, file_code=None, message="", value=None, unit=None):
    # CASE 1: No level provided - Indent output by 10 characters
    if level is None:
        indent = " " * 10
        if value is not None:
            unit_str = f" {unit}" if unit else ""
            print(f"{indent}{message}: {value}{unit_str}")
        else:
            print(f"{indent}{message}")
        return

    # CASE 2: Level provided - Standardized format
    # Truncate level to 8 chars to fit format
    level_str = str(level)[:8]
    f_code = file_code if file_code else "??"

    if value is not None:
        # Format with value
        spacing = max(0, 50 - len(message))
        unit_str = f" {unit}" if unit else ""
        # Using a fixed width for the bracketed section to keep alignment consistent
        print(f" >  [{level_str:<5} {f_code}] {message}{' ':>{spacing}}          {value}{unit_str}")
    else:
        # Format without value
        print(f" >  [{level_str:<5} {f_code}] {message}")

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
