import os
import numpy as np
import pandas as pd
import textwrap

# File code abbreviations
FILE_CODES = {
    "md_detection_tracking": "md",
    "antecedent_analysis": "an",
    "correlation_analysis": "ca",
    "soilmoiture_analysis": "sm",
    "speed_analysis": "sa",
    "heatflux_analysis": "hf",
    "before_after_24hr": "ba",
    "azimuthal_hovmoller": "ho",
}


def std_print(level=None, file_code=None, message="", value=None):
    """
    Args:
        level: 'INFO', 'WARN', 'ERROR', 'DEBUG', 'SAVED', 'EXPORT', None
        file_code: 'md', 'an', 'ca', 'az', 'sm', 'sa', 'hf', 'ts', 'ea', 'ev', None
        message: Message text
        value: Optional value to display
        unit: Optional unit for value
    """
    CLR = {
        "INFO": "\033[92m",  # Green
        "SAVED": "\033[94m",  # Blue
        "EXPORT": "\033[96m",  # Cyan
        "WARN": "\033[93m",  # Yellow
        "DEBUG": "\033[91m",  # Red
        "ERROR": "\033[91m",  # Red
        "RESET": "\033[0m",  # Reset to default
    }

    spacing = max(0, 50 - len(message))
    # CASE 1: No level provided
    if level is None:
        indent = " " * 7
        if value is not None:
            # unit_str = f" {unit}" if unit else ""
            print(f"{indent} {message}{' ':>{spacing}}       {value}")
        else:
            print(f"{indent} {message}")
        return

    # CASE 2: Level provided
    level_upper = str(level).upper()
    color = CLR.get(level_upper, CLR["RESET"])
    level_str = level_upper[:8]
    f_code = file_code if file_code else "??"
    tag = f"{color}{level_str:<6} {f_code}{CLR['RESET']}"

    if value is not None:
        # Format with value
        print(f"[{tag}] {message}{' ':>{spacing}}   {value}")
    else:
        # Format without value
        print(f"[{tag}] {message}")


# TODO:
# def std_print(level=None, file_code=None, message="", value=None):
#     COL1_WIDTH = 55
#     TOTAL_WIDTH = 110
#     # ANSI Colors
#     CLR = {
#         "INFO": "\033[92m",
#         "SAVED": "\033[94m",
#         "EXPORT": "\033[96m",
#         "WARN": "\033[93m",
#         "DEBUG": "\033[91m",
#         "ERROR": "\033[91m",
#         "RESET": "\033[0m",
#     }
#
#     if level:
#         level_upper = str(level).upper()
#         color = CLR.get(level_upper, CLR["RESET"])
#         f_code = file_code if file_code else "??"
#         tag = f"[{color}{level_upper[:6]:<6} {f_code}{CLR['RESET']}] "
#     else:
#         tag = " " * 10  # Indent for non-level prints
#
#     if value is not None:
#         # Format lists/arrays nicely before wrapping
#         if isinstance(value, (list, np.ndarray)):
#             value_str = ", ".join(map(str, value))
#         else:
#             value_str = str(value)
#         left_content = f"{tag}{message}"
#         padding_needed = COL1_WIDTH - len(left_content)
#         first_line_pad = " " * max(1, padding_needed)
#         wrapper = textwrap.TextWrapper(width=TOTAL_WIDTH, initial_indent="", subsequent_indent=" " * (COL1_WIDTH + 2))
#         wrapped_value = wrapper.fill(value_str)
#         print(f"{left_content}{first_line_pad}: {wrapped_value}")
#     else:
#         # If no value, wrap the message itself
#         wrapper = textwrap.TextWrapper(
#             width=TOTAL_WIDTH, initial_indent=tag, subsequent_indent=" " * len(tag)
#         )
#         print(wrapper.fill(message))


def setup_output_dir(base_dir, *subdirs):
    """Create output directory structure.
    Args:
        base_dir: Base output directory
        *subdirs: Subdirectories to create
    Returns:
        str: Full path to deepest subdirectory
    """
    path = os.path.join(base_dir, *subdirs) if subdirs else base_dir
    os.makedirs(path, exist_ok=True)
    std_print("INFO", "io", "Output directory ready", path)
    return path


def export_results_to_csv(df, output_dir, filename, file_code="io"):
    """Export DataFrame to CSV with standardized message"""
    os.makedirs(output_dir, exist_ok=True)
    filepath = os.path.join(output_dir, f"{filename}.csv")
    df.to_csv(filepath, index=False)
    std_print("SAVED", file_code, "Results exported", filepath)
    return filepath
