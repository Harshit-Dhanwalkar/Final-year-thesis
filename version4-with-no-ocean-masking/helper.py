#!/usr/bin/env python
"""
INTEGRATION HELPER SCRIPT
Helps you track progress through utils integration
Run this to verify each step of the integration process
"""

import os
import re
import sys
from pathlib import Path

class IntegrationHelper:
    """Track utils integration progress"""
    def __init__(self):
        self.project_root = Path.cwd()
        self.files_to_integrate = {
            'md_detection_tracking.py': 'md',
            'antecedent_analysis.py': 'an',
            'correlation_analysis.py': 'ca',
            'soilmoiture_analysis.py': 'sm',
            'speed_analysis.py': 'sa',
            'heatflux_analysis.py': 'hf',
            'before_after_24hr.py': 'ba',
            'azimuthal_hovmoller.py': 'ho'
        }
    
    def check_utils_installed(self):
        """Verify utils directory exists and has all modules"""
        print("\n--- STEP 0: VERIFY UTILS INSTALLATION ---")
        utils_dir = self.project_root / 'utils'
        required_files = [
            '__init__.py', 'track_utils.py',
            'spatial_utils.py', 'temporal_utils.py', 'io_utils.py', 'plot_utils.py'
        ]
        if not utils_dir.exists():
            print(f"❌ Utils directory not found at {utils_dir}")
            return False
        
        missing = [f for f in required_files if not (utils_dir / f).exists()]
        if missing:
            print(f"❌ Missing files: {missing}")
            return False
        print("✓ All utils modules installed correctly!")
        return True

    def check_file_imports(self, filename):
        """Check if file has utils imports"""
        fpath = self.project_root / filename
        if not fpath.exists(): return False
        return 'from utils import' in fpath.read_text()

    def get_match_line_numbers(self, pattern, content):
        """Helper to find line numbers for a specific regex pattern"""
        line_numbers = []
        for match in re.finditer(pattern, content):
            # Calculate line number by counting newlines before the match start index
            line_no = content.count('\n', 0, match.start()) + 1
            line_numbers.append(line_no)
        return line_numbers

    def count_old_prints(self, filename):
        """Count old-style print statements and return their line numbers"""
        fpath = self.project_root / filename
        if not fpath.exists():
            return 0, []
        content = fpath.read_text()
        pattern = r'print\(f" >  \['
        line_numbers = self.get_match_line_numbers(pattern, content)
        return len(line_numbers), line_numbers

    def check_landfall_usage(self, filename):
        """Check if file uses find_landfall_time() or old pattern"""
        fpath = self.project_root / filename
        if not fpath.exists(): return False
        content = fpath.read_text()
        uses_new = 'find_landfall_time' in content
        uses_old = 'for _, row in track' in content and 'is_point_land' in content
        return uses_new and not uses_old

    def generate_checklist(self):
        """Generate integration checklist"""
        print("\n--- STEP 1: INTEGRATION CHECKLIST ---")
        for filename, file_code in self.files_to_integrate.items():
            fpath = self.project_root / filename
            if not fpath.exists():
                continue

            print(f"\n{filename} ({file_code}):")

            # Imports
            has_imports = self.check_file_imports(filename)
            print(f"  {'✓' if has_imports else '❌'} Utils imports added")

            # Prints with line numbers
            count, lines = self.count_old_prints(filename)
            line_str = f" (Lines: {', '.join(map(str, lines))})" if lines else ""
            print(f"  {'✓' if count == 0 else '⚠️ '} Old-style prints: {count} remaining{line_str}")

            # Landfall
            uses_new_landfall = self.check_landfall_usage(filename)
            print(f"  {'✓' if uses_new_landfall else '❌'} Uses find_landfall_time()")
            
            # Status
            all_done = has_imports and count == 0 and uses_new_landfall
            status = "✓ DONE" if all_done else "⚠️  IN PROGRESS" if (has_imports or count < 10) else "❌ NOT STARTED"
            print(f"  Status: {status}")

    def run_integration_diagnostics(self):
        """Run diagnostics on all files with specific line pointers"""
        print("\n--- STEP 2: DETAILED DIAGNOSTICS ---")
        total_issues = 0
        
        for filename in self.files_to_integrate.keys():
            fpath = self.project_root / filename
            if not fpath.exists(): continue
            
            content = fpath.read_text()
            
            # Patterns to find
            patterns = {
                "Old Prints": r'print\(f" >  \[',
                "Manual Spatial Buffers": r'lat_buffer = buffer_km / 111',
                "Manual Landfall Loops": r'for.*in.*iterrows.*is_point_land'
            }
            
            file_header_printed = False
            for label, pattern in patterns.items():
                lines = self.get_match_line_numbers(pattern, content)
                if lines:
                    if not file_header_printed:
                        print(f"\n[{filename}]")
                        file_header_printed = True
                    print(f"  ⚠️  {label}: Found at line(s) {', '.join(map(str, lines))}")
                    total_issues += len(lines)
        
        if total_issues == 0:
            print("\n✓ No issues found! Integration looks solid.")
        else:
            print(f"\nSummary: {total_issues} patterns remaining to be consolidated.")

    def main(self):
        """Run all checks"""
        print("="*60)
        print("MONSOON DEPRESSION ANALYSIS - UTILS INTEGRATION HELPER")
        print("="*60)
        
        if not self.check_utils_installed():
            print("\n❌ Fix utils installation first!")
            return
            
        self.generate_checklist()
        self.run_integration_diagnostics()

if __name__ == '__main__':
    helper = IntegrationHelper()
    helper.main()
