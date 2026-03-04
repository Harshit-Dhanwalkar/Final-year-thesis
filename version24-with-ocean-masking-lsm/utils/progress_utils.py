import sys
import time

class ProgressSpinner:
    def __init__(self, total_files, description=""):
        self.total_files = total_files
        self.description = description
        self.current = 0
        self.start_time = time.time()
        self.spinner_frames = ["◜", "◝", "◞", "◟"]
        # self.spinner_frames = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]
        self.frame_index = 0

    def update(self, current=None, increment=1):
        """Update progress. Either set current value or increment by 1"""
        if current is not None:
            self.current = current
        else:
            self.current += increment

        # Calculate progress
        if self.total_files > 0:
            percent = (self.current / self.total_files) * 100
            bar_length = 20
            filled = int(bar_length * self.current / self.total_files)
            bar = "█" * filled + " " * (bar_length - filled)
            # Get spinner frame
            spinner = self.spinner_frames[self.frame_index % len(self.spinner_frames)]
            self.frame_index += 1
            # Build progress message
            msg = f"\r{spinner} {self.description} [{bar}] {self.current}/{self.total_files} ({percent:.1f}%)"
            # Write to stdout without newline
            sys.stdout.write(msg)
            sys.stdout.flush()

    def finish(self, final_message=""):
        elapsed = time.time() - self.start_time
        # Clear the line completely (overwrite with spaces)
        sys.stdout.write("\r" + " " * 100 + "\r")
        sys.stdout.flush()
        return elapsed, final_message


class ProgressBar:
    def __init__(self, total_files, description="Processing"):
        self.total_files = total_files
        self.description = description
        self.current = 0

    def update(self, current=None, increment=1):
        """Update progress"""
        if current is not None:
            self.current = current
        else:
            self.current += increment

        if self.total_files > 0:
            percent = (self.current / self.total_files) * 100
            bar_length = 30
            filled = int(bar_length * self.current / self.total_files)
            bar = "█" * filled + " " * (bar_length - filled)
            msg = f"\r{self.description}: [{bar}] {percent:.1f}% ({self.current}/{self.total_files})"
            sys.stdout.write(msg)
            sys.stdout.flush()
  
    def finish(self):
        """Clear the progress line"""
        sys.stdout.write("\r" + " " * 120 + "\r")
        sys.stdout.flush()
