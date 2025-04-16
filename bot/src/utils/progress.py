"""Progress indicator utilities for long-running operations."""

import sys
import time
import threading

# Global flag to control the progress indicator
stop_progress = False

def show_progress_continuous(message="Running"):
    """Continuous progress indicator that runs until stopped."""
    global stop_progress
    stop_progress = False
    chars = "|/-\\"
    i = 0
    while not stop_progress:
        char = chars[i % len(chars)]
        sys.stdout.write(f'\r{message}... {char}')
        sys.stdout.flush()
        time.sleep(0.1)
        i += 1

def stop_progress_indicator():
    """Stop the progress indicator thread."""
    global stop_progress
    stop_progress = True
    time.sleep(0.2)  # Give thread time to terminate
    sys.stdout.write('\r' + ' ' * 50 + '\r')  # Clear the progress line
    sys.stdout.flush()
