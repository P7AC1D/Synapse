"""Progress indicator utilities for long-running operations with thread-safe implementation."""

import sys
import time
import threading
from typing import Optional

class ProgressIndicator:
    """Thread-safe progress indicator implementation."""
    
    def __init__(self):
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()
    
    def _show_progress(self, message: str) -> None:
        """Internal method to display progress animation."""
        chars = "|/-\\"
        i = 0
        try:
            while not self._stop_event.is_set():
                with self._lock:
                    char = chars[i % len(chars)]
                    sys.stdout.write(f'\r{message}... {char}')
                    sys.stdout.flush()
                time.sleep(0.1)
                i += 1
        except Exception:
            pass  # Suppress any display errors during shutdown
        finally:
            # Ensure we clear the progress line
            with self._lock:
                sys.stdout.write('\r' + ' ' * 50 + '\r')
                sys.stdout.flush()

    def start(self, message: str = "Running") -> None:
        """Start the progress indicator in a new thread."""
        if self._thread is not None and self._thread.is_alive():
            return  # Don't start if already running
        
        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._show_progress,
            args=(message,),
            daemon=True
        )
        self._thread.start()

    def stop(self) -> None:
        """Stop the progress indicator and cleanup."""
        if self._thread is None:
            return
            
        self._stop_event.set()
        
        if self._thread.is_alive():
            try:
                self._thread.join(timeout=0.5)  # Wait for thread to finish
            except Exception:
                pass  # Ignore any join errors
        
        self._thread = None
        
        # Final cleanup of display
        with self._lock:
            sys.stdout.write('\r' + ' ' * 50 + '\r')
            sys.stdout.flush()

# Global progress indicator instance
_progress = ProgressIndicator()

def show_progress_continuous(message: str = "Running") -> None:
    """Start a continuous progress indicator that runs until stopped."""
    _progress.start(message)

def stop_progress_indicator() -> None:
    """Stop the progress indicator and cleanup."""
    _progress.stop()
