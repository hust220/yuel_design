import sys
from datetime import datetime


def section(title: str):
    """Print a section header"""
    print(f"\n{'='*60}")
    print(f" {title}")
    print(f"{'='*60}")


def info(message: str):
    """Print an info message"""
    timestamp = datetime.now().strftime("%H:%M:%S")
    print(f"[{timestamp}] INFO: {message}")


def success(message: str):
    """Print a success message"""
    timestamp = datetime.now().strftime("%H:%M:%S")
    print(f"[{timestamp}] ✓ SUCCESS: {message}")


def warn(message: str):
    """Print a warning message"""
    timestamp = datetime.now().strftime("%H:%M:%S")
    print(f"[{timestamp}] ⚠ WARNING: {message}")


def error(message: str):
    """Print an error message"""
    timestamp = datetime.now().strftime("%H:%M:%S")
    print(f"[{timestamp}] ✗ ERROR: {message}")


class spinner:
    """Context manager for showing a spinner"""
    def __init__(self, text: str):
        self.text = text
        self.spinning = False
        
    def __enter__(self):
        print(f"[{datetime.now().strftime('%H:%M:%S')}] {self.text}...")
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        pass
