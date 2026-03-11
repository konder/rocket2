#!/usr/bin/env python
"""
Wrapper for collect_grounding_data.py that auto-answers MinecraftStudio prompts
"""

import sys
import os
import threading
import time

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Monkey-patch input() to auto-answer 'y' for MinecraftStudio prompts
_original_input = __builtins__.input

def patched_input(prompt=""):
    if "huggingface" in prompt.lower() and "download" in prompt.lower():
        print(prompt + "y")  # Print the prompt and answer
        return "y"
    return _original_input(prompt)

__builtins__.input = patched_input

# Now import and run the actual script
from train import collect_grounding_data

if __name__ == "__main__":
    # Run the main script
    collect_grounding_data.main()

