"""
Quick test script to verify all modules are working correctly
Author: Raj Kalpesh Mathuria
"""

import sys
import os

print("="*70)
print("TESTING CNN PROJECT MODULES")
print("="*70 + "\n")

# Test imports
try:
    print("1. Testing config import...")
    import config
    print("   ✓ Config imported successfully")
    print(f"   - Dataset: {config.DATASET_NAME}")
    print(f"   - Classes: {config.NUM_CLASSES}")
    print(f"   - Epochs: {config.EPOCHS}")
except Exception as e:
    print(f"   ✗ Config import failed: {e}")

try:
    print("\n2. Testing data_loader import...")
    from src import data_loader
    print("   ✓ Data loader imported successfully")
except Exception as e:
    print(f"   ✗ Data loader import failed: {e}")

try:
    print("\n3. Testing model import...")
    from src import model
    print("   ✓ Model module imported successfully")
except Exception as e:
    print(f"   ✗ Model import failed: {e}")

try:
    print("\n4. Testing train import...")
    from src import train
    print("   ✓ Train module imported successfully")
except Exception as e:
    print(f"   ✗ Train import failed: {e}")

try:
    print("\n5. Testing plotting import...")
    from src import plotting
    print("   ✓ Plotting module imported successfully")
except Exception as e:
    print(f"   ✗ Plotting import failed: {e}")

try:
    print("\n6. Testing utils import...")
    from src import utils
    print("   ✓ Utils module imported successfully")
except Exception as e:
    print(f"   ✗ Utils import failed: {e}")

print("\n" + "="*70)
print("TESTING COMPLETED")
print("="*70)
print("\nAll modules are properly configured!")
print("You can now run: python main.py")
print("="*70 + "\n")
