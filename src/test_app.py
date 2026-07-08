#!/usr/bin/env python3
"""Test script to verify the FDS Visualizer app initializes correctly."""

import sys
import os
from PyQt5 import QtCore
from PyQt5.QtWidgets import QApplication
from load_data import load_all_data
from main import Main

# Dummy splash screen for testing
class DummySplash:
    def setMaximum(self, val): pass
    def setValue(self, val): print(f'  Loading: {val}/{24}', end='\r')
    def show(self): print('[Splash] Showing loading screen...')
    def close(self): print('[Splash] Loading complete!')

def main():
    print("=" * 60)
    print("FDS Visualizer - Application Test")
    print("=" * 60)
    
    # Create app
    print('\n[Step 1] Creating QApplication...')
    app = QApplication(sys.argv)
    print('  ✓ QApplication created')
    
    # Load data
    print('\n[Step 2] Loading FDS simulation data...')
    try:
        splash = DummySplash()
        splash.show()
        data, data_matrix = load_all_data(2, 2, 3, 2, splash, app)
        splash.close()
        print(f'\n  ✓ Data loaded successfully')
        print(f'    - Data shape: {data.shape}')
        print(f'    - Data matrix shape: {data_matrix.shape}')
        print(f'    - Memory usage: ~{data.nbytes / 1e9:.2f} GB')
    except Exception as e:
        print(f'\n  ✗ FAILED to load data: {e}')
        import traceback
        traceback.print_exc()
        return False
    
    # Create GUI
    print('\n[Step 3] Creating GUI window...')
    try:
        main_window = Main(data, data_matrix)
        print('  ✓ GUI window created successfully')
        print(f'    - Window size: {main_window.width()}x{main_window.height()}')
    except Exception as e:
        print(f'  ✗ FAILED to create GUI: {e}')
        import traceback
        traceback.print_exc()
        return False
    
    print("\n" + "=" * 60)
    print("✓ All tests passed! App is ready to run.")
    print("=" * 60)
    return True

if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)
