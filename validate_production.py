#!/usr/bin/env python3
# Production startup validation script

import sys
import os

def main():
    try:
        # Test imports
        import streamlit
        import pandas as pd
        import numpy as np
        import xgboost
        import tensorflow as tf
        import plotly
        print('All core dependencies imported successfully')
        return 0
    except Exception as e:
        print(f'Error importing dependencies: {e}', file=sys.stderr)
        return 1

if __name__ == '__main__':
    sys.exit(main())
