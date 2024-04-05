"""
Tests for gchemcalc module
"""
import numpy as np
import pandas as pd

import pytest

from geoscripts import gchemcalc as gcc

def test_classifyAlkaline():
    """Test Classify Alkaline Function"""

    # Create an alkaline and subalkaline data point
    df = pd.DataFrame(columns=['SiO2','Na2O','K2O'])
    df['SiO2'] = [50,55]
    df['Na2O'] = [4,1]
    df['K2O'] = [3,1]

    subalkaline = gcc.classifyAlkaline(df)

    # Test that return is the correct length
    assert len(subalkaline)==2

    # Test that the classification is correct
    assert subalkaline[0]==False
    assert subalkaline[1]==True
    
