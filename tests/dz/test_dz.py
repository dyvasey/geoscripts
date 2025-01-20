"""
Tests for dz.dz module
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import pytest

from geoscripts.dz import dz

# Create synthetic data
ages = pd.Series([50,150,145,250,252,255,300,355])

# Create basic dz sample
smp = dz.DZSample(name='test',agedata=ages)
smp.bestage = smp.agedata

def test_pie():
    """Test Pie function"""

    # Create figure
    fig,ax = plt.subplots(1)

    # Set spans
    spans = [(0,75),(100,301)]

    # Attempt to plot
    smp.pie(spans,ax=ax)

    # Test that the figure contains 1 axes
    assert len(fig.get_axes())==1

    


