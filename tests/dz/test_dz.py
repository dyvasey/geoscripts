"""
Tests for dz.dz module
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cartopy.crs as ccrs

import pytest

from geoscripts.dz import dz

# Create synthetic data
ages = pd.Series([50,150,145,250,252,255,300,355,1542,2555,3000])

# Create basic dz sample
smp = dz.DZSample(name='test',agedata=ages)
smp.bestage = smp.agedata
smp.latlon = (42,-118)

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

def test_add_pie():
    """Test Add Pie function"""

    # Create figure
    fig,ax = plt.subplots(1)

    # Set spans
    spans = [(0,75),(100,301),(500,700)]

    # Plot KDE
    smp.kde(ax=ax)

    # Add pie chart
    smp.add_pie(spans=spans,ax=ax)

    # Test that the axes contains a child axes
    assert len(ax.child_axes)==1

    # Test the same using the option in the KDE function with colors
    fig2,ax2 = plt.subplots(1)
    cmap = plt.get_cmap('magma')
    colors = cmap.colors
    smp.kde(ax=ax2,add_pie=True,spans=spans,span_colors=colors)

    # Test that the axes contains a child axes
    assert len(ax2.child_axes)==1

def test_add_pie_map():
    """Test add_pie_map function"""

    # Create figure with Cartopy
    fig,ax = plt.subplots(1,subplot_kw={'projection':ccrs.PlateCarree()})

    # Set spans
    spans = [(0,75),(100,301),(500,700)]

    # Plot cartopy map of sample location
    ax = smp.map_location(ax=ax)

    # Add pie chart to map
    ax_pie = smp.add_pie_map(spans=spans,ax=ax)

    # Test that the axes contains a child axes
    assert len(ax.child_axes)==1

def test_add_spans():
    """Test Add Spans function"""

    # Create figure
    fig,ax = plt.subplots(1)

    # Set spans
    spans = [(0,75),(100,301)]

    # Plot KDE
    smp.kde(ax=ax)

    # Test adding spans
    smp.add_spans(spans=spans,ax=ax)
    
    assert len(fig.get_axes())==1

    # Test the same using the option in the KDE function with colors
    fig2,ax2 = plt.subplots(1)
    cmap = plt.get_cmap('magma')
    colors = cmap.colors
    smp.kde(ax=ax2,spans=spans,span_colors=colors)

    assert len(fig2.get_axes())==1



    


