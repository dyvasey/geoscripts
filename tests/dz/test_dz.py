"""
Tests for dz.dz module
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
from matplotlib.collections import PolyCollection

import pytest

from geoscripts.dz import dz

# Create synthetic data
ages = [50,150,145,250,252,255,300,355,1542,2555,3000]
uth = [2,3,15,4,25,5,10,3,2,12,4]
agedata = pd.DataFrame({'Best Age':ages,'U/Th':uth})

# Create basic dz sample
smp = dz.DZSample(name='test',agedata=agedata)
smp.bestage = smp.agedata['Best Age']
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

def test_classify_uth():
    """Test classify_uth function"""

    smp.classify_uth('U/Th')

    # Test that only values above the cutoff (10) are classified as metamorphic
    expected = ['Igneous','Igneous','Metamorphic','Igneous','Metamorphic',
                'Igneous','Igneous','Igneous','Igneous','Metamorphic','Igneous']
    assert list(smp.agedata['Origin']) == expected

def test_kde():
    """Test KDE function"""

    # Create figure
    fig,ax = plt.subplots(1)

    # Plot basic KDE
    smp.kde(ax=ax,rug=False)

    # Test that one filled KDE plotted
    fills = [c for c in ax.collections if isinstance(c,PolyCollection)]
    assert len(fills)==1

    # Test KDE split into groups with hue
    # Classify here so test does not depend on test order
    smp.classify_uth('U/Th')

    fig2,ax2 = plt.subplots(1)
    smp.kde(ax=ax2,hue='Origin',rug=False)

    # Test that one filled KDE plotted for each group
    fills = [c for c in ax2.collections if isinstance(c,PolyCollection)]
    assert len(fills)==2

    # Test that legend shows both groups
    labels = [t.get_text() for t in ax2.get_legend().get_texts()]
    assert sorted(labels)==['Igneous','Metamorphic']

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



    


