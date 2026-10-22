"""
Tests for dz.dz module
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
from matplotlib.collections import LineCollection, PolyCollection
from scipy.stats import norm

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

    # Test fixed bandwidth gives the same kernel SD for groups with
    # different spreads. Compare plotted curves to a Gaussian KDE with
    # SD = h in log10 space, scaled by each group's share of grains
    # (seaborn's default common_norm=True).
    h = 0.05
    fig3,ax3 = plt.subplots(1)
    smp.kde(ax=ax3,hue='Origin',rug=False,bandwidth=h,bw_adjust=0.2)

    log_ages = np.log10(smp.bestage)
    origin = smp.agedata.loc[smp.bestage.index,'Origin']
    n_total = len(log_ages)
    expected_peaks = []
    for group in ['Igneous','Metamorphic']:
        x = log_ages[origin==group].values
        grid = np.linspace(x.min()-0.5,x.max()+0.5,2000)
        dens = norm.pdf(grid[:,None],x[None,:],h).mean(axis=1)
        expected_peaks.append(dens.max()*len(x)/n_total)

    fills = [c for c in ax3.collections if isinstance(c,PolyCollection)]
    plotted_peaks = [c.get_paths()[0].vertices[:,1].max() for c in fills]
    assert np.allclose(sorted(plotted_peaks),sorted(expected_peaks),rtol=0.01)

    # Test rug ticks are clipped to the axis x range but extend below it
    fig4,ax4 = plt.subplots(1)
    smp.kde(ax=ax4,hue='Origin')
    rugs = [c for c in ax4.collections if isinstance(c,LineCollection)]
    assert len(rugs)>0
    for c in rugs:
        assert c.get_clip_on()
        box = c.get_clip_box()
        assert np.isclose(box.x0,ax4.bbox.x0) and np.isclose(box.x1,ax4.bbox.x1)
        assert box.y0<ax4.bbox.y0

    # Test no minor ticks (log axis otherwise labels them, e.g. 6x10^2)
    fig5,ax5 = plt.subplots(1)
    smp.kde(ax=ax5,log_scale=True)
    fig5.canvas.draw()
    assert len(ax5.xaxis.get_minorticklocs())==0

    plt.close('all')

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



    


