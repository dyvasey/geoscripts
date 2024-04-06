"""
Tests for gchemplots module
"""
import matplotlib.pyplot as plt
import matplotlib.text
import numpy as np

import pytest

from mpltern.ternary import TernaryAxes
from geoscripts import gchemplots as gcp

def test_afm():
    """ Test AFM function """
    # Create Matplotlib figure/axes for ternary diagram
    fig = plt.figure()
    ax = fig.add_subplot(111,projection='ternary')

    # Create pseudorandom values for major elements
    rng = np.random.default_rng(seed=2042)
    K2O,Na2O,FeOt,MgO = [rng.uniform(low=0,high=50,size=10) for x in range(4)]

    gcp.afm(Na2O=Na2O,K2O=K2O,FeOt=FeOt,MgO=MgO,ax=ax)

    # Test that the figure contains 1 axes
    assert len(fig.get_axes())==1

    # Test that figure is a ternary axes
    assert isinstance(ax,TernaryAxes)

    # Test that the boundary line plotted
    assert len(ax.get_lines())==1

    # Test that the annotations plotted
    plot_text = [child.get_text() for child in ax.get_children() if isinstance(child,matplotlib.text.Text)]
    assert plot_text[0] == 'Tholeiitic'
    assert plot_text[1] == 'Calc-Alkaline'

    # Test that the scatter plot worked
    assert ax.collections

    # Test that the lables plotted
    assert ax.get_tlabel()=='F'
    assert ax.get_llabel()=='A'
    assert ax.get_rlabel()=='M'

    # Test that the ticks are removed
    assert ax.taxis.get_major_ticks()==[]
    assert ax.laxis.get_major_ticks()==[]
    assert ax.raxis.get_major_ticks()==[]

    # Test that the boundary line will not replot on a second call
    gcp.afm(Na2O=Na2O,K2O=K2O,FeOt=FeOt,MgO=MgO,ax=ax)
    assert len(ax.get_lines())==1

    # Test that KDE will plot
    gcp.afm(Na2O=Na2O,K2O=K2O,FeOt=FeOt,MgO=MgO,ax=ax,density=True,scatter=False)

def test_afm_line():
    """ Test AFM line function """
    A,F,M = gcp.afm_line()
    
    assert np.all(A>=0)
    assert np.all(F>=0)
    assert np.all(M>=0)
    assert np.all((A+F+M)==100)

def test_mantle_array():
    """ Test mantle array function"""
    # Create Matplotlib figure/axes
    fig,ax = plt.subplots(1)

    # Create pseudorandom values for trace elements
    rng = np.random.default_rng(seed=54132)
    Th,Nb,Yb = [rng.uniform(low=0,high=5,size=10) for x in range(3)]

    gcp.mantle_array(Th,Nb,Yb,ax=ax)

    # Test that the arc line plotted
    assert len(ax.get_lines())==1

    # Test that the mantle polygon plotted
    assert len(ax.patches)==1

    # Test that the scatter plot worked
    assert ax.collections

    # Test whether the annotations plotted
    plot_text = [child.get_text() for child in ax.get_children() if isinstance(child,matplotlib.text.Text)]
    assert plot_text[0] == 'Mantle Array'
    assert plot_text[1] == 'Arc Array'

    # Test that the lables plotted
    assert ax.get_xlabel()=='Nb/Yb'
    assert ax.get_ylabel()=='Th/Yb'

    # Test that the boundary line will not replot on a second call
    gcp.mantle_array(Th,Nb,Yb,ax=ax)
    assert len(ax.get_lines())==1

    # Test that KDE will plot
    gcp.mantle_array(Th,Nb,Yb,ax=ax,density=True,scatter=False)