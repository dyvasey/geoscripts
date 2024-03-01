"""
Tests for gchemplots module
"""
import matplotlib.pyplot as plt
import numpy as np

import pytest

from mpltern.ternary import TernaryAxes
from geoscripts import gchemplots as gcp

def test_afm():
    """ Test AFM function """
    # Create Matplotlib figure/axes for ternary diagram
    fig = plt.figure()
    ax = fig.add_subplot(111,projection='ternary')

    K2O,Na2O,FeOt,MgO = (1,0.5,3,2)

    gcp.afm(Na2O=Na2O,K2O=K2O,FeOt=FeOt,MgO=MgO,ax=ax)

    # Test that the figure contains 1 axes
    assert len(fig.get_axes())==1

    # Test that figure is a ternary axes
    assert isinstance(ax,TernaryAxes)

    # Test that the boundary line plotted
    assert len(ax.get_lines())==1

    # Test that the scatter plot worked
    assert ax.collections

def test_afm_line():
    """ Test AFM line function """
    A,F,M = gcp.afm_line()
    
    assert np.all(A>=0)
    assert np.all(F>=0)
    assert np.all(M>=0)
    assert np.all((A+F+M)==100)