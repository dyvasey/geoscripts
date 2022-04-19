"""
Test of CAD modules
"""
import pandas as pd

import dz

#%% Load Data

data = pd.read_csv('test_data.csv',header=None)

sample = dz.DZSample(name='test',agedata=data)

sample.bestage=sample.agedata[0]

#%% Plot CAD

sample.cad()

#%% Cawood Test

sample.cawood_classify(plot=True)
