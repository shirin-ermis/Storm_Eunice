import os
import copy
import pandas as pd
import numpy as np
import CycloneModule_12_4 as md
import glob
import sys

'''*******************************************
Set up Environment
*******************************************'''
print("Setting up environment.")
subset = "" # use "" if performing on all cyclones

_, inpath = sys.argv[:]
# inpath = "/home/ubuntu/notebooks/wind-eu/preliminary/output/tracking12_4TestTracks/"+subset
outpath = inpath

# create directories required


systemtracks = glob.glob(inpath+"SystemTracks/*/*.pkl")

for track in systemtracks:
    
    fname = track.split('/')[-1].split('.')[0]
    fyear = int(fname[-6:-2])
    fmon = int(fname[-2:])
    fpath = '/'.join(track.split('/')[:-1])
    
    pkl = pd.read_pickle(track)
    concat = pd.concat([x.data.assign(sid=x.sid,year=fyear,mon=fmon) for x in pkl])
    
    concat.to_hdf(fpath+'/'+fname+'.h5','systemtracks')