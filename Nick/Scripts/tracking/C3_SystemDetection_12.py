'''
Author: Alex Crawford
Date Created: 11 Jan 2016
Date Modified: 8 Dec 2017, 4 Jun 2019 (Python 3), 13 Jun 2019 (warning added)
Purpose: Convert a series of center tracks to system tracks. Warning: If a) you
wish to re-run this process on some of the data and b) you are using rg = 1
(allowing regeneration), you need to re-run from the reftime or accept that
some active storms at the re-start point will get truncated.

User inputs:
    Path Variables
    Bounding Box ID (bboxnum): 2-digit character string
    Time Variables: when to start, end, the time step of the data
    Regenesis Paramter: 0 or 1, depending on whether regenesis continues tracks
'''

'''********************
Import Modules
********************'''
# Import clock:
from time import perf_counter
# Start script stopwatch. The clock starts running when time is imported
start = perf_counter()

print("Loading modules.")
import pandas as pd
import CycloneModule_12_4 as md
import sys
import copy
import xarray as xr
import glob

'''*******************************************
Set up Environment
*******************************************'''
print("Setting up environment.")
subset = "" # use "" if performing on all cyclones

_, inpath, refpath = sys.argv[:]
# inpath = "/home/ubuntu/notebooks/wind-eu/preliminary/output/tracking12_4TestTracks/"+subset
outpath = inpath

'''*******************************************
Define Variables
*******************************************'''
print("Defining variables")
# Regenesis Paramater
rg = 1
# 0 = regenesis starts a new system track; 
# 1 = regenesis continues previous system track with new ptid

# Time Variables
# starttime = [1959,1,1,0,0,0] # Format: [Y,M,D,H,M,S]
# endtime = [2022,1,1,0,0,0] # stop BEFORE this time (exclusive)
# reftime = [1959,1,1,0,0,0]
monthstep = [0,1,0,0,0,0] # A Time step that increases by 1 month [Y,M,D,H,M,S]

# dateref = [1900,1,1,0,0,0] # [Y,M,D,H,M,S]

## NJL: automatic time variable detection
cyclone_files = glob.glob(inpath+'/CycloneTracks/*/*.pkl')
cyclone_months = [x.split('/')[-1].split('.')[0].split('tracks')[1] for x in cyclone_files]
starttime = pd.to_datetime([x+'01' for x in cyclone_months]).min()
endtime = pd.to_datetime([x+'01' for x in cyclone_months]).max()
endtime = pd.to_datetime('{}-{}-{}'.format(endtime.year + (1 if endtime.month+1 > 12 else 0),(endtime.month+1)%12 if endtime.month+1 > 12 else endtime.month+1,endtime.day))
timeref_ds = xr.open_dataset(glob.glob(refpath+'*.nc')[0],decode_times=False).time
dateref = pd.to_datetime([x for x in timeref_ds.units.split(' ') if '-' in x][0])
calendar = timeref_ds.calendar

try:
    starttime=pd.to_datetime(starttime)
    endtime=pd.to_datetime(endtime)
    dateref=pd.to_datetime(dateref)
except:
    pass

convert_times_to_list = lambda x: [int(y) for y in x.strftime('%Y-%m-%d-%H-%M-%S').split('-')]
starttime = convert_times_to_list(starttime)
endtime = convert_times_to_list(endtime)
dateref = convert_times_to_list(dateref)
reftime = copy.deepcopy(starttime)

## NJL: check if MOHC model (BOOOO)
if calendar == '360_day':
    daysBetween_dpy = 360
    daysBetween_lys = 0
    print('calendar is 360 day noleap')
elif calendar == '365_day' or calendar == 'noleap':
    daysBetween_dpy = 365
    daysBetween_lys = 0
    print('calendar is 365 day noleap')
else:
    daysBetween_dpy = 365
    daysBetween_lys = 1
    print('calendar is 365 day leap')

mons = ["01","02","03","04","05","06","07","08","09","10","11","12"]

'''*******************************************
Main Analysis
*******************************************'''
print("Main Analysis")

mt = starttime
while mt != endtime:
    # Extract date
    Y = str(mt[0])
    M = mons[mt[1]-1]
    print (" " + Y + " - " + M)
    
    # Load Cyclone Tracks
    ct = pd.read_pickle(inpath+"/CycloneTracks/"+Y+"/"+subset+"cyclonetracks"+Y+M+".pkl")
    
    # Create System Tracks
    # NJL: include new track logic if month is October
    if mt == reftime or mt[1]==10:
        cs, cs0 = md.cTrack2sTrack(ct,[],dateref,rg,dpy=daysBetween_dpy,lyb=daysBetween_lys)
        pd.to_pickle(cs,inpath+"/SystemTracks/"+Y+"/"+subset+"systemtracks"+Y+M+".pkl")
    
    else:
        # Extract date for previous month
        mt0 = md.timeAdd(mt,[-d for d in monthstep],dpy=daysBetween_dpy,lys=daysBetween_lys)
        Y0 = str(mt0[0])
        M0 = mons[mt0[1]-1]
        
        # Load previous month's system tracks
        cs0 = pd.read_pickle(inpath+"/SystemTracks/"+Y0+"/"+subset+"systemtracks"+Y0+M0+".pkl")
        
        # Create system tracks
        cs, cs0 = md.cTrack2sTrack(ct,cs0,dateref,rg,dpy=daysBetween_dpy,lyb=daysBetween_lys)
        pd.to_pickle(cs,inpath+"/SystemTracks/"+Y+"/"+subset+"systemtracks"+Y+M+".pkl")
        pd.to_pickle(cs0,inpath+"/SystemTracks/"+Y0+"/"+subset+"systemtracks"+Y0+M0+".pkl")
    
    # Increment Time Step
    mt = md.timeAdd(mt,monthstep,dpy=daysBetween_dpy,lys=daysBetween_lys)
    
    # NJL: if new month is April, skip to October
    if mt[1] == 4:
        mt[1] = 10

print('Elapsed time:',round(perf_counter()-start,2),'seconds')
