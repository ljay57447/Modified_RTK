import matplotlib.pyplot as plt
import numpy as np
import sys

import pppssr
import rinex as rn
import gnss as gn
from gnss import rSigRnx, time2str
from rtk import rtkpos

# Reference data
xyz_ref = [-3962108.7007, 3381309.5532, 3668678.6648]
pos_ref = gn.ecef2pos(xyz_ref)

# Signal configuration
def generate_signals(prefixes, suffixes):
    return [rSigRnx(f"{p}{s}") for p in prefixes for s in suffixes]

sigs = generate_signals(["G", "E", "J", "G", "E", "J"], ["C1C", "C2W", "C5Q", "L1C", "L2L"])
sigsb = generate_signals(["G", "E", "J"], ["C1C", "C2W", "C5X", "L1X", "L2X"])

# File paths
bdir = 'data/'
navfile, obsfile = bdir+'SEPT238A.23P', bdir+'SEPT238A.23O'
basefile = bdir+'3034238A.23O'

# Rover setup
dec = rn.rnxdec()
dec.setSignals(sigs)
nav = gn.Nav()
dec.decode_nav(navfile, nav)
dec.decode_obsh(obsfile)
dec.autoSubstituteSignals()

# Base station setup
nav.rb = [-3959400.6443, 3385704.4948, 3667523.1275]  # GSI 3034 fujisawa
decb = rn.rnxdec()
decb.setSignals(sigsb)
decb.decode_obsh(basefile)
decb.autoSubstituteSignals()

# RTK setup
rtk = rtkpos(nav, dec.pos, 'test_rtk.log')
rr = dec.pos

# Process data for 3 minutes
nep = 180
t, enu, smode = np.zeros(nep), np.zeros((nep, 3)), np.zeros(nep, dtype=int)

for ne in range(nep):
    obs, obsb = rn.sync_obs(dec, decb)
    if ne == 0:
        t0 = nav.t = obs.t
    pppssr.process(obs, obsb=obsb)
    t[ne] = gn.timediff(nav.t, t0)
    sol = nav.xa[0:3] if nav.smode == 4 else nav.x[0:3]
    enu[ne] = gn.ecef2enu(pos_ref, sol - xyz_ref)
    smode[ne] = nav.smode

    # Output logging
    sys.stdout.write(f'\r {time2str(obs.t)} ENU {enu[ne, 0]:7.4f} {enu[ne, 1]:7.4f} {enu[ne, 2]:7.4f}, '
                     f'2D {np.sqrt(enu[ne, 0]**2 + enu[ne, 1]**2):6.4f}, mode {smode[ne]:1d}')

# Close files
dec.fobs.close()
decb.fobs.close()

# Plotting function
def plt_enu(t, enu, dmax=0.1):
    plt.figure(figsize=(5, 4))
    plt.plot(t, enu)
    plt.ylabel('Position error [m]')
    plt.xlabel('Time [s]')
    plt.legend(['East', 'North', 'Up'])
    plt.grid()
    plt.axis([0, nep, -dmax, dmax])
    plt.show()

plt_enu(t, enu)
