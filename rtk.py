from pppssr import pppos
import numpy as np
from copy import deepcopy
from ephemeris import satposs


class rtkpos(pppos):
    """ Class for RTK (Real-Time Kinematic) positioning """

    def __init__(self, nav, pos0=np.zeros(3), logfile=None):
        """ Initialize RTK variables """

        # Call parent class with PPP-RTK settings (e.g. no tropo/iono corrections)
        super().__init__(nav=nav, pos0=pos0, logfile=logfile,
                         trop_opt=0, iono_opt=0, phw_opt=0)

        # Set error model and AR (Ambiguity Resolution) settings
        self.nav.eratio = np.ones(self.nav.nf) * 50  # Error ratio for signals
        self.nav.err = [0, 0.01, 0.005] / np.sqrt(2)  # Measurement error (in meters)
        self.nav.sig_p0 = 30.0  # Initial sigma value for positioning (in meters)
        self.nav.thresar = 2.0  # AR threshold for acceptance
        self.nav.armode = 1     # Enable AR

    def base_process(self, obs, obsb, rs, dts, svh):
        """ Process observations from base station in RTK """

        # Get base station satellite positions and velocities
        rsb, vsb, dtsb, svhb, _ = satposs(obsb, self.nav)

        # Calculate residuals for base station
        yr, er, elr = self.zdres(
            obsb, None, None, rsb, vsb, dtsb, self.nav.rb, 0)

        # Edit observations (base/rover)
        sat_ed_r = self.qcedit(obsb, rsb, dtsb, svhb, rr=self.nav.rb)
        sat_ed_u = self.qcedit(obs, rs, dts, svh)

        # Find common satellites between base and rover
        sat_ed = np.intersect1d(sat_ed_u, sat_ed_r, True)
        ir = np.intersect1d(obsb.sat, sat_ed, True, True)[1]
        iu = np.intersect1d(obs.sat, sat_ed, True, True)[1]
        ns = len(iu)

        # Prepare arrays for residuals and satellite directions
        y = np.zeros((ns*2, self.nav.nf*2))
        e = np.zeros((ns*2, 3))

        # Copy base station residuals and satellite direction info
        y[ns:, :] = yr[ir, :]
        e[ns:, :] = er[ir, :]

        # Copy and adjust rover observations by subtracting base station data
        obs_ = deepcopy(obs)
        obs_.sat = obs.sat[iu]
        obs_.L = obs.L[iu, :] - obsb.L[ir, :]
        obs_.P = obs.P[iu, :] - obsb.P[ir, :]

        return y, e, iu, obs_
