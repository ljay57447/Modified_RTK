"""
module for standard PPP positioning
"""

import numpy as np

from ephemeris import satposs
from gnss import sat2id, sat2prn, rSigRnx, uTYP, uGNSS, rCST
from gnss import uTropoModel, ecef2pos, tropmodel, geodist, satazel
from gnss import time2str, timediff, tropmapf, uIonoModel
from cssrlib import sCSSRTYPE as sc
from mlambda import mlambda

# format definition for logging
fmt_ztd = "{}         ztd      ({:3d},{:3d}) {:10.3f} {:10.3f} {:10.3f}\n"
fmt_ion = "{} {}-{} ion {} ({:3d},{:3d}) {:10.3f} {:10.3f} {:10.3f} " + \
    "{:10.3f} {:10.3f}\n"
fmt_res = "{} {}-{} res {} ({:3d}) {:10.3f} sig_i {:10.3f} sig_j {:10.3f}\n"
fmt_amb = "{} {}-{} amb {} ({:3d},{:3d}) {:10.3f} {:10.3f} {:10.3f} " + \
    "{:10.3f} {:10.3f} {:10.3f}\n"


class pppos():
    """ class for PPP processing """

    nav = None
    VAR_HOLDAMB = 0.001

    def __init__(self, nav, pos0=np.zeros(3),
                 logfile=None, trop_opt=1, iono_opt=1, phw_opt=1):
        """ initialize variables for PPP """

        self.nav = nav

        # Number of frequencies (actually signals!)
        #
        self.nav.ephopt = 2  # SSR-APC

        # Select tropospheric model
        #
        self.nav.trpModel = uTropoModel.SAAST

        # Select iono model
        #
        self.nav.ionoModel = uIonoModel.KLOBUCHAR

        # 0: use trop-model, 1: estimate, 2: use cssr correction
        self.nav.trop_opt = trop_opt

        # 0: use iono-model, 1: estimate, 2: use cssr correction
        self.nav.iono_opt = iono_opt

        # 0: none, 1: full model, 2: local/regional model
        self.nav.phw_opt = phw_opt

        # carrier smoothing
        self.nav.csmooth = False

        # Position (+ optional velocity), zenith tropo delay and
        # slant ionospheric delay states
        #
        self.nav.ntrop = (1 if self.nav.trop_opt == 1 else 0)
        self.nav.niono = (uGNSS.MAXSAT if self.nav.iono_opt == 1 else 0)

        self.nav.na = (3 if self.nav.pmode == 0 else 6)
        self.nav.nq = (3 if self.nav.pmode == 0 else 6)

        self.nav.na += self.nav.ntrop + self.nav.niono
        self.nav.nq += self.nav.ntrop + self.nav.niono

        # State vector dimensions (including slant iono delay and ambiguities)
        #
        self.nav.nx = self.nav.na+uGNSS.MAXSAT*self.nav.nf

        self.nav.x = np.zeros(self.nav.nx)
        self.nav.P = np.zeros((self.nav.nx, self.nav.nx))

        self.nav.xa = np.zeros(self.nav.na)
        self.nav.Pa = np.zeros((self.nav.na, self.nav.na))

        self.nav.phw = np.zeros(uGNSS.MAXSAT)
        self.nav.el = np.zeros(uGNSS.MAXSAT)

        # Parameters for PPP
        #
        # Observation noise parameters
        #
        self.nav.eratio = np.ones(self.nav.nf)*100  # [-] factor
        self.nav.err = [0, 0.000, 0.003]       # [m] sigma

        # Initial sigma for state covariance
        #
        self.nav.sig_p0 = 100.0   # [m]
        self.nav.sig_v0 = 1.0     # [m/s]
        self.nav.sig_ztd0 = 0.1  # [m]
        self.nav.sig_ion0 = 10.0  # [m]
        self.nav.sig_n0 = 30.0    # [cyc]

        # Process noise sigma
        #
        if self.nav.pmode == 0:
            self.nav.sig_qp = 100.0/np.sqrt(1)     # [m/sqrt(s)]
            self.nav.sig_qv = None
        else:
            self.nav.sig_qp = 0.01/np.sqrt(1)      # [m/sqrt(s)]
            self.nav.sig_qv = 1.0/np.sqrt(1)       # [m/s/sqrt(s)]
        self.nav.sig_qztd = 0.05/np.sqrt(3600)     # [m/sqrt(s)]
        self.nav.sig_qion = 10.0/np.sqrt(1)        # [m/s/sqrt(s)]

        # Processing options
        #
        self.nav.thresar = 3.0  # AR acceptance threshold
        # 0:float-ppp,1:continuous,2:instantaneous,3:fix-and-hold
        self.nav.armode = 0
        self.nav.elmaskar = np.deg2rad(20.0)  # elevation mask for AR
        self.nav.elmin = np.deg2rad(10.0)

        # Initial state vector
        #
        self.nav.x[0:3] = pos0
        if self.nav.pmode >= 1:  # kinematic
            self.nav.x[3:6] = 0.0  # velocity

        # Diagonal elements of covariance matrix
        #
        dP = np.diag(self.nav.P)
        dP.flags['WRITEABLE'] = True

        dP[0:3] = self.nav.sig_p0**2
        # Velocity
        if self.nav.pmode >= 1:  # kinematic
            dP[3:6] = self.nav.sig_v0**2

        # Tropo delay
        if self.nav.trop_opt == 1:  # trop is estimated
            if self.nav.pmode >= 1:  # kinematic
                dP[6] = self.nav.sig_ztd0**2
            else:
                dP[3] = self.nav.sig_ztd0**2

        # Process noise
        #
        self.nav.q = np.zeros(self.nav.nq)
        self.nav.q[0:3] = self.nav.sig_qp**2

        # Velocity
        if self.nav.pmode >= 1:  # kinematic
            self.nav.q[3:6] = self.nav.sig_qv**2

        if self.nav.trop_opt == 1:  # trop is estimated
            # Tropo delay
            if self.nav.pmode >= 1:  # kinematic
                self.nav.q[6] = self.nav.sig_qztd**2
            else:
                self.nav.q[3] = self.nav.sig_qztd**2

        if self.nav.iono_opt == 1:  # iono is estimated
            # Iono delay
            if self.nav.pmode >= 1:  # kinematic
                self.nav.q[7:7+uGNSS.MAXSAT] = self.nav.sig_qion**2
            else:
                self.nav.q[4:4+uGNSS.MAXSAT] = self.nav.sig_qion**2

        # Logging level
        #
        self.monlevel = 0
        self.nav.fout = None
        if logfile is None:
            self.nav.monlevel = 0
        else:
            self.nav.fout = open(logfile, 'w')

    def valpos(self, v, R, thres=4.0):
        """ post-fit residual test """
        nv = len(v)
        fact = thres**2
        for i in range(nv):
            if v[i]**2 <= fact*R[i, i]:
                continue
            if self.nav.monlevel > 1:
                txt = "{:3d} is large: {:8.4f} ({:8.4f})".format(
                    i, v[i], R[i, i])
                if self.nav.fout is None:
                    print(txt)
                else:
                    self.nav.fout.write(txt+"\n")
        return True

    def initx(self, x0, v0, i):
        """ initialize x and P for index i """
        self.nav.x[i] = x0
        for j in range(self.nav.nx):
            self.nav.P[j, i] = self.nav.P[i, j] = v0 if i == j else 0

    def IB(self, s, f, na=3):
        """ return index of phase ambguity """
        idx = na+uGNSS.MAXSAT*f+s-1
        return idx

    def II(self, s, na):
        """ return index of slant ionospheric delay estimate """
        return na-uGNSS.MAXSAT+s-1

    def IT(self, na):
        """ return index of zenith tropospheric delay estimate """
        return na-uGNSS.MAXSAT-1

    def varerr(self, nav, el, f):
        """ variation of measurement """
        s_el = max(np.sin(el), 0.1*rCST.D2R)
        fact = nav.eratio[f-nav.nf] if f >= nav.nf else 1
        a = fact*nav.err[1]
        b = fact*nav.err[2]
        return (a**2+(b/s_el)**2)

    def sysidx(self, satlist, sys_ref):
        """ return index of satellites with sys=sys_ref """
        idx = []
        for k, sat in enumerate(satlist):
            sys, _ = sat2prn(sat)
            if sys == sys_ref:
                idx.append(k)
        return idx

    def udstate(self, obs):
        """Time propagation of states and initialization."""

        # Time difference between current and previous epoch
        tt = timediff(obs.t, self.nav.t)

        ns = len(obs.sat)  # Number of satellites
        sat = obs.sat
        sys = [sat2prn(sat_i)[0] for sat_i in obs.sat]  # Satellite systems

        # Update position, velocity, and state transition matrix
        nx = self.nav.nx
        Phi = np.eye(nx)
        if self.nav.pmode > 0:
            self.nav.x[0:3] += self.nav.x[3:6] * tt  # Position propagation
            Phi[0:3, 3:6] = np.eye(3) * tt  # State transition for position
        self.nav.P[0:nx, 0:nx] = Phi @ self.nav.P[0:nx, 0:nx] @ Phi.T  # Update covariance

        # Process noise for position and velocity states
        dP = np.diag(self.nav.P)
        dP.flags['WRITEABLE'] = True
        dP[0:self.nav.nq] += self.nav.q[0:self.nav.nq] * tt

        # Update Kalman filter states for each frequency
        for f in range(self.nav.nf):
            for i in range(uGNSS.MAXSAT):
                sat_ = i + 1
                sys_i, _ = sat2prn(sat_)
                self.nav.outc[i, f] += 1  # Outage counter

                # Check if we should reset phase ambiguity or ionosphere
                reset = (self.nav.outc[i, f] > self.nav.maxout or np.any(self.nav.edt[i, :] > 0))
                if sys_i not in obs.sig.keys():
                    continue

                # Reset phase ambiguity
                j = self.IB(sat_, f, self.nav.na)
                if reset and self.nav.x[j] != 0.0:
                    self.initx(0.0, 0.0, j)
                    self.nav.outc[i, f] = 0
                    if self.nav.monlevel > 0:
                        self.nav.fout.write(f"{time2str(obs.t)}  {sat2id(sat_)} - reset ambiguity\n")

                # Reset ionospheric delay
                if self.nav.niono > 0:
                    j = self.II(sat_, self.nav.na)
                    if reset and self.nav.x[j] != 0.0:
                        self.initx(0.0, 0.0, j)
                        if self.nav.monlevel > 0:
                            self.nav.fout.write(f"{time2str(obs.t)}  {sat2id(sat_)} - reset ionosphere\n")

            # Calculate bias and ionospheric delay for each satellite
            bias = np.zeros(ns)
            ion = np.zeros(ns)
            for i in range(ns):
                if np.any(self.nav.edt[sat[i]-1, :] > 0):  # Skip invalid observations
                    continue

                # Dual-frequency ionospheric delay calculation
                if self.nav.nf > 1 and self.nav.niono > 0:
                    sig1 = obs.sig[sys[i]][uTYP.C][0]
                    sig2 = obs.sig[sys[i]][uTYP.C][1]
                    pr1 = obs.P[i, 0]
                    pr2 = obs.P[i, 1]

                    if pr1 == 0.0 or pr2 == 0.0:  # Skip zero observations
                        continue

                    # Get frequency values
                    f1, f2 = (sig1.frequency(self.nav.glo_ch[sat[i]]), sig2.frequency(self.nav.glo_ch[sat[i]])) if sys[i] == uGNSS.GLO else (sig1.frequency(), sig2.frequency())

                    # Compute ionospheric delay
                    ion[i] = (pr1 - pr2) / (1.0 - (f1 / f2)**2)

                # Get pseudorange and carrier-phase observation
                sig = obs.sig[sys[i]][uTYP.L][f]
                fi = sig.frequency(self.nav.glo_ch[sat[i]]) if sys[i] == uGNSS.GLO else sig.frequency()
                lam = rCST.CLIGHT / fi
                cp = obs.L[i, f]
                pr = obs.P[i, f]

                if cp == 0.0 or pr == 0.0 or lam is None:  # Skip invalid data
                    continue

                # Calculate phase ambiguity bias
                bias[i] = cp - pr / lam + 2.0 * ion[i] / lam * (f1 / fi)**2

            # Initialize phase ambiguity and ionospheric delay
            for i in range(ns):
                j = self.IB(sat[i], f, self.nav.na)
                if bias[i] != 0.0 and self.nav.x[j] == 0.0:
                    self.initx(bias[i], self.nav.sig_n0**2, j)
                    if self.nav.monlevel > 0:
                        self.nav.fout.write(f"{time2str(obs.t)}  {sat2id(sat[i])} - init ambiguity {bias[i]:12.3f}\n")

                if self.nav.niono > 0:
                    j = self.II(sat[i], self.nav.na)
                    if ion[i] != 0 and self.nav.x[j] == 0.0:
                        self.initx(ion[i], self.nav.sig_ion0**2, j)
                        if self.nav.monlevel > 0:
                            self.nav.fout.write(f"{time2str(obs.t)}  {sat2id(sat[i])} - init ionosphere {ion[i]:12.3f}\n")

        return 0


    def find_bias(self, cs, sigref, sat, inet=0):
        """ find satellite signal bias from correction """
        nf = len(sigref)
        v = np.zeros(nf)

        if nf == 0:
            return v

        ctype = sigref[0].typ
        if ctype == uTYP.C:
            if cs.lc[inet].cbias is None or \
                    sat not in cs.lc[inet].cbias.keys():
                return v
            sigc = cs.lc[inet].cbias[sat]
        else:
            if cs.lc[inet].pbias is None or \
                    sat not in cs.lc[inet].pbias.keys():
                return v
            sigc = cs.lc[inet].pbias[sat]

        # work-around for Galileo HAS: L2P -> L2W
        if cs.cssrmode in [sc.GAL_HAS_SIS, sc.GAL_HAS_IDD]:
            if ctype == uTYP.C and rSigRnx('GC2P') in sigc.keys():
                sigc[rSigRnx('GC2W')] = sigc[rSigRnx('GC2P')]
            if ctype == uTYP.L and rSigRnx('GL2P') in sigc.keys():
                sigc[rSigRnx('GL2W')] = sigc[rSigRnx('GL2P')]

        for k, sig in enumerate(sigref):
            if sig in sigc.keys():
                v[k] = sigc[sig]
            elif sig.toAtt('X') in sigc.keys():
                v[k] = sigc[sig.toAtt('X')]
        return v
    
    def shapiro(rsat, rrcv):
        """ relativistic shapiro effect """
        rs = np.linalg.norm(rsat)
        rr = np.linalg.norm(rrcv)
        rrs = np.linalg.norm(rsat-rrcv)
        corr = (2*gn.rCST.GME/gn.rCST.CLIGHT**2)*np.log((rs+rr+rrs)/(rs+rr-rrs))
        return corr

    def zdres(self, obs, cs, bsx, rs, vs, dts, rr, rtype=1):
        """Non-differential residual computation"""
        
        # Constants
        _c = rCST.CLIGHT
        ns2m = _c * 1e-9  # Convert nanoseconds to meters
        nf = self.nav.nf  # Number of frequencies
        n = len(obs.P)  # Number of satellites
        
        # Initialize variables
        y = np.zeros((n, nf * 2))  # Residual matrix for phase and pseudorange
        el = np.zeros(n)  # Satellite elevation angles
        e = np.zeros((n, 3))  # Line-of-sight vectors
        rr_ = rr.copy()  # Receiver position

        # Get receiver geodetic position
        pos = ecef2pos(rr_)

        # Get tropospheric delays
        trop_hs, trop_wet, _ = tropmodel(obs.t, pos, model=self.nav.trpModel)

        # CSSR-based corrections (optional)
        if self.nav.trop_opt == 2 or self.nav.iono_opt == 2:
            inet = cs.find_grid_index(pos)
            dlat, dlon = cs.get_dpos(pos)
        else:
            inet = -1

        # Tropo from CSSR
        if self.nav.trop_opt == 2:
            trph, trpw = cs.get_trop(dlat, dlon)
            trop_hs0, trop_wet0, _ = tropmodel(obs.t, [pos[0], pos[1], 0], model=self.nav.trpModel)
            r_hs = trop_hs / trop_hs0
            r_wet = trop_wet / trop_wet0

        # Iono from CSSR
        if self.nav.iono_opt == 2:
            stec = cs.get_stec(dlat, dlon)

        # Prepare for range and phase corrections
        cpc = np.zeros((n, nf))  # Carrier-phase corrections
        prc = np.zeros((n, nf))  # Pseudorange corrections

        for i in range(n):
            sat = obs.sat[i]
            sys, _ = sat2prn(sat)

            # Skip satellites with edited observations
            if np.any(self.nav.edt[sat-1, :] > 0):
                continue

            # Skip if satellite not in CSSR
            if inet > 0 and sat not in cs.lc[inet].sat_n:
                continue

            # Get signals
            sigsPR = obs.sig[sys][uTYP.C]  # Pseudorange signals
            sigsCP = obs.sig[sys][uTYP.L]  # Carrier-phase signals

            # Calculate wavelength and frequency
            if sys == uGNSS.GLO:
                lam = np.array([s.wavelength(self.nav.glo_ch[sat]) for s in sigsCP])
                frq = np.array([s.frequency(self.nav.glo_ch[sat]) for s in sigsCP])
            else:
                lam = np.array([s.wavelength() for s in sigsCP])
                frq = np.array([s.frequency() for s in sigsCP])

            # Biases (optional from BSX or CSSR)
            cbias, pbias = np.zeros(nf), np.zeros(nf)
            if self.nav.ephopt == 4:  # From Bias-SINEX
                cbias = np.array([-bsx.getosb(sat, obs.t, s) * ns2m for s in sigsPR])
                pbias = np.array([-bsx.getosb(sat, obs.t, s) * ns2m for s in sigsCP])
            elif cs:
                cbias = self.find_bias(cs, sigsPR, sat)
                pbias = self.find_bias(cs, sigsCP, sat)

            # Check for invalid biases
            if np.isnan(cbias).any() or np.isnan(pbias).any():
                continue

            # Compute geometric distance, elevation, and line-of-sight
            r, e[i, :] = geodist(rs[i, :], rr_)
            _, el[i] = satazel(pos, e[i, :])
            if el[i] < self.nav.elmin:
                continue

            # Shapiro relativistic effect
            relatv = self.shapiro(rs[i, :], rr_)

            # Tropospheric delay
            if self.nav.iono_opt == 2:
                trop = mapfh * trph * r_hs + mapfw * trpw * r_wet
            else:
                mapfh, mapfw = tropmapf(obs.t, pos, el[i], model=self.nav.trpModel)
                trop = mapfh * trop_hs + mapfw * trop_wet

            # Ionospheric delay
            if self.nav.iono_opt == 2:
                idx_l = cs.lc[inet].sat_n.index(sat)
                iono = np.array([40.3e16 / (f * f) * stec[idx_l] for f in frq])
            else:
                iono = np.zeros(nf)

            # Apply range corrections
            prc[i, :] = trop + iono - cbias
            cpc[i, :] = trop - iono - pbias

            # Adjust geometric distance with relativistic and clock bias corrections
            r += relatv - _c * dts[i]

            # Compute phase and pseudorange residuals for each frequency
            for f in range(nf):
                y[i, f] = obs.L[i, f] * lam[f] - (r + cpc[i, f])  # Carrier-phase residual
                y[i, f + nf] = obs.P[i, f] - (r + prc[i, f])  # Pseudorange residual

        return y, e, el


    def sdres(self, obs, x, y, e, sat, el):
        """
        Compute single-difference (SD) phase/code residuals.

        Parameters
        ----------
        obs : Obs()
            Data structure with observations
        x   : np.array
            State vector elements (positions, tropospheric delays, ambiguities, etc.)
        y   : np.array
            Corrected, un-differenced observations (code/phase measurements)
        e   : np.array
            Line-of-sight vectors for satellites
        sat : np.array of int
            List of satellite IDs
        el  : np.array of float
            Elevation angles for the satellites

        Returns
        -------
        v : np.array of float
            Residuals for the SD measurements
        H : np.array of float
            Jacobian matrix (partial derivatives of state variables)
        R : np.array of float
            Covariance matrix for the SD measurements
        """

        nf = self.nav.nf  # number of frequencies (or signals)
        ns = len(el)      # number of satellites
        nc = len(obs.sig.keys())  # number of constellations
        mode = 1 if len(y) == ns else 0  # Mode: 0 for DD, 1 for SD

        # Initialize residuals, Jacobian matrix, and covariance placeholders
        H = np.zeros((ns * nf * 2, self.nav.nx))
        v = np.zeros(ns * nf * 2)
        Ri = np.zeros(ns * nf * 2)
        Rj = np.zeros(ns * nf * 2)
        
        # Geodetic position from ECEF coordinates
        pos = ecef2pos(x[:3])

        # Loop over constellations (GPS, GLONASS, etc.)
        for sys in obs.sig.keys():

            # Loop over twice the number of frequencies: first for carrier-phase, then pseudorange
            for f in range(nf * 2):
                idx = self.sysidx(sat, sys)  # Select satellites for this constellation

                if len(idx) == 0:
                    continue

                # Reference satellite with highest elevation
                i = idx[np.argmax(el[idx])]

                for j in idx:

                    # Skip edited or invalid measurements
                    if np.any(self.nav.edt[sat[j] - 1, :] > 0) or y[i, f] == 0.0 or y[j, f] == 0.0 or i == j:
                        continue

                    # Frequency and ionospheric ratio calculations
                    freq0 = obs.sig[sys][uTYP.L][0].frequency() if sys != uGNSS.GLO else obs.sig[sys][uTYP.L][0].frequency(0)
                    sig = obs.sig[sys][uTYP.L if f < nf else uTYP.C][f % nf]
                    freq = sig.frequency(self.nav.glo_ch[sat[j]]) if sys == uGNSS.GLO else sig.frequency()
                    mu = -(freq0 / freq) ** 2 if f < nf else +(freq0 / freq) ** 2

                    # Compute SD or DD residual
                    if mode == 0:  # Double-difference (DD)
                        v[nv] = (y[i, f] - y[i + ns, f]) - (y[j, f] - y[j + ns, f])
                    else:  # Single-difference (SD)
                        v[nv] = y[i, f] - y[j, f]

                    # Line-of-sight vector difference
                    H[nv, :3] = -e[i, :] + e[j, :]

                    # Tropospheric correction (if estimated)
                    if self.nav.ntrop > 0:
                        mapfwi, mapfwj = tropmapf(obs.t, pos, el[i], model=self.nav.trpModel)[1], tropmapf(obs.t, pos, el[j], model=self.nav.trpModel)[1]
                        idx_trop = self.IT(self.nav.na)
                        H[nv, idx_trop] = mapfwi - mapfwj
                        v[nv] -= (mapfwi - mapfwj) * x[idx_trop]

                    # Ionospheric correction (if estimated)
                    if self.nav.niono > 0:
                        idx_iono_i, idx_iono_j = self.II(sat[i], self.nav.na), self.II(sat[j], self.nav.na)
                        H[nv, idx_iono_i], H[nv, idx_iono_j] = mu, -mu
                        v[nv] -= mu * (x[idx_iono_i] - x[idx_iono_j])

                    # Ambiguity correction for carrier-phase
                    if f < nf:
                        idx_amb_i, idx_amb_j = self.IB(sat[i], f, self.nav.na), self.IB(sat[j], f, self.nav.na)
                        lami = sig.wavelength(self.nav.glo_ch[sat[i]]) if sys == uGNSS.GLO else sig.wavelength()
                        lamj = lami
                        H[nv, idx_amb_i], H[nv, idx_amb_j] = lami, -lamj
                        v[nv] -= lami * (x[idx_amb_i] - x[idx_amb_j])

                    # Measurement variances
                    Ri[nv] = self.varerr(self.nav, el[i], f)
                    Rj[nv] = self.varerr(self.nav, el[j], f)
                    self.nav.vsat[sat[i] - 1, f], self.nav.vsat[sat[j] - 1, f] = 1, 1

                    nv += 1

        # Resize the residual, Jacobian, and covariance matrices to fit the data
        v = np.resize(v, nv)
        H = np.resize(H, (nv, self.nav.nx))
        R = self.ddcov(nb, b, Ri, Rj, nv)

        return v, H, R


    def ddcov(self, nb, n, Ri, Rj, nv):
        """ DD measurement error covariance """
        R = np.zeros((nv, nv))
        k = 0
        for b in range(n):
            for i in range(nb[b]):
                for j in range(nb[b]):
                    R[k+i, k+j] = Ri[k+i]
                    if i == j:
                        R[k+i, k+j] += Rj[k+i]
            k += nb[b]
        return R

    def kfupdate(self, x, P, H, v, R):
        """
        Kalman filter measurement update.

        Parameters:
        x (ndarray): State estimate vector
        P (ndarray): State covariance matrix
        H (ndarray): Observation model matrix
        v (ndarray): Innovation vector
                     (residual between measurement and prediction)
        R (ndarray): Measurement noise covariance

        Returns:
        x (ndarray): Updated state estimate vector
        P (ndarray): Updated state covariance matrix
        S (ndarray): Innovation covariance matrix
        """

        PHt = P@H.T
        S = H@PHt+R
        K = PHt@np.linalg.inv(S)
        x += K@v
        # P = P - K@H@P
        IKH = np.eye(P.shape[0])-K@H
        P = IKH@P@IKH.T + K@R@K.T  # Joseph stabilized version

        return x, P, S

    def restamb(self, bias, nb):
        """ Restore SD ambiguity """
        na = self.nav.na
        xa = self.nav.x.copy()
        xa[:na] = self.nav.xa[:na]  # Use the ambiguity-corrected values

        for m in range(uGNSS.GNSSMAX):
            for f in range(self.nav.nf):
                indices = [self.IB(i + 1, f, na) for i in range(uGNSS.MAXSAT)
                        if sat2prn(i + 1)[0] == m and self.nav.fix[i, f] == 2]
                if len(indices) < 2:
                    continue

                xa[indices[0]] = self.nav.x[indices[0]]
                for i in range(1, len(indices)):
                    xa[indices[i]] = xa[indices[0]] - bias[i - 1]
                    
        return xa


    def ddidx(self, nav, sat):
        """ Index for SD to DD transformation matrix D """
        na = nav.na
        n = uGNSS.MAXSAT
        ix = []
        nav.fix = np.zeros((n, nav.nf), dtype=int)

        for m in range(uGNSS.GNSSMAX):
            k = na
            for f in range(nav.nf):
                valid_indices = []
                for i in range(k, k + n):
                    sat_i = i - k + 1
                    sys, _ = sat2prn(sat_i)
                    if sys == m and sat_i in sat and nav.x[i] != 0.0 and nav.vsat[sat_i - 1, f] != 0:
                        if nav.el[sat_i - 1] >= nav.elmaskar:
                            nav.fix[sat_i - 1, f] = 2
                            valid_indices.append(i)
                
                for j in valid_indices:
                    sat_j = j - k + 1
                    if j != k and sat_j in sat and nav.x[j] != 0.0 and nav.vsat[sat_j - 1, f] != 0:
                        if nav.el[sat_j - 1] >= nav.elmaskar:
                            ix.append([j, k])
                            nav.fix[sat_j - 1, f] = 2
                k += n

        return np.array(ix)


    def resamb_lambda(self, sat):
        """ Resolve integer ambiguity using the LAMBDA method """
        na = self.nav.na
        nx = self.nav.nx

        # Find double-differenced indices
        ix = self.ddidx(self.nav, sat)
        nb = len(ix)
        
        if nb <= 0:
            print("No valid DD")
            return -1, -1

        # Calculate differences and matrices
        y = self.nav.x[ix[:, 0]] - self.nav.x[ix[:, 1]]
        DP = self.nav.P[ix[:, 0], na:nx] - self.nav.P[ix[:, 1], na:nx]
        Qb = DP[:, ix[:, 0] - na] - DP[:, ix[:, 1] - na]
        Qab = self.nav.P[:na, ix[:, 0]] - self.nav.P[:na, ix[:, 1]]

        # Apply LAMBDA method
        b, s = mlambda(y, Qb)
        if s[0] <= 0.0 or s[1] / s[0] >= self.nav.thresar:
            self.nav.xa = self.nav.x[:na].copy()
            self.nav.Pa = self.nav.P[:na, :na].copy()
            bias = b[:, 0]
            y -= bias
            K = Qab @ np.linalg.inv(Qb)
            self.nav.xa -= K @ y
            self.nav.Pa -= K @ Qab.T

            # Restore SD ambiguity
            xa = self.restamb(bias, nb)
        else:
            nb = 0
            xa = np.zeros(na)

        return nb, xa



    def holdamb(self, xa):
        """ Hold integer ambiguity """
        nb = self.nav.nx - self.nav.na
        v = np.zeros(nb)
        H = np.zeros((nb, self.nav.nx))
        nv = 0

        for m in range(uGNSS.GNSSMAX):
            for f in range(self.nav.nf):
                indices = [self.IB(i + 1, f, self.nav.na) for i in range(uGNSS.MAXSAT)
                        if sat2prn(i + 1)[0] == m and self.nav.fix[i, f] == 2]
                for i in range(1, len(indices)):
                    v[nv] = (xa[indices[0]] - xa[indices[i]]) - \
                            (self.nav.x[indices[0]] - self.nav.x[indices[i]])
                    H[nv, indices[0]] = 1.0
                    H[nv, indices[i]] = -1.0
                    nv += 1
                    self.nav.fix[indices[0], f] = 3  # Hold ambiguity
                    self.nav.fix[indices[i], f] = 3  # Hold ambiguity

        if nv > 0:
            R = np.eye(nv) * self.VAR_HOLDAMB
            self.nav.x, self.nav.P, _ = self.kfupdate(self.nav.x, self.nav.P, H[:nv], v[:nv], R)

        return 0


    def qcedit(self, obs, rs, dts, svh, rr=None):
        """ Coarse quality control and editing of observations """

        # Calculate predicted position
        tt = timediff(obs.t, self.nav.t)
        rr_ = rr if rr is not None else self.nav.x[:3] + self.nav.x[3:6] * tt

        # Convert ECEF to geodetic position
        pos = ecef2pos(rr_)

        # Initialize editing results
        ns = uGNSS.MAXSAT
        self.nav.edt = np.zeros((ns, self.nav.nf), dtype=int)

        # Process each satellite
        sat = []
        for i in range(ns):
            sat_i = i + 1
            sys_i, _ = sat2prn(sat_i)

            # Skip if satellite is not in the observation
            if sat_i not in obs.sat:
                self.nav.edt[i, :] = 1
                continue

            # Skip if satellite is excluded
            if sat_i in self.nav.excl_sat:
                self.nav.edt[i, :] = 1
                self._log_exclusion(obs.t, sat_i)
                continue

            j = np.where(obs.sat == sat_i)[0][0]

            # Check for invalid orbit and clock offset
            if np.isnan(rs[j, :]).any() or np.isnan(dts[j]):
                self.nav.edt[i, :] = 1
                self._log_invalid_eph(obs.t, sat_i)
                continue

            # Check satellite health
            if svh[j] > 0:
                self.nav.edt[i, :] = 1
                self._log_unhealthy(obs.t, sat_i)
                continue

            # Check elevation angle
            _, e = geodist(rs[j, :], rr_)
            _, el = satazel(pos, e)
            if el < self.nav.elmin:
                self.nav.edt[i, :] = 1
                self._log_low_elevation(obs.t, sat_i, el)
                continue

            # Check pseudorange, carrier-phase, and C/N0 signals
            if not self._check_signals(obs, j, sys_i, sat_i):
                continue

            sat.append(sat_i)

        return np.array(sat, dtype=int)

    def _log_exclusion(self, time, sat_i):
        if self.nav.monlevel > 0:
            self.nav.fout.write(f"{time2str(time)}  {sat2id(sat_i)} - edit - satellite excluded\n")

    def _log_invalid_eph(self, time, sat_i):
        if self.nav.monlevel > 0:
            self.nav.fout.write(f"{time2str(time)}  {sat2id(sat_i)} - edit - invalid eph\n")

    def _log_unhealthy(self, time, sat_i):
        if self.nav.monlevel > 0:
            self.nav.fout.write(f"{time2str(time)}  {sat2id(sat_i)} - edit - satellite unhealthy\n")

    def _log_low_elevation(self, time, sat_i, elevation):
        if self.nav.monlevel > 0:
            self.nav.fout.write(f"{time2str(time)}  {sat2id(sat_i)} - edit - low elevation {np.rad2deg(elevation):5.1f} deg\n")

    def _check_signals(self, obs, j, sys_i, sat_i):
        """ Check pseudorange, carrier-phase, and C/N0 signals """
        for f in range(self.nav.nf):
            if obs.lli[j, f] == 1:
                self.nav.edt[sat_i - 1, f] = 1
                self._log_signal_issue(obs.t, sat_i, obs.sig[sys_i][uTYP.L].str(), "LLI")
                return False

            if obs.P[j, f] == 0.0:
                self.nav.edt[sat_i - 1, f] = 1
                self._log_signal_issue(obs.t, sat_i, obs.sig[sys_i][uTYP.C].str(), "invalid PR obs")
                return False

            if obs.L[j, f] == 0.0:
                self.nav.edt[sat_i - 1, f] = 1
                self._log_signal_issue(obs.t, sat_i, obs.sig[sys_i][uTYP.L].str(), "invalid CP obs")
                return False

            cnr_min = self.nav.cnr_min_gpy if obs.sig[sys_i][uTYP.S].isGPS_PY() else self.nav.cnr_min
            if obs.S[j, f] < cnr_min:
                self.nav.edt[sat_i - 1, f] = 1
                self._log_signal_issue(obs.t, sat_i, obs.sig[sys_i][uTYP.S].str(), f"low C/N0 {obs.S[j, f]:4.1f} dB-Hz")
                return False

        return True

    def _log_signal_issue(self, time, sat_i, signal_type, issue):
        if self.nav.monlevel > 0:
            self.nav.fout.write(f"{time2str(time)}  {sat2id(sat_i)} - edit {signal_type} - {issue}\n")


    def base_process(self, obs, obsb, rs, dts, svh):
        """ Process base station observations for RTK """

        # Get base station satellite positions and velocities
        rsb, vsb, dtsb, svhb, _ = satposs(obsb, self.nav)

        # Calculate residuals for base station
        yr, er, elr = self.zdres(obsb, None, None, rsb, vsb, dtsb, self.nav.rb, 0)

        # Quality control for both base and rover observations
        sat_ed_r = self.qcedit(obsb, rsb, dtsb, svhb, rr=self.nav.rb)
        sat_ed_u = self.qcedit(obs, rs, dts, svh)

        # Common satellites between base and rover
        sat_ed = np.intersect1d(sat_ed_u, sat_ed_r)
        ir = np.intersect1d(obsb.sat, sat_ed, return_indices=True)[1]
        iu = np.intersect1d(obs.sat, sat_ed, return_indices=True)[1]

        # Prepare residuals and directions
        y, e = np.zeros((len(iu) * 2, self.nav.nf * 2)), np.zeros((len(iu) * 2, 3))
        y[len(iu):], e[len(iu):] = yr[ir, :], er[ir, :]

        # Adjust rover observations based on base station data
        obs_ = deepcopy(obs)
        obs_.sat = obs.sat[iu]
        obs_.L, obs_.P = obs.L[iu, :] - obsb.L[ir, :], obs.P[iu, :] - obsb.P[ir, :]

        return y, e, iu, obs_


    def process(self, obs, cs=None, orb=None, bsx=None, obsb=None):
        """
        RTK positioning
        """
        if len(obs.sat) == 0:
            return

        # Get satellite positions, velocities, and clock offsets
        rs, vs, dts, svh, nsat = satposs(obs, self.nav, cs=cs, orb=orb)
        
        if nsat < 6:
            print(f"Too few satellites < 6: nsat={nsat}")
            return

        # Quality control on observations
        sat_ed = self.qcedit(obs, rs, dts, svh)

        # Process base station if provided, else default to PPP/PPP-RTK
        if obsb:
            y, e, iu, obs_ = self.base_process(obs, obsb, rs, dts, svh)
        else:
            iu = np.where(np.isin(obs.sat, sat_ed))[0]
            y, e = np.zeros((len(iu), self.nav.nf * 2)), np.zeros((len(iu), 3))
            obs_ = obs
        
        if len(iu) < 6:
            print(f"Too few satellites < 6: ns={len(iu)}")
            return

        # Kalman filter setup and residuals calculation
        self.udstate(obs_)
        xa, xp = np.zeros(self.nav.nx), self.nav.x.copy()

        # Compute residuals
        yu, eu, elu = self.zdres(obs, cs, bsx, rs, vs, dts, xp[:3])
        y[:len(iu), :], e[:len(iu), :], el = yu[iu, :], eu[iu, :], elu[iu]
        
        # Update navigation state
        self.nav.sat, self.nav.el[sat - 1], self.nav.y = obs.sat[iu], el, y

        # Ensure enough valid observations
        if y.shape[0] < 6:
            self.nav.P[np.diag_indices(3)] = 1.0
            self.nav.smode = 5
            return -1

        # Kalman filter measurement update
        v, H, R = self.sdres(obs, xp, y, e, obs.sat[iu], el)
        xp, Pp, _ = self.kfupdate(xp, self.nav.P.copy(), H, v, R)

        # Check and update for valid solution
        yu, eu, elu = self.zdres(obs, cs, bsx, rs, vs, dts, xp[:3])
        y, e = yu[iu, :], eu[iu, :]
        if y.shape[0] < 6:
            return -1

        v, H, R = self.sdres(obs, xp, y, e, obs.sat[iu], el)
        if self.valpos(v, R):
            self.nav.x, self.nav.P = xp, Pp
            self.nav.ns = np.sum(self.nav.vsat[obs.sat[iu] - 1, 0] > 0)
            self.nav.smode = 4 if self.nav.armode > 0 else 5
        else:
            self.nav.smode = 0

        # Ambiguity resolution if enabled
        if self.nav.armode > 0:
            nb, xa = self.resamb_lambda(obs.sat[iu])
            if nb > 0:
                yu, eu, elu = self.zdres(obs, cs, bsx, rs, vs, dts, xa[:3])
                v, H, R = self.sdres(obs, xa, yu[iu, :], eu[iu, :], obs.sat[iu], el)
                if self.valpos(v, R):
                    if self.nav.armode == 3:
                        self.holdamb(xa)
                    self.nav.smode = 4

        # Store current epoch time
        self.nav.t = obs.t
        return 0
