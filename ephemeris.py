# Import necessary modules
import numpy as np
from gnss import uGNSS, rCST, sat2prn, timediff, timeadd, vnorm
from gnss import gtime_t, Geph, Eph

# Constants for Kepler's Equation
MAX_ITER_KEPLER = 30
RTOL_KEPLER = 1e-13

# Max time difference for each GNSS
MAXDTOE_t = {uGNSS.GPS: 7201.0, uGNSS.GAL: 14400.0, uGNSS.QZS: 7201.0,
             uGNSS.BDS: 7201.0, uGNSS.IRN: 7201.0, uGNSS.GLO: 1800.0,
             uGNSS.SBS: 360.0}

# Find the ephemeris for a satellite
def findeph(nav, t, sat, iode=-1, mode=0):
    sys, _ = sat2prn(sat)
    eph, tmin = None, MAXDTOE_t[sys] + 1.0
    for eph_ in nav:
        if eph_.sat == sat and (iode < 0 or iode == eph_.iode) and eph_.mode == mode:
            dt = abs(timediff(t, eph_.toe))
            if dt <= tmin and dt <= MAXDTOE_t[sys]:
                eph, tmin = eph_, dt
    return eph

# Adjust time considering week rollover
def dtadjust(t1, t2, tw=604800):
    dt = timediff(t1, t2)
    return dt - tw if dt > tw else dt + tw if dt < -tw else dt

# Calculate satellite dynamics
def deq(x, acc):
    r2 = np.dot(x[:3], x[:3])
    if r2 <= 0:
        return np.zeros(6)

    r3 = r2 * np.sqrt(r2)
    omg2 = rCST.OMGE_GLO**2
    a = 1.5 * rCST.J2_GLO * rCST.MU_GLO * rCST.RE_GLO**2 / r2 / r3
    b = 5.0 * x[2]**2 / r2
    c = -rCST.MU_GLO / r3 - a * (1.0 - b)

    xdot = np.zeros(6)
    xdot[:3] = x[3:]
    xdot[3] = (c + omg2) * x[0] + 2.0 * rCST.OMGE_GLO * x[4]
    xdot[4] = (c + omg2) * x[1] - 2.0 * rCST.OMGE_GLO * x[3]
    xdot[5] = (c - 2.0 * a) * x[2]
    xdot[3:] += acc
    return xdot

# Calculate GLONASS satellite orbit
def glorbit(t, x, acc):
    k1 = deq(x, acc)
    w = x + k1 * t / 2.0
    k2 = deq(w, acc)
    w = x + k2 * t / 2.0
    k3 = deq(w, acc)
    w = x + k3 * t
    k4 = deq(w, acc)
    return x + (k1 + 2.0 * k2 + 2.0 * k3 + k4) * t / 6.0

# GLONASS satellite position calculation
def geph2pos(time: gtime_t, geph: Geph, flg_v=False, TSTEP=1.0):
    t, dts = timediff(time, geph.toe), -geph.taun + geph.gamn * timediff(time, geph.toe)
    x = np.concatenate([geph.pos, geph.vel])
    tt = -TSTEP if t < 0 else TSTEP
    while abs(t) > 1e-9:
        tt = t if abs(t) < TSTEP else tt
        x = glorbit(tt, x, geph.acc)
        t -= tt
    return (x[:3], x[3:], dts) if flg_v else (x[:3], dts)

# Compute clock offset for GLONASS satellite
def geph2clk(time: gtime_t, geph: Geph):
    t = ts = timediff(time, geph.toe)
    for _ in range(2):
        t = ts - (-geph.taun + geph.gamn * t)
    return -geph.taun + geph.gamn * t

# Kepler's equation solver for eccentric anomaly
def eccentricAnomaly(M, e):
    E = M
    for _ in range(10):
        Eold, sE = E, np.sin(E)
        E = M + e * sE
        if abs(Eold - E) < 1e-12:
            break
    return E, sE

# Get constants for different GNSS systems
def sys2MuOmega(sys):
    if sys == uGNSS.GAL:
        return rCST.MU_GAL, rCST.OMGE_GAL
    elif sys == uGNSS.BDS:
        return rCST.MU_BDS, rCST.OMGE_BDS
    return rCST.MU_GPS, rCST.OMGE

# Calculate satellite position from ephemeris
def eph2pos(t: gtime_t, eph: Eph, flg_v=False):
    sys, _ = sat2prn(eph.sat)
    mu, omge = sys2MuOmega(sys)
    dt = dtadjust(t, eph.toe)
    n = np.sqrt(mu / eph.A**3) + eph.deln
    M = eph.M0 + n * dt
    E, sE = eccentricAnomaly(M, eph.e)
    nue = 1 - eph.e * np.cos(E)
    r = eph.A * nue + eph.crc * np.cos(2 * (np.arctan2(np.sin(E), nue) + eph.omg))
    u = r * np.cos(eph.i0 + eph.idot * dt)
    rs = np.array([u * np.cos(eph.OMG0), u * np.sin(eph.OMG0)])
    
    if flg_v:  # include velocity if requested
        return rs, dts, vs
    return rs, dts

# Compute satellite clock offset from ephemeris
def eph2clk(time, eph):
    t = timediff(time, eph.toc)
    for _ in range(2):
        t -= eph.af0 + eph.af1 * t + eph.af2 * t**2
    return eph.af0 + eph.af1 * t + eph.af2 * t**2

# Main function to calculate satellite position, velocity, and clock offset
def satposs(obs, nav, cs=None, orb=None):
    n, nsat = obs.sat.shape[0], 0
    rs, vs, dts, svh = np.zeros((n, 3)), np.zeros((n, 3)), np.zeros(n), np.zeros(n, dtype=int)

    for i in range(n):
        sat = obs.sat[i]
        sys, _ = sat2prn(sat)

        if sys not in obs.sig.keys():
            continue

        pr = obs.P[i, 0]
        t = timeadd(obs.t, -pr / rCST.CLIGHT)
        eph = findeph(nav.eph, t, sat)

        if not eph:
            continue

        rs[i, :], vs[i, :], dts[i] = eph2pos(t, eph, True)
        svh[i], nsat = eph.svh, nsat + 1

    return rs, vs, dts, svh, nsat
