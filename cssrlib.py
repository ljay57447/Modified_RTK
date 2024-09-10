"""
module for Compact SSR processing
"""

from enum import IntEnum

class sCSSRTYPE(IntEnum):
    QZS_CLAS = 0     # QZS CLAS PPP-RTK
    QZS_MADOCA = 1   # MADOCA-PPP
    GAL_HAS_SIS = 2  # Galileo HAS Signal-In-Space
    GAL_HAS_IDD = 3  # Galileo HAS Internet Data Distribution
    BDS_PPP = 4      # BDS PPP
    IGS_SSR = 5
    RTCM3_SSR = 6
    PVS_PPP = 7      # PPP via SouthPAN
    SBAS_L1 = 8      # L1 SBAS
    SBAS_L5 = 9      # L5 SBAS (DFMC)
    DGPS = 10        # DGPS (QZSS SLAS)
    STDPOS = 11

class sCType(IntEnum):
    """ class to define correction message types """
    MASK = 0
    ORBIT = 1
    CLOCK = 2
    CBIAS = 3
    PBIAS = 4
    STEC = 5
    TROP = 6
    URA = 7
    AUTH = 8
    HCLOCK = 9
    VTEC = 10
    MAX = 11