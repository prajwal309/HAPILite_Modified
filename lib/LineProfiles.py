import numpy as np
from numba import njit
from scipy.special import wofz


#Define the constant
cSqrtPI = np.sqrt(np.pi)
cSqrtLN2 = np.sqrt(np.log(2))
FinalConst = cSqrtPI*cSqrtLN2

cLn2 = 0.6931471805599
cSqrtLn2divSqrtPi = 0.469718639319144059835

Const1 = 1.1774100225154747 #np.sqrt(2 * np.log(2))
Const2 = 1.4142135623730951 #sqrt(2)
Const3 = 2.5066282746310002 #np.sqrt(2*np.pi)

alpha = 0.18121


def PROFILE_VOIGT(sg0,GamD,Gam0,sg):
    """
    # Input parameters:
    #   sg0: Unperturbed line position in cm-1 (Input).
    #   GamD: Doppler HWHM in cm-1 (Input)
    #   Gam0: Speed-averaged line-width in cm-1 (Input).
    #   sg: Current WaveNumber of the Computation in cm-1 (Input).
    """
    sg = sg - sg0
    sigma = GamD/ Const1
    return np.real(wofz((sg + 1j*Gam0)/sigma/Const2))/ sigma/Const3


def PROFILE_PSEUDOVOIGT(sg0,GamD,Gam0,sg):
    """
    # This formulation is taken from Liu et. al 2001
    and is the fastest of all different method.
    ###############################################################
    ###############################################################
    # Input parameters:
    #   sg0: Unperturbed line position in cm-1 (Input).
    #   GamD: Doppler HWHM in cm-1 (Input)
    #   Gam0: Speed-averaged line-width in cm-1 (Input).
    #   sg: Current WaveNumber of the Computation in cm-1 (Input).
    ################################################################
    """
    d = (Gam0-GamD)/(Gam0+GamD)
    beta = 0.023665*np.exp(0.6*d)+0.00418*np.exp(-1.9*d)
    cL = 0.6818817+0.6129331*d-0.1838439*d*d-0.1156844*d*d*d
    cG = 0.3246017-0.6182531*d+0.1768139*d*d+0.1210944*d*d*d
    GamV = (Gam0 + GamD)*(1- alpha*(1-d*d) - beta*np.sin(np.pi*d))

    Profile = cL*1./np.pi*GamV/((sg - sg0)*(sg - sg0)+GamV*GamV) + \
              cG*cSqrtLn2divSqrtPi/GamV*np.exp(-cLn2*(sg-sg0)*(sg-sg0)/(GamV*GamV))
    return Profile



def PROFILE_LORENTZ(sg0,GamD,Gam0,sg):
    """
    # Lorentz profile.
    # Input parameters:
    #   sg0: Unperturbed line position in cm-1 (Input).
    #   GamD: Doppler HWHM in cm-1 (Input)
    #   Gam0: Speed-averaged line-width in cm-1 (Input).
    #   sg: Current WaveNumber of the Computation in cm-1 (Input).
    """
    return Gam0/(np.pi*(Gam0**2+(sg-sg0)**2))


def PROFILE_DOPPLER(sg0,GamD,Gam0,sg):
    """
    # Doppler profile.
    # Input parameters:
    #   sg0: Unperturbed line position in cm-1 (Input).
    #   GamD: Doppler HWHM in cm-1 (Input)
    #   Gam0: Speed-averaged line-width in cm-1 (Input).
    #   sg: Current WaveNumber of the Computation in cm-1 (Input).
    """
    return cSqrtLn2divSqrtPi*np.exp(-cLn2*((sg-sg0)/GamD)**2)/GamD
