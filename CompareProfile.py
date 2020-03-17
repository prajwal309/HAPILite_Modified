from hapi import PROFILE_VOIGT
import numpy as np
import matplotlib.pyplot as plt
from astropy.modeling.functional_models import Voigt1D
from scipy.special import wofz
import time

from lib import LineProfiles



sg0 = 2086.321945
GamD =  0.004465456697268726
Gam0 = 0.07616950296974222
wn = np.arange(sg0-5.0, sg0+5.0, 0.01)

StartTime = time.time()
voi_HAPI = PROFILE_VOIGT(sg0,GamD,Gam0, wn)[0]
print("The time taken is::", time.time()-StartTime)



#LocalVoigtValues = LocalVoigtV(wn, GamD, Gam0)
StartTime = time.time()
LocalVoigtValues = LineProfiles.PROFILE_VOIGT(sg0, GamD, Gam0, wn)
print("The time taken is::", time.time()-StartTime)


fig, (ax0,ax1) = plt.subplots(nrows=2, ncols=1, figsize=(12,8), sharex=True)
ax0.plot(wn, voi_HAPI, "ko", label="HAPI Profile")
ax0.plot(wn, LocalVoigtValues, "r-", label="Local Profile")
ax0.set_ylabel("HAPI Profile")
ax0.legend()
ax1.plot(wn, voi_HAPI - LocalVoigtValues, "ko", label="Local Voigt Values")
ax1.set_ylabel("Residuals")
ax1.set_xlabel("WaveNumber")
plt.savefig("Profile Comparison")
plt.show()
