import numpy as np
import time
import hapi
from HAPILite import CalcCrossSection
from lib.ReadComputeFunc import ReadData
import matplotlib.pyplot as plt


Molecule = "CO"
TempValue = 1000.0
P_Value = 0.1#0.00001

#TempValue = 1000.0
#P_Value = 10.0
OmegaWingValue = 500.0
OmegaRangeValue = [2086.32193,2093.410992]
WaveNumber = np.arange(OmegaRangeValue[0], OmegaRangeValue[1]+0.01, 0.01)
Env= {'T': TempValue, 'p': P_Value}


#StartTime = time.time()
hapi.db_begin('TestData')
nu_hapi_Doppler, abs_hapi_Doppler = hapi.absorptionCoefficient_Doppler(SourceTables=Molecule,OmegaGrid=WaveNumber,
                    HITRAN_units=True, Environment=Env,  GammaL='gamma_self', OmegaWing=OmegaWingValue, OmegaWingHW=0.0, LineShift=False)#, OmegaStep=0.01)
nu_hapi_Voigt, abs_hapi_Voigt = hapi.absorptionCoefficient_Voigt(SourceTables=Molecule,OmegaGrid=WaveNumber,
                    HITRAN_units=True, Environment=Env,  GammaL='gamma_air', OmegaWing=OmegaWingValue, OmegaWingHW=0.0, LineShift=False)#, OmegaStep=0.01)
#TimeTakenHapi = time.time() - StartTime
#print("The time taken for HAPI is::",TimeTakenHapi)

StartTime = time.time()
Database = ReadData(Molecule, Location="data/")

CrossSectionDoppler =  CalcCrossSection(Database,Temp=TempValue,P = P_Value, Broadening="Self", WN_Grid=WaveNumber, Profile="Doppler", OmegaWing=OmegaWingValue, OmegaWingHW=0.0, NCORES=1)
CrossSectionVoigt =  CalcCrossSection(Database,Temp=TempValue, P = P_Value, Broadening="Air", WN_Grid=WaveNumber, Profile="PseudoVoigt", OmegaWing=OmegaWingValue, OmegaWingHW=0.0, NCORES=1)
TimeTakenHAPILite = time.time() - StartTime


fig, (ax0, ax1) = plt.subplots(figsize=(14,6), ncols=1, nrows=2, sharex=True)
ax0.plot(nu_hapi_Doppler, abs_hapi_Doppler, "k.", label="Old HAPI" )
ax0.plot(WaveNumber, CrossSectionDoppler, "r-", linewidth=1.0, label="HAPI Lite" )
ax0.set_xlabel("WaveNumber (per cm)")
ax0.set_ylabel("Cross-Section (per cm)")
ax0.set_xlim([2089, 2091])
ax0.legend(loc=1)
ax1.plot(nu_hapi_Doppler, abs_hapi_Doppler - CrossSectionDoppler, "r-" )
ax1.set_xlabel("WaveNumber (per cm)")
ax1.set_ylabel("Residual")
ax1.set_xlim([2089, 2091])
plt.suptitle("Doppler   %s (Temp: %s Pressure: %s)" %(Molecule, str(int(TempValue)),str(round(P_Value,3))))
plt.tight_layout()
plt.savefig("BenchmarkTest_%s_Doppler.png" %(Molecule))
#plt.show()
plt.close()


fig, (ax0, ax1) = plt.subplots(figsize=(14,6), ncols=1, nrows=2, sharex=True)
ax0.plot(nu_hapi_Voigt, abs_hapi_Voigt, "k.", label="Old HAPI" )
ax0.plot(WaveNumber, CrossSectionVoigt, "r-",linewidth=1.0, label="HAPI Lite" )
ax0.set_xlabel("WaveNumber (per cm)")
ax0.set_ylabel("Cross-Section (per cm)")
ax0.legend(loc=1)
ax0.set_xlim([2089, 2091])
ax1.plot(nu_hapi_Voigt, abs_hapi_Voigt - CrossSectionVoigt, "ko" )
ax1.set_xlabel("WaveNumber (per cm)")
ax1.set_ylabel("Residual")
ax1.set_xlim([2089, 2091])
plt.suptitle("Voigt ---   %s (Temp: %s Pressure: %s)" %(Molecule, str(int(TempValue)),str(round(P_Value,3))))
plt.tight_layout()
plt.savefig("BenchmarkTest_%s_Voigt.png" %(Molecule))
plt.show()

plt.close()
