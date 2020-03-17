"""
Test to test truncation error and
"""
import numpy as np
import time
import matplotlib.pyplot as plt
from HAPILite import CalcCrossSection, CalcCrossSectionWithError
from lib.ReadComputeFunc import ReadData
from lib.PartitionFunction import BD_TIPS_2017_PYTHON


from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.ticker import FormatStrFormatter

Molecule = "CO"
TempValue = 300.0
Tref=296.0
P_Value = 1.0

OmegaWingValue = 100.0
OmegaRangeValue = [1./30000.*1e7, 1./300.*1e7]

WaveNumber = np.arange(OmegaRangeValue[0], OmegaRangeValue[1]+0.01, 0.001)


Database = ReadData(Molecule, Location="data/")
MoleculeNumberDB, IsoNumberDB, LineCenterDB, LineIntensityDB, \
LowerStateEnergyDB, GammaSelf, TempRatioPower, ErrorArray = Database

SelectIndex = np.logical_and(LineCenterDB>OmegaRangeValue[0], LineCenterDB<OmegaRangeValue[1])

LineCenterDB = LineCenterDB[SelectIndex]
LineIntensityDB = LineIntensityDB[SelectIndex]
LowerStateEnergyDB = LowerStateEnergyDB[SelectIndex]

const_R = 1.4388028496642257 #Radiation Constant


TempRange = np.arange(100,901,100)
expPRange = np.arange(-5.,3.1,0.5)

ErrorMatrix = np.zeros((len(TempRange), len(expPRange)))

for TCounter, TempValue in enumerate(TempRange):
    for PCounter, expP in enumerate(expPRange):
        PValue = 10**expP

        ch = np.exp(-const_R*LowerStateEnergyDB/TempValue)*(1-np.exp(-const_R*LineCenterDB/TempValue))
        zn = np.exp(-const_R*LowerStateEnergyDB/Tref)*(1-np.exp(-const_R*LineCenterDB/Tref))
        SigmaT = BD_TIPS_2017_PYTHON(MoleculeNumberDB,IsoNumberDB,TempValue)
        SigmaTref = BD_TIPS_2017_PYTHON(MoleculeNumberDB,IsoNumberDB,Tref)

        LineIntensityScaled = LineIntensityDB*SigmaTref/SigmaT*ch/zn

        TotalIntensity = np.sum(LineIntensityScaled)



        CrossSection =  CalcCrossSection(Database,Temp=TempValue, P = PValue,\
                         WN_Grid=WaveNumber, Profile="Voigt", OmegaWing=OmegaWingValue,\
                         OmegaWingHW=100.0, NCORES=-1)


        #Generate the area
        Area = np.trapz(CrossSection, WaveNumber)

        #Relative error
        RelativeError =  (TotalIntensity - Area)/TotalIntensity*100.0

        print(TempValue, PValue, "::", RelativeError)

        ErrorMatrix[TCounter, PCounter] = RelativeError

np.savetxt("ErrorValue_HW_HS.csv", ErrorMatrix, delimiter=",")
