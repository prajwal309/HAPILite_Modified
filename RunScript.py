import numpy as np
import itertools
from lib.CrossSectionFunctions import GetWaveNumbers
import os
from sys import getsizeof
import time

from HAPILite import CalcCrossSection
from lib.ReadComputeFunc import ReadData

#Base Location
BaseLocation = "Mar17_2020"
print("The base location is::", BaseLocation)

if not(os.path.exists(BaseLocation)):
    os.system("mkdir %s" %BaseLocation)
    print("The base location folder created")
else:
    print("The base location folder already exists.")



#Temperature is goi
TemperatureGrid = np.array([100, 110, 120, 130, 140, 160, 180, 200, 230, 260, 290, 330, 370, 410, 460, 510, 580, 650, 730, 810])
PressureGrid = np.array([-5.00, -4.20, -3.80,  -3.50, -3.25, -3.05, -2.95, -2.85, -2.75, -2.65, -2.55, -2.45, -2.30, -2.15, \
                          -2.00, -1.85, -1.70,  -1.55, -1.4,  -1.25, -1.1,  -0.95, -0.80,  -0.70, -0.60,  -0.50,  -0.40, \
                          -0.30, -0.20, -0.10, 0.00, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 1.00,  1.10, 1.20, \
                          1.30, 1.40, 1.50, 1.60, 1.70,  1.80, 1.90, 2.00])

AllName = ["CS_6","CS_7"]
AllName = [BaseLocation+"/"+Item for Item in AllName]

#CS_8 is exomol

AllError = [0,0]
AllBroadener = ["both", "both"]
AllOmegaWidth = [50, 50]

#Note the resolution will be constant
Molecules = ["CO", "H2", "N2", "H2O", "CH4", "CO2", "O3"]


#Generate the cross section:
Resolution = 2000000


#Low resolution wavelength
WavelengthGrid, WaveNumberGrid = GetWaveNumbers(300, 30000, Resolution)

#Save the temperature, pressure and run script
np.savetxt(BaseLocation+"/Temperature.txt", TemperatureGrid)
np.savetxt(BaseLocation+"/Pressure.txt", PressureGrid)
np.savetxt(BaseLocation+"/Wavelength.txt", WavelengthGrid)
np.savetxt(BaseLocation+"/Wavenumber.txt", WaveNumberGrid)



for Name, Error, Broadener, OmegaWidth in zip(AllName, AllError, AllBroadener, AllOmegaWidth):
    #Make the folder
    print("The name of the folder is given by::", Name)
    if not(os.path.exists(Name)):
        os.system("mkdir %s" %Name)
        print("The location of the path is::", Name)
    else:
        print("The folder already exists.")


    for Molecule in Molecules:

        SaveMatrixName = Name+"/"+Molecule+".npy"

        Database = ReadData(Molecule, Location="data/")
        SigmaMatrix = np.zeros((len(TemperatureGrid), len(PressureGrid), len(WaveNumberGrid)),  dtype=np.float32)

        for TCounter, TValue in enumerate(TemperatureGrid):
            print("Percentage Complete", int((TCounter+0.5)/len(TemperatureGrid))*100)
            for PCounter, PValue in enumerate(PressureGrid):
                #check if error are different
                SigmaMatrix[TCounter, PCounter, :]  =  CalcCrossSection(Database, Name, Temp=TValue,P = 10**PValue, \
                          Broadening=Broadener, WN_Grid=WaveNumberGrid, Profile="Voigt",\
                          OmegaWing=0.0, OmegaWingHW=OmegaWidth, NCORES=-1)[::-1]
            

        #Now save the matrix here
        np.save(SaveMatrixName, SigmaMatrix)
