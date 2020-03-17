import glob
import numpy as np
from lib.CrossSectionFunctions import GetWaveNumbers, BinModel
import matplotlib.pyplot as plt
import os
import itertools
import time
import multiprocessing as mp

#parse the parameters.ini which contains the information
Data = [f.split(":") for f in open("CrossSectionParams/Parameters.ini",'r+')][1:]
Values = [Item[1].split("#")[0] for Item in Data]


#Load the parameters for creating
TempStart = float(Values[0])    #Step size of the temperature
TempStop = float(Values[1])     #Step size of the temperature
TempStep = float(Values[2])     #Step size of the temperature

expP_Start = float(Values[3])   #The largest log10(pressure) in atm
expP_Stop = float(Values[4])    #The smallest log10(pressure) in atm
expP_Step = float(Values[5])    #Step size of the pressure


Broadener = Values[6].replace(" ","")                                           #Broadening either self or air at this point
OmegaWidth = float(Values[7])                                                   #Consider the omegawidth -- how far the lines have to be considered
LowWavelength = float(Values[8])                                                #Shortest Wavelength coverage range
HighWavelength = float(Values[9])                                               #Longest Wavelength coverage range
WN_Resolution = float(Values[10])                                               #Resolution of the Wave Number
LineShapeProfile = Values[11].replace(" ","")                                   #Voigt profile by default
MoleculeList = Values[12].split(",")                                             #Get the list of Molecular species
Cores = int(Values[13])
Error = Values[14].replace("\t","")



TempRange = np.arange(TempStart,TempStop+TempStep, TempStep)                    #Temperature in K
expP_Range = np.arange(expP_Start, expP_Stop-expP_Step, -expP_Step)             #Pressure in log(P) atm


#Define the wavenumber range values...
WaveNumberStart = 1./(HighWavelength*1.e-7)            #in per cm
WaveNumberStop= 1./(LowWavelength*1.e-7)               #in per cm
WaveNumberRange = np.arange(WaveNumberStart, WaveNumberStop, WN_Resolution)
#Plotting in the ascending order
WaveLengthRange = 1./WaveNumberRange
WaveLengthRange = WaveLengthRange[::-1]



#Now get the assign the resolution values
Resolution = 10000

#Low resolution wavelength
Wavelength_LR, WaveNumber_LR = GetWaveNumbers(LowWavelength, HighWavelength, Resolution)


Folder2Save = "R1SIG"+str(Resolution)

if not(os.path.exists(Folder2Save)):
    os.system("mkdir %s" %(Folder2Save))

np.savetxt(Folder2Save+"/Temperature.txt", TempRange, delimiter=",")
np.savetxt(Folder2Save+"/exp_Pressure.txt", expP_Range, delimiter=",")
np.savetxt(Folder2Save+"/WaveLength.txt", Wavelength_LR, delimiter=",")
np.savetxt(Folder2Save+"/Molecules.txt", MoleculeList, delimiter=",", fmt='%s')

BaseLocation = "DataMatrix1SIG/"
MoleculesFiles = glob.glob(BaseLocation+"*.npy")
NumMolecules = len(MoleculesFiles)
NumTempValues = len(TempRange)
NumPValues = len(expP_Range)
NumWL_Values = len(Wavelength_LR)

#Initiate a database matrix
DatabaseMatrix = np.ones((NumTempValues, NumPValues, NumMolecules, NumWL_Values), dtype=np.float32)

StartTime = time.time()
for MoleculeCount, Molecule in enumerate(MoleculeList):
    #Read the molecule name
    MoleculeLocation = BaseLocation+Molecule+".npy"
    TP_Counter = list(itertools.product(range(len(TempRange)),range(len(expP_Range))))

    #Using multiprocessing
    SigmaMatrix = np.load(MoleculeLocation,mmap_mode='r')

    print("The molecule is given by::", Molecule, ".   Now loading the data....")
    print("The location of the molecule is given by::", MoleculeLocation)

    while(len(TP_Counter)>0):
        NUMCORES = min([mp.cpu_count(), len(TP_Counter)])
        CPU_Pool = mp.Pool(NUMCORES)
        Tasks = []
        TempCounterValues = []
        PCounterValues = []
        for i in range(NUMCORES):

            Item = TP_Counter[0]
            TempCounter, PCounter = Item
            TempCounterValues.append(TempCounter)
            PCounterValues.append(PCounter)
            TP_Counter.pop(0)

            #High resolution cross-section data
            Sigma_HR = SigmaMatrix[TempCounter, PCounter, :]
            Tasks.append(CPU_Pool.apply_async(BinModel, (WaveLengthRange, Sigma_HR, Wavelength_LR)))

        CPU_Pool.close()
        CPU_Pool.join()

        for Count, task in enumerate(Tasks):
            Result=task.get()
            #Assign the value to the datamatrix
            DatabaseMatrix[TempCounterValues[Count], PCounterValues[Count], MoleculeCount,:] = Result



#Now save the datamatrix
print("Time taken::", time.time() - StartTime)
np.save(Folder2Save+"/DataBase_%d.npy" %Resolution, DatabaseMatrix)
