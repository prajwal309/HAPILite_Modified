#import the libraries
import matplotlib.pyplot as plt

#This code uses
import numpy as np
from time import time
import os
from HAPILite import CalcCrossSection, CalcCrossSectionWithError
from lib.ReadComputeFunc import ReadData


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
MoleculeList = Values[12].split(",")                                            #Get the list of Molecular species
Cores = int(Values[13])
Error = Values[14].replace("\t","")

SaveFolder = "DataMatrix"+Error.replace("-","Neg").replace(" ","")
MoleculeList = [Item.replace(" ", "").replace("\t","") for Item in MoleculeList]

TempRange = np.arange(TempStart,TempStop+TempStep, TempStep)                    #Temperature in K
expP_Range = np.arange(expP_Start, expP_Stop-expP_Step, -expP_Step)             #Pressure in log(P) atm

WaveNumberStart = 1./(HighWavelength*1.e-7)            #in per cm
WaveNumberStop= 1./(LowWavelength*1.e-7)               #in per cm
WaveNumberRange = np.arange(WaveNumberStart, WaveNumberStop, WN_Resolution)



for Molecule in MoleculeList:
    print("\n\n Starting Molecule::", Molecule)
    StartTime = time()
    Database = ReadData(Molecule, Location="data/")

    #initiate the saving matrix for each case
    SigmaMatrix = np.zeros((len(TempRange),len(expP_Range),len(WaveNumberRange)),dtype=np.float32)
    print("The shape of the Sigma Matrix is given by:", np.shape(SigmaMatrix))

    for TempCount, TempValue in enumerate(TempRange):
            for PCount, expPValue in enumerate(expP_Range):

                P_Value = 10**expPValue
                print("-"*15)
                print("Temperature:", TempValue)
                print("Pressure:", P_Value)

                if "0SIG" in Error.upper():
                    SigmaMatrix[TempCount, PCount, :] = CalcCrossSection(Database, Temp=TempValue, P = P_Value, WN_Grid=WaveNumberRange,   \
                                                        Profile=LineShapeProfile, OmegaWing=100.0, OmegaWingHW=0.0, NCORES=Cores)

                elif "1SIG" in Error.upper() or "2SIG" in Error.upper():
                    SigmaMatrix[TempCount, PCount, :] = CalcCrossSectionWithError(Database, Temp=TempValue, P = P_Value, WN_Grid=WaveNumberRange,   \
                                                        Profile=LineShapeProfile, OmegaWing=100.0, OmegaWingHW=0.0, NCORES=Cores, Err=Error)

                else:
                    raise Exception("Error in GenerateMatrix. The error has to be 0SIG, 1SIG/-1SIG or 2SIG/-2SIG ")

    #Now save the file


    np.save(SaveFolder+"/"+Molecule+".npy", SigmaMatrix)
    print("For %s, time taken is %5.2f" %(Molecule, time() -StartTime))
