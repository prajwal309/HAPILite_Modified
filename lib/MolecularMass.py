import numpy as np


#Data ID (Global ID, local ID), (Abundance, Molar Mass)
DataFromHITRAN = [((1,1),(0.997317,18.010565)), ((1,2),(0.002000,20.014811)), ((1,3),(3.718840e-4,19.01478)),   #H2O
                  ((2,1),(0.984204,43.98983)),  ((2,2),(0.011057,44.993185)), ((2,3),(0.003947,45.994076)),     #CO2
                  ((5,1),(0.986544,27.994915)), ((5,2),(0.011084,28.99827)),  ((5,3),(0.001978,29.999161)),     #CO
                  ((6,1),(0.988274,16.0313)),   ((6,2),(0.011103,17.034655)), ((6,3),(6.157510e-4,17.037475)),  #CH4
                  ((7,1),(0.995262,31.98983)),  ((7,2),(0.003991,33.994076)),((7,3),(7.422350e-4,32.994045)),   #O2
                  ((22,1),(0.992687,28.006148)),((22,2), (0.007478,29.003182)),                                 #N2
                  ((45,1),(0.999688,2.01565)),  ((45,2),(3.114320e-4,3.021825))]                                #H2

def GetMolecularMass(M,I):
    ALLGlobalID = np.array([Item[0][0] for Item in DataFromHITRAN])
    ALLLocalID = np.array([Item[0][1] for Item in DataFromHITRAN])
    AllAbundance = np.array([Item[1][0] for Item in DataFromHITRAN])
    AllMolecularMass = np.array([Item[1][1] for Item in DataFromHITRAN])
    MatchIndex = np.logical_and(M==ALLGlobalID, I==ALLLocalID)
    if np.sum(MatchIndex)<0.1: #Np matches found
        print("You would have to add the data for the molecule individually at MolecularMass file inside lib folder. Currently implemented only for \
        Water, CO2, CO, CH4, O2, N2, H2.")
    return AllMolecularMass[MatchIndex][0]
