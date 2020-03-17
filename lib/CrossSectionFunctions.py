import numpy as np
import glob
from scipy.interpolate import interp1d
import multiprocessing as mp
from bisect import bisect

def GetWaveNumbers(StartWavelength=300, EndWavelength=30000, Resolution=100000):
        '''
        Returns the wavelengths corresponding to the resolution and in units
        of cm.
        '''
        WaveLengthValues = []

        #Converting values to
        StartWavelength = StartWavelength*1.0e-7       #nm to cm
        EndWavelength = EndWavelength*1.0e-7           #nm to cm
        WaveLengthValues = [StartWavelength]

        while WaveLengthValues[-1]<EndWavelength:
            WaveLengthValues.append(WaveLengthValues[-1]+WaveLengthValues[-1]/Resolution)
        WaveLengthValues = np.array(WaveLengthValues)
        WaveNumberRange = 1./WaveLengthValues

        WaveNumberRange = np.array(sorted(WaveNumberRange))
        return WaveLengthValues, WaveNumberRange

def BinModel(nu_HR,abs_HR,nu_Grid):
    '''
    This function takes a cross-section at high resolution:
    nu_HR is the wavenumber in increasing order
    abs_HR is the absorption cross-section in an increasing order
    The stepsize in the WaveNumberGrid is not expected to be the equal
    '''

    InterpValues = np.zeros(len(nu_Grid))
    Start = 0
    i = 0
    while i<len(nu_Grid):
        StartIndex = bisect(nu_HR, Start)
        StopIndex = bisect(nu_HR, nu_Grid[i])
        InterpValues[i] = np.mean(abs_HR[StartIndex:StopIndex])
        Start=nu_Grid[i]
        i+=1
    NanIndex = np.isnan(InterpValues)
    InterpValues[NanIndex] = 0.0
    return InterpValues


def SymplecticInterpolation(nu_HR,abs_HR,nu_Grid):
      '''
      This function takes a cross-section at high resolution:
      nu_HR is the wavenumber in increasing order
      abs_HR is the absorption cross-section in an increasing order
      The stepsize in the WaveNumberGrid is not expected to be the equal
      '''
      #Assert the Wavenumber is strictly increasing
      assert nu_HR[-1]>nu_HR[0], "The high resolution nu should also be strictly increasing."
      assert nu_Grid[-1]>nu_Grid[0], "The low resolution wavenumber should also be strictly increasing."

      InterpolatedValues = np.zeros(len(nu_Grid))
      for i in range(len(nu_Grid)):
          if i+1<len(nu_Grid):
              StepSize= nu_Grid[i+1] - nu_Grid[i]
          SelectIndex = np.abs(nu_HR-nu_Grid[i])<StepSize/2.0
          InterpolatedValues[i] =  np.mean(abs_HR[SelectIndex])
      NanIndex = np.isnan(InterpolatedValues)
      InterpolatedValues[NanIndex] = 0.0
      return InterpolatedValues
