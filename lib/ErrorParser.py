import numpy as np


def MapError(ErrorArray):
    '''
    Parameters:
    ==========================================

    #mapping the error
    #0  1 or unreported Unreported or unavailable
    #1  Between 1e−1 and 1e0 Default or constant
    #2  Between 1e−2and 1e−1 Average or estimate
    #3  Between 1e−3and 1e−2 Greater than 20%
    #4  Between 1e−4and 1e−3 Between 10% and 20%
    #5  Between 1e−5and 1e−4 Between 5% and 10%
    #6  Between 1e−6and 1e−5 Between 2% and 5%
    #7  Between 1e−7and 1e−6 Between 1% and 2%
    #8  Between 1e−8and 1e−7 <1%

    #Uncertainty of the line position
    #Uncertainty of the line Intensity
    #Uncertainty of the air-broadened half width
    #Uncertainty of the self-broadened half width
    #Uncertainty of the temperature dependence of air-broadened half width
    #Uncertainty of the line shifts
    '''


    Values = np.ones((len(ErrorArray),6))*np.nan

    RowIndex = 0
    while RowIndex<len(ErrorArray):
        ErrorValue = str(ErrorArray[RowIndex])
        Position = 0
        while(Position<6):
            try:
                Item = int(ErrorValue[Position:Position+1])
            except:
                pass
            if Position==0:
                #Case of line center
                if Item==0:
                    Values[RowIndex,Position] = 1.e0
                elif Item==1:
                    Values[RowIndex,Position] = 1.e0
                elif Item==2:
                    Values[RowIndex,Position] = 1.e-1
                elif Item==3:
                    Values[RowIndex,Position] = 1.e-2
                elif Item==4:
                    Values[RowIndex,Position] = 1.e-3
                elif Item==5:
                    Values[RowIndex,Position] = 1.e-4
                elif Item==6:
                    Values[RowIndex,Position] = 1.e-5
                elif Item==7:
                    Values[RowIndex,Position] = 1.e-6
                elif Item==8:
                    Values[RowIndex,Position] = 1.e-7
                else:
                    pass
            else:
                #Case for other parameters
                #0  1 or unreported Unreported or unavailable
                #1  Between 1e−1 and 1e0 Default or constant
                #2  Between 1e−2and 1e−1 Average or estimate
                #3  Between 1e−3and 1e−2 Greater than 20%
                #4  Between 1e−4and 1e−3 Between 10% and 20%
                #5  Between 1e−5and 1e−4 Between 5% and 10%
                #6  Between 1e−6and 1e−5 Between 2% and 5%
                #7  Between 1e−7and 1e−6 Between 1% and 2%
                #8  Between 1e−8and 1e−7 <1%
                if Item==0:
                    Values[RowIndex,Position] = 0.20
                elif Item==1:
                    Values[RowIndex,Position] = 1.0
                elif Item==2:
                    Values[RowIndex,Position] = 0.50
                elif Item==3:
                    Values[RowIndex,Position] = 0.25
                elif Item==4:
                    Values[RowIndex,Position] = 0.20
                elif Item==5:
                    Values[RowIndex,Position] = 0.10
                elif Item==6:
                    Values[RowIndex,Position] = 0.050
                elif Item==7:
                    Values[RowIndex,Position] = 0.020
                elif Item==8:
                    Values[RowIndex,Position] = 0.010
            Position+=1
        RowIndex+=1
    return Values
