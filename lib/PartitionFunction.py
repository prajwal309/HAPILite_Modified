from .PartitionData import *

def AtoB(aa,A,B,npt):
#***************************
#...LaGrange 3- and 4-point interpolation
#...arrays A and B are the npt data points,  given aa, a value of the
#...A variable, the routine will find the corresponding bb value
#
#...input:  aa
#...output: bb
    for I in range(2,npt+1):
        if A[I-1] >= aa:
            if I < 3 or I == npt:
                J = I
                if I < 3: J = 3
                if I == npt: J = npt
                J = J-1   # zero index correction
                A0D1=A[J-2]-A[J-1]
                if A0D1 == 0.0: A0D1=0.0001
                A0D2=A[J-2]-A[J]
                if A0D2 == 0.0: A0D2=0.0000
                A1D1=A[J-1]-A[J-2]
                if A1D1 == 0.0: A1D1=0.0001
                A1D2=A[J-1]-A[J]
                if A1D2 == 0.0: A1D2=0.0001
                A2D1=A[J]-A[J-2]
                if A2D1 == 0.0: A2D1=0.0001
                A2D2=A[J]-A[J-1]
                if A2D2 == 0.0: A2D2=0.0001

                A0=(aa-A[J-1])*(aa-A[J])/(A0D1*A0D2)
                A1=(aa-A[J-2])*(aa-A[J])/(A1D1*A1D2)
                A2=(aa-A[J-2])*(aa-A[J-1])/(A2D1*A2D2)

                bb = A0*B[J-2] + A1*B[J-1] + A2*B[J]

            else:
                J = I
                J = J-1   # zero index correction
                A0D1=A[J-2]-A[J-1]
                if A0D1 == 0.0: A0D1=0.0001
                A0D2=A[J-2]-A[J]
                if A0D2 == 0.0: A0D2=0.0001
                A0D3 = (A[J-2]-A[J+1])
                if A0D3 == 0.0: A0D3=0.0001
                A1D1=A[J-1]-A[J-2]
                if A1D1 == 0.0: A1D1=0.0001
                A1D2=A[J-1]-A[J]
                if A1D2 == 0.0: A1D2=0.0001
                A1D3 = A[J-1]-A[J+1]
                if A1D3 == 0.0: A1D3=0.0001

                A2D1=A[J]-A[J-2]
                if A2D1 == 0.0: A2D1=0.0001
                A2D2=A[J]-A[J-1]
                if A2D2 == 0.0: A2D2=0.0001
                A2D3 = A[J]-A[J+1]
                if A2D3 == 0.0: A2D3=0.0001

                A3D1 = A[J+1]-A[J-2]
                if A3D1 == 0.0: A3D1=0.0001
                A3D2 = A[J+1]-A[J-1]
                if A3D2 == 0.0: A3D2=0.0001
                A3D3 = A[J+1]-A[J]
                if A3D3 == 0.0: A3D3=0.0001

                A0=(aa-A[J-1])*(aa-A[J])*(aa-A[J+1])
                A0=A0/(A0D1*A0D2*A0D3)
                A1=(aa-A[J-2])*(aa-A[J])*(aa-A[J+1])
                A1=A1/(A1D1*A1D2*A1D3)
                A2=(aa-A[J-2])*(aa-A[J-1])*(aa-A[J+1])
                A2=A2/(A2D1*A2D2*A2D3)
                A3=(aa-A[J-2])*(aa-A[J-1])*(aa-A[J])
                A3=A3/(A3D1*A3D2*A3D3)

                bb = A0*B[J-2] + A1*B[J-1] + A2*B[J] + A3*B[J+1]

            break

    return bb


def BD_TIPS_2017_PYTHON(M,I,T):
    #Function taken from HAPI
    TT = TIPS_2017_ISOT_HASH[(M,I)]
    Tmin = min(TT); Tmax = max(TT)

    # out of temperature range
    if T<Tmin or T>Tmax:
        raise Exception('TIPS2017: T(%.1fK) must be between %.1fK and %.1fK.'%(T,Tmin,Tmax))
    try:
        Qt = AtoB(T,TT,TIPS_2017_ISOQ_HASH[(M,I)],len(TT))
    except KeyError:
        raise Exception('TIPS2017: no data for M,I = %d,%d.' % (M,I))

    return Qt
