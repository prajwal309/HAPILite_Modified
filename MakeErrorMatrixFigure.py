import numpy as np
import matplotlib.pyplot as plt

import matplotlib as mpl
mpl.rc('font',**{'family':'sans-serif', 'serif':['Computer Modern Serif'],'sans-serif':['Cambria'], 'size':15,'weight':500, 'variant':'normal'})
mpl.rc('axes',**{'labelweight':'normal', 'linewidth':1})
mpl.rc('ytick',**{'major.pad':12, 'color':'k'})
mpl.rc('xtick',**{'major.pad':8,})
mpl.rc('mathtext',**{'default':'regular','fontset':'cm','bf':'monospace:bold'})
mpl.rc('text', **{'usetex':True})
mpl.rc('text.latex',preamble=r'\usepackage{cmbright},\usepackage{relsize},'+r'\usepackage{upgreek}, \usepackage{amsmath}')
mpl.rc('contour', **{'negative_linestyle':'solid'})

TempRange = np.arange(100,901,100)
expPRange = np.arange(-5.,3.1,0.5)
PValue = 10**expPRange
#Load the data

Name4File = "ErrorValue_HW.csv"
ErrorData = np.loadtxt(Name4File, delimiter=",")


fig, ax = plt.subplots(figsize=(16,8))
Image = plt.imshow(ErrorData, cmap="viridis", origin="lower")
plt.xlabel("Pressure (Pa)", fontsize=25)
plt.ylabel("Temperature (K)", fontsize=25)
XTicks = np.arange(0,len(PValue),1)
YTicks = np.arange(0,len(TempRange),1)
plt.xticks(XTicks, expPRange)
plt.yticks(YTicks, TempRange)
CLB = plt.colorbar(Image,  fraction=0.046, pad=0.04)
CLB.set_label("Relative Error", labelpad=-2,rotation=-90)
plt.tick_params(which='both', direction='in')
plt.tight_layout()
plt.savefig(Name4File.replace(".csv", ".png"))
plt.close()
