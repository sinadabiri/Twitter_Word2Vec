import numpy as np
import pickle
import os
import csv
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import sys
import os

with open('TSI_for_NT&RT.pickle', mode='rb') as f:
    traffic_TSI, non_traffic_TSI = pickle.load(f)

f, ax = plt.subplots(1)
plt.rcParams['font.family'] = ['serif'] # default is sans-serif
plt.rcParams['font.serif'] = ['Times New Roman']
plt.rcParams['font.size'] = 12
plt.hist(non_traffic_TSI, bins='auto', histtype='step', color='r', label='NT tweets')
plt.hist(traffic_TSI, bins='auto', histtype='step', color='b', label='TR tweets')
plt.legend(prop={'size': 10})
yLine = np.linspace(0, 85)
threshold = (sum(traffic_TSI) / len(traffic_TSI) + sum(non_traffic_TSI) / len(non_traffic_TSI)) / 2
ave_TSI_NT = sum(non_traffic_TSI) / len(non_traffic_TSI)
ave_TSI_TR = sum(traffic_TSI) / len(traffic_TSI)
std_NT = np.std(np.array(traffic_TSI))
std_RT = np.std(np.array(non_traffic_TSI))
Mu_NT = 50 * [ave_TSI_NT]
Mu_TR = 50 * [ave_TSI_TR]
Threshold = 50 * [threshold]
# Use the parameters of Line2D class
plt.plot(Mu_NT, yLine, linewidth=1, linestyle='dashed', color='r')
plt.plot(Mu_TR, yLine, linewidth=1, linestyle='dashed', color='b')
plt.plot(Threshold, yLine, linewidth=1, linestyle='dashed', color='black')
#plt.axvline(x=.3, color='black')
plt.ylim([0, 100])
plt.ylabel('Frequency')
plt.xlabel('TSI for Tweets')
#plt.title('TI_Ave_Distance')
plt.savefig('Pape2_Histogram.png', dpi=1200)


plt.show()


A = 2