# -- coding: utf-8 --
import seaborn as sn
import matplotlib.pyplot as plt

# -- coding: utf-8 --

import  matplotlib.pyplot as plt
import numpy as np

# -*- coding: utf-8 -*-
"""
Created on Thu Apr 19 11:09:06 2018

@author: butany
"""

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from matplotlib.ticker import MaxNLocator

a=np.divide(np.array([0]),np.array([0]))
rmse = np.nan_to_num(a)
print(rmse)

HA_mae =[6.185242 ,6.185242 ,6.185242 ,6.185242 ,6.185242 ,6.185242]
ARIMA_mae =[5.883667 ,6.046000 ,6.104512 ,6.374540 ,6.360256 ,6.168883]
LSTM_mae =[5.517776 ,5.632012 ,5.777946 ,5.946635 ,6.041769 ,6.079495]
Bi_LSTM_mae =[5.524607 ,5.632932 ,5.771882 ,5.954062 ,6.055177 ,6.096319]
FI_RNNs_mae =[5.520539 ,5.626097 ,5.768340 ,5.932466 ,6.028442 ,6.048296]
PSPNN_mae =[5.386255 ,5.468801 ,5.567571 ,5.716054 ,5.768782 ,5.749586]
MDL_mae =[5.429035 ,5.509909 ,5.627575 ,5.784230 ,5.813673 ,5.804645]
T_GCN_mae =[5.670292 ,5.675068 ,5.746703 ,5.863256 ,5.888195 ,5.831183]
AST_GAT_mae =[5.248924 ,5.258951 ,5.329265 ,5.428566 ,5.421416 ,5.314834]
GMAN_mae =[5.233438 ,5.208744 ,5.256062 ,5.329036 ,5.347667 ,5.204790]
STGIN_mae =[5.175827 ,5.100160 ,5.149714 ,5.222380 ,5.245451 ,5.150738]

HA_rmse = [10.139710,10.139710,10.139710,10.139710,10.139710,10.139710]
ARIMA_rmse =[9.429440 ,9.798609 ,9.769539 ,10.209594,10.030069,9.769304]
LSTM_rmse =[9.125115 ,9.284576 ,9.433199 ,9.641906 ,9.626469 ,9.586535]
Bi_LSTM_rmse =[9.129195 ,9.285857 ,9.423597 ,9.640355 ,9.636107 ,9.591586]
FI_RNNs_rmse =[9.116093 ,9.267610 ,9.432268 ,9.622239 ,9.603523 ,9.569430]
PSPNN_rmse =[8.880815 ,9.022358 ,9.145341 ,9.307195 ,9.271415 ,9.167632]
MDL_rmse =[8.916788 ,9.076988 ,9.176666 ,9.381263 ,9.304234 ,9.256170]
T_GCN_rmse =[9.134161 ,9.173527 ,9.245162 ,9.387154 ,9.317363 ,9.195807]
AST_GAT_rmse =[8.705578 ,8.784430 ,8.828416 ,8.943468 ,8.809244 ,8.712925]
GMAN_rmse =[8.730351 ,8.795372 ,8.797104 ,8.908348 ,8.818618 ,8.598270]
STGIN_rmse =[8.644908 ,8.640150 ,8.701153 ,8.819650 ,8.724352 ,8.527315]


HA_mape = [0.126192 ,0.126192 ,0.126192 ,0.126192 ,0.126192 ,0.126192]
ARIMA_mape =[0.102238 ,0.159162 ,0.130914 ,0.110670 ,0.129478 ,0.131120]
LSTM_mape =[0.133428 ,0.140966 ,0.126444 ,0.119835 ,0.122028 ,0.134512]
Bi_LSTM_mape =[0.132028 ,0.140429 ,0.127086 ,0.122740 ,0.127745 ,0.141035]
FI_RNNs_mape =[0.137907 ,0.144544 ,0.130363 ,0.123199 ,0.129686 ,0.140039]
PSPNN_mape =[0.130217 ,0.135699 ,0.120082 ,0.114041 ,0.116743 ,0.128589]
MDL_mape =[0.130555 ,0.134097 ,0.120055 ,0.115990 ,0.117240 ,0.129795]
T_GCN_mape =[0.138232 ,0.141509 ,0.124293 ,0.118078 ,0.119514 ,0.132048]
AST_GAT_mape =[0.124102 ,0.128212 ,0.114900 ,0.110572 ,0.110653 ,0.121733]
GMAN_mape =[0.130036 ,0.131432 ,0.116220 ,0.107743 ,0.111887 ,0.117915]
STGIN_mape =[0.126175 ,0.127834 ,0.112100 ,0.104818 ,0.108805 ,0.117945]

font2 = {'family' : 'Times New Roman',
'weight' : 'normal',
'size'   : 12.,
}
plt.ylabel('Loss(ug/m3)',font2)

font1 = {'family' : 'Times New Roman',
'weight' : 'normal',
'size'   : 12,
}


LSTM = pd.read_csv('results/LSTM.csv',encoding='utf-8').values[108:]
Bi_LSTM = pd.read_csv('results/BILSTM.csv',encoding='utf-8').values[108:]
FI_RNNs = pd.read_csv('results/FI-RNN.csv',encoding='utf-8').values[108:]
GMAN = pd.read_csv('results/GMAN.csv',encoding='utf-8').values[108:]
STGIN = pd.read_csv('results/STGIN.csv',encoding='utf-8').values
PSPNN = pd.read_csv('results/PSPNN.csv',encoding='utf-8').values[108:]
MDL = pd.read_csv('results/MDL.csv',encoding='utf-8').values[108:]
AST_GAT=pd.read_csv('results/AST-GAT.csv',encoding='utf-8').values[108:]
T_GCN=pd.read_csv('results/T-GCN.csv',encoding='utf-8').values[108:]

LSTM_pre = []
LSTM_obs = []
Bi_LSTM_pre = []
Bi_LSTM_obs = []
FI_RNNs_pre = []
FI_RNNs_obs = []
GMAN_pre = []
GMAN_obs = []
STGIN_pre = []
STGIN_obs = []
PSPNN_pre = []
PSPNN_obs = []
MDL_pre = []
MDL_obs = []
AST_GAT_pre = []
AST_GAT_obs = []
T_GCN_pre = []
T_GCN_obs = []

K = 10
site_num=108
for i in range(site_num,site_num*K,site_num):
    LSTM_obs.append(LSTM[i:i+site_num,19:25])
    LSTM_pre.append(LSTM[i:i + site_num, 25:])

for i in range(site_num,site_num*K,site_num):
    Bi_LSTM_obs.append(Bi_LSTM[i:i+site_num,19:25])
    Bi_LSTM_pre.append(Bi_LSTM[i:i + site_num, 25:])

for i in range(site_num,site_num*K,site_num):
    FI_RNNs_obs.append(FI_RNNs[i:i+site_num,19:25])
    FI_RNNs_pre.append(FI_RNNs[i:i + site_num, 25:])

for i in range(site_num,site_num*K,site_num):
    GMAN_obs.append(GMAN[i:i+site_num,19:25])
    GMAN_pre.append(GMAN[i:i + site_num, 25:])

for i in range(site_num,site_num*K,site_num):
    STGIN_obs.append(STGIN[i:i+site_num,19:25])
    STGIN_pre.append(STGIN[i:i + site_num, 25:])

for i in range(site_num,site_num*K,site_num):
    PSPNN_obs.append(PSPNN[i:i+site_num,19:25])
    PSPNN_pre.append(PSPNN[i:i + site_num, 25:])

for i in range(site_num,site_num*K,site_num):
    MDL_obs.append(MDL[i:i+site_num,19:25])
    MDL_pre.append(MDL[i:i + site_num, 25:])

for i in range(site_num,site_num*K,site_num):
    AST_GAT_obs.append(AST_GAT[i:i+site_num,19:25])
    AST_GAT_pre.append(AST_GAT[i:i + site_num, 25:])

for i in range(site_num,site_num*K,site_num):
    T_GCN_obs.append(T_GCN[i:i+site_num,19:25])
    T_GCN_pre.append(T_GCN[i:i + site_num, 25:])

'''
plt.subplot(3, 1, 1)
i,j=8,4
print(STGIN_obs[i][j])
plt.xticks(range(1, 7), ['2021.8.14 6:45', '7:00', '7:15', '7:30', '7:45', '8:00'])
plt.plot(range(1, 7), STGIN_obs[i][j], marker='d', color='blue', label=u'Observed', linewidth=1)
plt.plot(range(1, 7), STGIN_pre[i][j], marker='X', color='#a55af4', label=u'STGIN', linewidth=1)
plt.plot(range(1, 7), MDL_pre[i][j], marker='p', color='#f504c9', label=u'MDL', linewidth=1)
plt.plot(range(1, 7), GMAN_pre[i][j], marker='^', color='#d0c101', label=u'GMAN', linewidth=1)
plt.plot(range(1, 7), AST_GAT_pre[i][j], marker='*', color='#82cafc', label=u'AST-GAT', linewidth=1)
plt.plot(range(1, 7), PSPNN_pre[i][j], marker='s', color='#0cdc73', label=u'PSPNN', linewidth=1)
plt.plot(range(1, 7), T_GCN_pre[i][j], marker='d', color='#ff5b00', label=u'T-GCN', linewidth=1)
plt.legend(loc='upper left', prop=font2)
# plt.xlabel("Target time steps", font2)
plt.ylabel("Taffic speed", font1)
plt.title("Road segment 1", font1)

i,j=8,10
print(STGIN_obs[i][j])
plt.subplot(3, 1, 2)
# i,j=1,0
plt.xticks(range(1, 7), ['2021.8.14 6:45', '7:00', '7:15', '7:30', '7:45', '8:00'])
plt.plot(range(1, 7), STGIN_obs[i][j], marker='d', color='blue', label=u'Observed', linewidth=1)
plt.plot(range(1, 7), STGIN_pre[i][j], marker='X', color='#a55af4', label=u'STGIN', linewidth=1)
plt.plot(range(1, 7), MDL_pre[i][j], marker='p', color='#f504c9', label=u'MDL', linewidth=1)
plt.plot(range(1, 7), GMAN_pre[i][j], marker='^', color='#d0c101', label=u'GMAN', linewidth=1)
plt.plot(range(1, 7), PSPNN_pre[i][j], marker='s', color='#0cdc73', label=u'PSPNN', linewidth=1)
plt.plot(range(1, 7), T_GCN_pre[i][j], marker='d', color='#ff5b00', label=u'T-GCN', linewidth=1)
plt.plot(range(1, 7), AST_GAT_pre[i][j], marker='*', color='#82cafc', label=u'AST-GAT', linewidth=1)
# plt.legend(loc='upper left', prop=font2)
# plt.xlabel("Target time steps", font2)
plt.ylabel("Taffic speed", font1)
plt.title("Road segment 2", font1)

i,j=8,97
print(STGIN_obs[i][j])
plt.subplot(3, 1, 3)
# i,j=1,0
plt.xticks(range(1, 7), ['2021.8.14 6:45', '7:00', '7:15', '7:30', '7:45', '8:00'])
plt.plot(range(1, 7), STGIN_obs[i][j], marker='d', color='blue', label=u'Observed', linewidth=1)
plt.plot(range(1, 7), STGIN_pre[i][j], marker='X', color='#a55af4', label=u'STGIN', linewidth=1)
plt.plot(range(1, 7), MDL_pre[i][j], marker='p', color='#f504c9', label=u'MDL', linewidth=1)
plt.plot(range(1, 7), GMAN_pre[i][j], marker='^', color='#d0c101', label=u'GMAN', linewidth=1)
plt.plot(range(1, 7), AST_GAT_pre[i][j], marker='*', color='#82cafc', label=u'AST-GAT', linewidth=1)
plt.plot(range(1, 7), PSPNN_pre[i][j], marker='s', color='#0cdc73', label=u'PSPNN', linewidth=1)
plt.plot(range(1, 7), T_GCN_pre[i][j], marker='d', color='#ff5b00', label=u'T-GCN', linewidth=1)
# plt.legend(loc='upper left', prop=font2)
# plt.xlabel("Target time steps", font2)
plt.ylabel("Taffic speed", font1)
plt.title("Road segment 3", font1)

plt.show()
'''


'''
for i in range(8, len(STGIN_pre)):
    for j in range(108):
        print(i, j)
        # plt.figure()
        plt.subplot(1,1,1)
        # i,j=1,0
        plt.xticks(range(1,7), ['2021.8.26 7:50','7:55','8:00','8:05','8:10','8:15'])
        plt.plot(range(1,7),STGIN_obs[i][j],marker='d',color= 'black', label=u'Observed value', linewidth=1)
        plt.plot(range(1,7),STGIN_pre[i][j],marker='o', color= 'red', label=u'STGIN', linewidth=1)
        plt.plot(range(1,7),GMAN_pre[i][j],marker='s', color= '#d0c101', label=u'GMAN', linewidth=1)
        plt.plot(range(1,7),AST_GAT_pre[i][j],marker='s', color= '#82cafc', label=u'AST_GAT', linewidth=1)
        plt.plot(range(1, 7), PSPNN_pre[i][j], marker='s', color='#0cdc73', label=u'PSPNN', linewidth=1)
        plt.plot(range(1,7),T_GCN_pre[i][j],marker='s', color= 'blue', label=u'T_GCN', linewidth=1)
        plt.legend(loc='upper left',prop=font2)
        # plt.xlabel("Target time steps", font2)
        plt.ylabel("Taffic speed", font2)
        # plt.title("Entrance toll dataset (sample 1)", font2)


        # plt.subplot(2,1,2)
        # i,j=10,16
        # plt.xticks(range(1,7), ['2021.8.26 7:50','7:55','8:00','8:05','8:10','8:15'])
        # plt.plot(range(1,7),STGIN_obs[i][j],marker='d',color= 'black', label=u'Observed value', linewidth=1)
        # plt.plot(range(1,7),STGIN_pre[i][j],marker='o', color= 'orange', label=u'STGIN', linewidth=1)
        # plt.plot(range(1,7),GMAN_pre[i][j],marker='s', color= '#0cdc73', label=u'GMAN', linewidth=1)
        # plt.plot(range(1,7),AST_GAT_pre[i][j],marker='s', color= '#0cdc73', label=u'GMAN', linewidth=1)
        # plt.plot(range(1,7),T_GCN_pre[i][j],marker='s', color= '#0cdc73', label=u'GMAN', linewidth=1)
        # plt.legend(loc='upper left',prop=font2)
        # plt.xlabel("Target time steps", font2)
        # plt.ylabel("Taffic speed", font2)
        # # plt.title("Gantry dataset (sample 2)", font2)
        plt.show()
'''
#
# plt.subplot(6,1,2)
# i,j=10,3
# plt.xticks(range(1,7), ['7:50','7:55','8:00','8:05','8:10','8:15'])
# plt.plot(range(1,7),mtstnet_obs_2[i][j],marker='d',color= 'black', label=u'Observed value', linewidth=1)
# plt.plot(range(1,7),mtstnet_pre_2[i][j],marker='o', color= 'orange', label=u'MT-STNet', linewidth=1)
# plt.plot(range(1,7),gman_pre_2[i][j],marker='s', color= '#0cdc73', label=u'GMAN', linewidth=1)
# plt.legend(loc='upper left',prop=font2)
# # plt.xlabel("Target time steps", font2)
# # plt.ylabel("Taffic flow", font2)
# plt.title("Exit toll dataset (sample 1)", font2)
#
# plt.subplot(6,1,3)
# i,j=10,39
# plt.xticks(range(1,7), ['7:50','7:55','8:00','8:05','8:10','8:15'])
# plt.plot(range(1,7),mtstnet_obs_3[i][j],marker='d',color= 'black', label=u'Observed value', linewidth=1)
# plt.plot(range(1,7),mtstnet_pre_3[i][j],marker='o', color= 'orange', label=u'MT-STNet', linewidth=1)
# plt.plot(range(1,7),gman_pre_3[i][j],marker='s', color= '#0cdc73', label=u'GMAN', linewidth=1)
# plt.legend(loc='upper left',prop=font2)
# # plt.xlabel("Target time steps", font2)
# # plt.ylabel("Taffic flow", font2)
# plt.title("Gantry dataset (sample 1)", font2)

# plt.subplot(6,1,4)
# i,j=10,4
# plt.gca().yaxis.set_major_locator(MaxNLocator(integer=True))
# plt.xticks(range(1,7), ['2021.8.26 7:50','7:55','8:00','8:05','8:10','8:15'])
# plt.plot(range(1,7),STGIN[i][j],marker='d',color= 'black', label=u'Observed value', linewidth=1)
# plt.plot(range(1,7),mtstnet_pre_1[i][j],marker='o', color= 'orange', label=u'MT-STNet', linewidth=1)
# plt.plot(range(1,7),gman_pre_1[i][j],marker='s', color= '#0cdc73', label=u'GMAN', linewidth=1)
# plt.legend(loc='upper left',prop=font2)
# plt.xlabel("Target time steps", font2)
# plt.ylabel("Taffic flow", font2)
# plt.title("Entrance toll dataset (sample 2)", font2)

# plt.subplot(6,1,5)
# i,j=10,6
# plt.gca().yaxis.set_major_locator(MaxNLocator(integer=True))
# plt.xticks(range(1,7), ['7:50','7:55','8:00','8:05','8:10','8:15'])
# plt.plot(range(1,7),mtstnet_obs_2[i][j],marker='d',color= 'black', label=u'Observed value', linewidth=1)
# plt.plot(range(1,7),mtstnet_pre_2[i][j],marker='o', color= 'orange', label=u'MT-STNet', linewidth=1)
# plt.plot(range(1,7),gman_pre_2[i][j],marker='s', color= '#0cdc73', label=u'GMAN', linewidth=1)
# plt.legend(loc='upper left',prop=font2)
# plt.xlabel("Target time steps", font2)
# # plt.ylabel("Taffic flow", font2)
# plt.title("Exit toll dataset (sample 2)", font2)



# y=x的拟合可视化图
# '''
# plt.figure()
plt.subplot(3,3,1)
plt.scatter(LSTM_obs,LSTM_pre,alpha=0.7,color='dimgray',edgecolor = "black",marker='o',label=u'LSTM',linewidths=1)
a=[i for i in range(150)]
b=[i for i in range(150)]
plt.plot(a,b,'black',linewidth=2)
#plt.scatter(a, b)
#plt.plot(test_y_out,'r*:',label=u'predicted value')
# plt.title("Entrance tall dataset", font2)
#设置横纵坐标的名称以及对应字体格式
# plt.xlabel("Observed PM2.5 (ug/m3)", font2)
plt.ylabel("Predicted traffic spedd", font2)
plt.legend(loc='upper left',prop=font2)

plt.subplot(3,3,2)
plt.scatter(Bi_LSTM_obs,Bi_LSTM_pre,alpha=0.7,color='dimgray',edgecolor = "black",marker='o',label=u'Bi-LSTM',linewidths=1)
a=[i for i in range(150)]
b=[i for i in range(150)]
plt.plot(a,b,'black',linewidth=2)
#plt.scatter(a, b)
#plt.plot(test_y_out,'r*:',label=u'predicted value')
# plt.title("Entrance tall dataset", font2)
#设置横纵坐标的名称以及对应字体格式
# plt.xlabel("Observed PM2.5 (ug/m3)", font2)
# plt.ylabel("Predicted traffic spedd", font2)
plt.legend(loc='upper left',prop=font2)

plt.subplot(3,3,3)
plt.scatter(FI_RNNs_obs,FI_RNNs_pre,alpha=0.7,color='dimgray',edgecolor = "black",marker='o',label=u'FI-RNNs',linewidths=1)
a=[i for i in range(150)]
b=[i for i in range(150)]
plt.plot(a,b,'black',linewidth=2)
#plt.scatter(a, b)
#plt.plot(test_y_out,'r*:',label=u'predicted value')
# plt.title("Entrance tall dataset", font2)
#设置横纵坐标的名称以及对应字体格式
# plt.xlabel("Observed PM2.5 (ug/m3)", font2)
# plt.ylabel("Predicted traffic spedd", font2)
plt.legend(loc='upper left',prop=font2)



plt.subplot(3,3,4)
plt.scatter(PSPNN_obs,PSPNN_pre,alpha=0.7,color='dimgray',edgecolor = "black",marker='o',label=u'PSPNN',linewidths=1)
a=[i for i in range(150)]
b=[i for i in range(150)]
plt.plot(a,b,'black',linewidth=2)
#plt.scatter(a, b)
#plt.plot(test_y_out,'r*:',label=u'predicted value')
# plt.title("Entrance tall dataset", font2)
#设置横纵坐标的名称以及对应字体格式
# plt.xlabel("Observed PM2.5 (ug/m3)", font2)
plt.ylabel("Predicted traffic spedd", font2)
plt.legend(loc='upper left',prop=font2)

plt.subplot(3,3,5)
plt.scatter(MDL_obs,MDL_pre,alpha=0.7,color='dimgray',edgecolor = "black",marker='o',label=u'MDL',linewidths=1)
plt.plot(a,b,'black',linewidth=2)
#plt.scatter(a, b)
#plt.plot(test_y_out,'r*:',label=u'predicted value')
# plt.title("Exit tall dataset", font2)
# plt.xlabel("Observed PM2.5 (μg/m3)", font2)
# plt.ylabel("Predicted PM2.5 (μg/m3)", font2)
plt.legend(loc='upper left',prop=font2)

plt.subplot(3,3,6)
plt.scatter(T_GCN_obs,T_GCN_pre,alpha=0.7,color='dimgray',edgecolor = "black",marker='o',label=u'T-GCN',linewidths=1)
c=[i for i in range(150)]
d=[i for i in range(150)]
plt.plot(c,d,'black',linewidth=2)
# plt.title("Gantry dataset", font2)
#设置横纵坐标的名称以及对应字体格式
# plt.xlabel("Observed PM2.5 (μg/m3)", font2)
# plt.ylabel("Predicted PM2.5 (μg/m3)", font2)
plt.legend(loc='upper left',prop=font2)

plt.subplot(3,3,7)
plt.scatter(AST_GAT_obs,AST_GAT_pre,alpha=0.7,color='dimgray',edgecolor = "black",marker='o',label=u'AST-GAT',linewidths=1)
plt.plot(a,b,'black',linewidth=2)
#设置横纵坐标的名称以及对应字体格式
plt.xlabel("Observed traffic speed", font2)
plt.ylabel("Predicted traffic speed", font2)
plt.legend(loc='upper left',prop=font2)

plt.subplot(3,3,8)
plt.scatter(GMAN_obs,GMAN_pre,alpha=0.7,color='dimgray',edgecolor = "black",marker='o',label=u'GMAN',linewidths=1)
plt.plot(a,b,'black',linewidth=2)
#设置横纵坐标的名称以及对应字体格式
plt.xlabel("Observed traffic speed", font2)
# plt.ylabel("Predicted PM2.5 (μg/m3)", font2)
plt.legend(loc='upper left',prop=font2)

plt.subplot(3,3,9)
plt.scatter(STGIN_obs,STGIN_pre,alpha=0.7,color='dimgray',edgecolor = "black",marker='o',label=u'STGIN',linewidths=1)
plt.plot(c,d,'black',linewidth=2)
#设置横纵坐标的名称以及对应字体格式
plt.xlabel("Observed traffic speed", font2)
# plt.ylabel("Predicted PM2.5 (μg/m3)", font2)
plt.legend(loc='upper left',prop=font2)
plt.show()
# '''



# 可视化每个模型在MAPE上的一个表现，柱状图
'''
x = np.arange(1, 8, 1)
total_width, n = 0.8, 3
width = total_width / n
x = x - (total_width - width) / 2
plt.subplot(1,2,1)
rmse_1=[5.6085,5.5612,5.7057,5.8235,5.6626,6.0323,5.5336]
rmse_2=[5.3883,5.4198,5.5582,5.6798,5.5089,5.7972,5.3924]
rmse_3=[7.4951,7.4619,7.5817,7.5810,7.4643,7.8891,7.4103]
mape_1=[0.3756,0.3498,0.4027,0.3600,0.3527,0.3900,0.3516]
mape_2=[0.3545,0.3255,0.3665,0.3464,0.3340,0.3653,0.3282]
mape_3=[0.2836,0.2789,0.2810,0.2735,0.2685,0.2871,0.2765]
plt.ylim(4,8)
plt.xticks(range(1,9),['GMAN','MT-STNet','STNet','STNet-1','STNet-2','STNet-3','STNet-4'])
plt.bar(x, rmse_1, width=width,label='Entrance toll dataset',color = 'red')
plt.bar(x + width, rmse_2, width=width,label='Exit toll dataset',color = 'black')
plt.bar(x + 2 * width, rmse_3, width=width,label='Gantry dataset',color='salmon')
plt.ylabel('RMSE',font2)
# plt.title('Target time steps $Q$ = 6 ([0-30 min])',font2)
plt.legend()

plt.subplot(1,2,2)
plt.ylim(0.2, 0.45)
plt.xticks(range(1,9),['GMAN','MT-STNet','STNet','STNet-1','STNet-2','STNet-3','STNet-4'])
plt.bar(x, mape_1, width=width,label='Entrance toll dataset',color = 'red')
plt.bar(x + width, mape_2, width=width,label='Exit toll dataset',color = 'black')
plt.bar(x + 2 * width, mape_3, width=width,label='Gantry dataset',color='salmon')
plt.ylabel('MAPE',font2)
# plt.title('Target time steps $Q$ = 6 ([0-30 min])',font2)
plt.legend()
plt.show()
'''


# 可视化每个模型在MAE，RMSE和MAPE上的一个表现
'''
plt.subplot(1,3,1)
plt.plot(range(1,7,1),LSTM_mae,marker='P',color='red',linestyle='-', linewidth=1,label='LSTM')
plt.plot(range(1,7,1),Bi_LSTM_mae,marker='h',color='blue',linestyle='-', linewidth=1,label='Bi-LSTM')
plt.plot(range(1,7,1),FI_RNNs_mae,marker='o',color='orange',linestyle='-', linewidth=1,label='FI-RNNs')
plt.plot(range(1,7,1), PSPNN_mae,marker='s', color='#0cdc73',linestyle='-',linewidth=1,label='PSPNN')
plt.plot(range(1,7,1), MDL_mae,marker='p', color='#f504c9',linestyle='-',linewidth=1,label='MDL')
plt.plot(range(1,7,1),GMAN_mae,marker='^',color='#d0c101',linestyle='-', linewidth=1,label='GMAN')
plt.plot(range(1,7,1),T_GCN_mae ,marker='d', color='#ff5b00',linestyle='-',linewidth=1,label='T-GCN')
plt.plot(range(1,7,1), AST_GAT_mae,marker='*', color='#82cafc',linestyle='-',linewidth=1,label='AST-GAT')
plt.plot(range(1,7,1), STGIN_mae,marker='X', color='#a55af4',linestyle='-',linewidth=1,label='STGIN')
plt.legend(loc='upper left',prop=font1)
plt.grid(axis='y')
plt.ylabel('MAE',font2)
plt.xlabel('Target time steps',font2)
# plt.title('MAE',font2)
plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))

plt.subplot(1,3,2)
# plt.xticks(range(1,8), range(0,31,5))
plt.plot(range(1,7,1),LSTM_rmse,marker='P',color='red',linestyle='-', linewidth=1,label='LSTM')
plt.plot(range(1,7,1),Bi_LSTM_rmse,marker='h',color='blue',linestyle='-', linewidth=1,label='Bi-LSTM')
plt.plot(range(1,7,1),FI_RNNs_rmse,marker='o',color='orange',linestyle='-', linewidth=1,label='FI-RNNs')
plt.plot(range(1,7,1), PSPNN_rmse,marker='s', color='#0cdc73',linestyle='-',linewidth=1,label='PSPNN')
plt.plot(range(1,7,1), MDL_rmse,marker='p', color='#f504c9',linestyle='-',linewidth=1,label='MDL')
plt.plot(range(1,7,1),GMAN_rmse,marker='^',color='#d0c101',linestyle='-', linewidth=1,label='GMAN')
plt.plot(range(1,7,1),T_GCN_rmse ,marker='d', color='#ff5b00',linestyle='-',linewidth=1,label='T-GCN')
plt.plot(range(1,7,1), AST_GAT_rmse,marker='*', color='#82cafc',linestyle='-',linewidth=1,label='AST-GAT')
plt.plot(range(1,7,1), STGIN_rmse,marker='X', color='#a55af4',linestyle='-',linewidth=1,label='STGIN')
plt.legend(loc='upper left',prop=font1)
plt.grid(axis='y')
plt.ylabel('RMSE',font2)
plt.xlabel('Target time steps',font2)
# plt.title('Exit tall dataset',font2)
plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))

plt.subplot(1,3,3)
plt.plot(range(1,7,1),LSTM_mape,marker='P',color='red',linestyle='-', linewidth=1,label='LSTM')
plt.plot(range(1,7,1),Bi_LSTM_mape,marker='h',color='blue',linestyle='-', linewidth=1,label='Bi-LSTM')
plt.plot(range(1,7,1),FI_RNNs_mape,marker='o',color='orange',linestyle='-', linewidth=1,label='FI-RNNs')
plt.plot(range(1,7,1), PSPNN_mape,marker='s', color='#0cdc73',linestyle='-',linewidth=1,label='PSPNN')
plt.plot(range(1,7,1), MDL_mape,marker='p', color='#f504c9',linestyle='-',linewidth=1,label='MDL')
plt.plot(range(1,7,1),GMAN_mape,marker='^',color='#d0c101',linestyle='-', linewidth=1,label='GMAN')
plt.plot(range(1,7,1),T_GCN_mape ,marker='d', color='#ff5b00',linestyle='-',linewidth=1,label='T-GCN')
plt.plot(range(1,7,1), AST_GAT_mape,marker='*', color='#82cafc',linestyle='-',linewidth=1,label='AST-GAT')
plt.plot(range(1,7,1), STGIN_mape,marker='X', color='#a55af4',linestyle='-',linewidth=1,label='STGIN')
plt.ylabel('MAPE',font2)
plt.xlabel('Target time steps',font2)
# plt.title('Gantry dataset',font2)
plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
plt.legend(loc='upper left',prop=font1)
plt.grid(axis='y')
plt.show()
'''