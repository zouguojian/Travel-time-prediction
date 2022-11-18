# -- coding: utf-8 --
import seaborn as sn
import matplotlib.pyplot as plt

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from matplotlib.ticker import MaxNLocator
import seaborn as sns

DNN_mae =[5.187, 12.089, 1.256, 4.205]
CoDriverETA_mae =[5.200, 12.104, 1.380, 4.236]
DeepTTE_mae =[6.108, 13.106, 3.716, 5.524]
WDR_mae =[5.172, 12.059, 1.406, 4.212]
CompactETA_mae =[5.154, 12.075, 1.236, 4.193]
CTTE_mae =[5.191, 11.963, 1.271, 4.197]
MTSTAN_mae =[5.130, 11.978, 1.229, 4.167]
# T_GCN_mae =[5.670292 ,5.675068 ,5.746703 ,5.863256 ,5.888195 ,5.831183]
# AST_GAT_mae =[5.248924 ,5.258951 ,5.329265 ,5.428566 ,5.421416 ,5.314834]
# GMAN_mae =[5.233438 ,5.208744 ,5.256062 ,5.329036 ,5.347667 ,5.204790]
# STGIN_mae =[5.175827 ,5.100160 ,5.149714 ,5.222380 ,5.245451 ,5.150738]

DNN_rmse = [28.663, 53.375, 1.679, 28.636]
CoDriverETA_rmse =[28.630, 53.209, 1.817, 28.724]
DeepTTE_rmse =[29.718, 53.394, 5.944, 26.512]
WDR_rmse =[28.666, 53.126, 1.827, 28.687]
CompactETA_rmse =[28.641, 53.082, 1.620, 28.714]
CTTE_rmse =[28.617, 53.182, 1.960, 28.486]
MTSTAN_rmse =[28.615, 53.087, 1.656,28.672]
# T_GCN_rmse =[9.134161 ,9.173527 ,9.245162 ,9.387154 ,9.317363 ,9.195807]
# AST_GAT_rmse =[8.705578 ,8.784430 ,8.828416 ,8.943468 ,8.809244 ,8.712925]
# GMAN_rmse =[8.730351 ,8.795372 ,8.797104 ,8.908348 ,8.818618 ,8.598270]
# STGIN_rmse =[8.644908 ,8.640150 ,8.701153 ,8.819650 ,8.724352 ,8.527315]


DNN_mape = [0.09376, 0.14569, 0.05434, 0.09158]
CoDriverETA_mape =[0.09414, 0.15145, 0.05980, 0.09077]
DeepTTE_mape =[0.12104, 0.17635, 0.13182, 0.18333]
WDR_mape =[0.0925, 0.14720, 0.06171, 0.08989]
CompactETA_mape =[0.09285, 0.15010, 0.05341, 0.08833]
CTTE_mape =[0.09384, 0.14335, 0.05195, 0.08942]
MTSTAN_mape =[0.09190, 0.14592, 0.05386,0.08774]
# T_GCN_mape =[0.138232 ,0.141509 ,0.124293 ,0.118078 ,0.119514 ,0.132048]
# AST_GAT_mape =[0.124102 ,0.128212 ,0.114900 ,0.110572 ,0.110653 ,0.121733]
# GMAN_mape =[0.130036 ,0.131432 ,0.116220 ,0.107743 ,0.111887 ,0.117915]
# STGIN_mape =[0.126175 ,0.127834 ,0.112100 ,0.104818 ,0.108805 ,0.117945]

font2 = {'family' : 'Times New Roman',
'weight' : 'normal',
'size'   : 12.,
}

font1 = {'family' : 'Times New Roman',
'weight' : 'normal',
'size'   : 12,
}


"""
用来画红色路段和绿色路段重合部分，行程时间时段上的总的行程轨迹
"""
'''
sns.set_theme(style='ticks', font_scale=1.3,font='Times New Roman')
data = pd.read_csv('/Users/guojianzou/Travel-time-prediction/data/statistic/1.csv',encoding='utf-8')
g=sns.displot(data, x='period', col="Road Name",
              row_order=['[0, 10)','[10, 20)','[20, 30)','[30, 40)','[40, 50)','[50, 60)','[60, )'],)
g.set_axis_labels('Travel Time Interval (min)',y_var='Total Amount of the Vehicles')
# g.set_xticklabels(fontsize=12)
# g.set_yticklabels(fontsize=10)
plt.show()
'''


CoDriverETA = pd.read_csv('results/CoDriverETA-1.csv',encoding='utf-8')
WDR= pd.read_csv('results/WDR-1.csv',encoding='utf-8')
CompactETA = pd.read_csv('results/CompactETA-4.csv',encoding='utf-8')
CTTE= pd.read_csv('results/CTTE-4.csv',encoding='utf-8')
MT_STAN = pd.read_csv('results/MT-STAN-1.csv',encoding='utf-8')


"""研究每条路上有无出现的车辆对行程时间的影响，以及确定我们的模型在每条路上的实际拟合情况，在第一个数据集上"""
# '''
MT_STAN = MT_STAN[(MT_STAN['label_sum']<50)]
# g=sns.jointplot(x="vehicle type", y="label_sum", data=MT_STAN, hue='first appear')
# g.set_axis_labels(xlabel='Vehicle Type', ylabel='Observed Individual Travel Time (min)')
# plt.show()
#
# MT_STAN = MT_STAN[((MT_STAN['vehicle type']==1)|(MT_STAN['vehicle type']==11)|
#                   (MT_STAN['vehicle type']==12)|(MT_STAN['vehicle type']==13)|
#                   (MT_STAN['vehicle type']==14)|(MT_STAN['vehicle type']==16))]

sns.set_theme(style='ticks', font_scale=1.3,font='Times New Roman')
# data=MT_STAN.query('label_sum<100')
# 用以表示在训练集中的出现过的ID和没出现过的ID，他们在测试集上的预测表现
for i in range(5):
    # “scatter” | “kde” | “hist” | “hex” | “reg” | “resid”
    g=sns.jointplot(x="segment_label_"+str(i), y="segment_pre_"+str(i), data=MT_STAN, hue='first appear')
    # g = sns.jointplot(data=MT_STAN, x="segment_label_"+str(i), y="segment_pre_"+str(i), hue="vehicle type", kind="kde")
    g.set_axis_labels(xlabel='Observed Individual Travel Time (min)', ylabel='Predicted Individual Travel Time (min)')
    plt.show()
# '''


"""实现展示模型的拟合情况, 在第四个数据集上G20"""
'''
CompactETA = CompactETA[(CompactETA['label_sum']<55)]
CTTE = CTTE[(CTTE['label_sum']<55)]
MT_STAN = MT_STAN[(MT_STAN['label_sum']<55)]
sns.set_theme(style='ticks', font_scale=1.,font='Times New Roman')
# f, (ax1) = plt.subplots(nrows=1,ncols=3)
CompactETA.rename(columns={'label_sum':'Observed Individual Travel Time (min)','pre_sum':'Predicted Individual Travel Time (min)'},inplace=True)
CTTE.rename(columns={'label_sum':'Observed Individual Travel Time (min)','pre_sum':'Predicted Individual Travel Time (min)'},inplace=True)
MT_STAN.rename(columns={'label_sum':'Observed Individual Travel Time (min)','pre_sum':'Predicted Individual Travel Time (min)'},inplace=True)
# sns.regplot(x="Observed Individual Travel Time (min)", y="Predicted Individual Travel Time (min)", data=CompactETA, color='#969696',label='CompactETA')
# sns.regplot(x="Observed Individual Travel Time (min)", y="Predicted Individual Travel Time (min)", data=CTTE, color='#969696',label='CTTE')
sns.regplot(x="Observed Individual Travel Time (min)", y="Predicted Individual Travel Time (min)", data=MT_STAN, color='#969696',label='MT-STAN')
plt.ylim(10,35)
plt.legend(loc='upper left')
plt.show()
'''


"""研究车型、稀疏性对预测的影响，在第一个数据集上"""
'''
MT_STAN = MT_STAN[(MT_STAN['label_sum']<50)]

# 这个用来可视化车型和真实行程时间之间的关系
# g=sns.jointplot(x="vehicle type", y="label_sum", data=MT_STAN, hue='first appear')
# g.set_axis_labels(xlabel='Vehicle Type', ylabel='Observed Individual Travel Time (min)')
# plt.show()

MT_STAN = MT_STAN[((MT_STAN['vehicle type']==1)|(MT_STAN['vehicle type']==11)|
                  (MT_STAN['vehicle type']==12)|(MT_STAN['vehicle type']==13)|
                  (MT_STAN['vehicle type']==14)|(MT_STAN['vehicle type']==16))&(MT_STAN['first appear']==1)]

sns.set_theme(style='ticks', font_scale=1.,font='Times New Roman')
# data=MT_STAN.query('label_sum<100')
# 用以表示在训练集中的出现过的ID和没出现过的ID，他们在测试集上的预测表现
g=sns.jointplot(x="label_sum", y="pre_sum", data=MT_STAN, hue='vehicle type')
g.set_axis_labels(xlabel='Observed Individual Travel Time (min)', ylabel='Predicted Individual Travel Time (min)')
plt.show()
'''

"""实现展示模型的拟合情况，最后我们论文中使用的这个图，在第4个数据集上"""
'''
# plt.figure()
CompactETA = CompactETA[(CompactETA['label_sum']<50)]
CTTE = CTTE[(CTTE['label_sum']<50)]
MT_STAN = MT_STAN[(MT_STAN['label_sum']<50)]
plt.subplot(1,3,1)
plt.scatter(CompactETA['label_sum'], CompactETA['pre_sum'],alpha=0.7,color='#969696',edgecolor = "black",marker='o',label=u'CompactETA',linewidths=1)
a=[i for i in range(12, 35)]
b=[i for i in range(12, 35)]
plt.plot(a,b,'black',linewidth=2)
#设置横纵坐标的名称以及对应字体格式
plt.xlabel("Observed Individual Travel Time (min)", font2)
plt.ylabel("Predicted Individual Travel Time (min)", font2)
plt.legend(loc='upper left',prop=font2)

plt.subplot(1,3,2)
plt.scatter(CTTE['label_sum'], CTTE['pre_sum'],alpha=0.7,color='#969696',edgecolor = "black",marker='o',label=u'CTTE',linewidths=1)
plt.plot(a,b,'black',linewidth=2)
plt.xlabel("Observed Individual Travel Time (min)", font2)

plt.legend(loc='upper left',prop=font2)
plt.subplot(1,3,3)
plt.scatter(MT_STAN['label_sum'], MT_STAN['pre_sum'],alpha=0.7,color='#969696',edgecolor = "black",marker='o',label=u'MT-STAN',linewidths=1)
plt.plot(a,b,'black',linewidth=2)
# plt.title("Gantry dataset", font2)
#设置横纵坐标的名称以及对应字体格式
plt.xlabel("Observed Individual Travel Time (min)", font2)
plt.legend(loc='upper left',prop=font2)
plt.show()
'''


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