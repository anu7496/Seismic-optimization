# -*- coding: utf-8 -*-
"""
Created on Sun May 17 09:33:12 2020

@author: Anubrata Roy
"""
from pymongo import MongoClient
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
#%%
plt.rcParams['font.family'] = 'DeJavu Serif'
plt.rcParams['font.serif'] = ['Times New Roman']
plt.rcParams['font.size'] = '10'
plt.rcParams["figure.dpi"] = 120
#%%
client = MongoClient()
db_mer_eq = client.mer_error_eq_all
db_mwa_eq = client.mwa_error_eq_all
db_stalta_eq = client.stalta_error_eq_all

df_mer_eq = pd.DataFrame(list(db_mer_eq.error_p.find({})))
df_mwa_eq = pd.DataFrame(list(db_mwa_eq.error_p.find({})))
df_stalta_eq = pd.DataFrame(list(db_stalta_eq.error_p.find({})))

client.close()
err_mer_eq = df_mer_eq.drop(['_id','PGA_pos',], axis = 1)
err_mwa_eq = df_mwa_eq.drop(['_id','PGA_pos',], axis = 1)
err_stalta_eq = df_stalta_eq.drop(['_id','PGA_pos',], axis = 1)
err_mer_eq['Detector'] = 'MER'
err_mwa_eq['Detector'] = 'MWA'
err_stalta_eq['Detector'] = 'STALTA'
err_df_eq = pd.concat([err_mer_eq,err_mwa_eq,err_stalta_eq])
#%%
err_df_eq['Detected'] = True
err_df_eq['Error'] = err_df_eq['Error'].replace(np.nan, False)
err_df_eq.loc[err_df_eq['Error']==False ,'Detected'] = False
err_df_eq['Error'] = err_df_eq['Error'].replace(False, np.nan)
depth_bins = [0,70,300]
depth_labels = ['Shallow(< 70km)','Deep(>= 70km)']
pga_bins = [0,38.245935,176.5197,637.43225]
pga_labels = ['Light(I-IV)','Moderate(V-VI)','Severe(> VI)']
err_df_eq['Earthquake Type'] = pd.cut(err_df_eq['Depth_km'], bins=depth_bins, labels = depth_labels)
err_df_eq['Earthquake Level(MMI)'] = pd.cut(err_df_eq['PGA_gal'], bins=pga_bins,labels = pga_labels)

L_TPR = []
M_TPR = []
S_TPR = []

MER_TP = len(err_df_eq[(err_df_eq['Earthquake Level(MMI)']=='Light(I-IV)') 
                    & (err_df_eq['Detector']=='MER')
                    & (err_df_eq['Detected']==True)])
MER_T = len(err_df_eq[(err_df_eq['Earthquake Level(MMI)']=='Light(I-IV)') 
                    & (err_df_eq['Detector']=='MER')])
L_TPR.append(MER_TP/MER_T)

MWA_TP = len(err_df_eq[(err_df_eq['Earthquake Level(MMI)']=='Light(I-IV)') 
                    & (err_df_eq['Detector']=='MWA')
                    & (err_df_eq['Detected']==True)])
MWA_T = len(err_df_eq[(err_df_eq['Earthquake Level(MMI)']=='Light(I-IV)') 
                    & (err_df_eq['Detector']=='MWA')])
L_TPR.append(MWA_TP/MWA_T)

STALTA_TP = len(err_df_eq[(err_df_eq['Earthquake Level(MMI)']=='Light(I-IV)') 
                    & (err_df_eq['Detector']=='STALTA')
                    & (err_df_eq['Detected']==True)])
STALTA_T = len(err_df_eq[(err_df_eq['Earthquake Level(MMI)']=='Light(I-IV)') 
                    & (err_df_eq['Detector']=='STALTA')])
L_TPR.append(STALTA_TP/STALTA_T)


MER_TP = len(err_df_eq[(err_df_eq['Earthquake Level(MMI)']=='Moderate(V-VI)') 
                    & (err_df_eq['Detector']=='MER')
                    & (err_df_eq['Detected']==True)])
MER_T = len(err_df_eq[(err_df_eq['Earthquake Level(MMI)']=='Moderate(V-VI)') 
                    & (err_df_eq['Detector']=='MER')])
M_TPR.append(MER_TP/MER_T)

MWA_TP = len(err_df_eq[(err_df_eq['Earthquake Level(MMI)']=='Moderate(V-VI)') 
                    & (err_df_eq['Detector']=='MWA')
                    & (err_df_eq['Detected']==True)])
MWA_T = len(err_df_eq[(err_df_eq['Earthquake Level(MMI)']=='Moderate(V-VI)') 
                    & (err_df_eq['Detector']=='MWA')])
M_TPR.append(MWA_TP/MWA_T)

STALTA_TP = len(err_df_eq[(err_df_eq['Earthquake Level(MMI)']=='Moderate(V-VI)') 
                    & (err_df_eq['Detector']=='STALTA')
                    & (err_df_eq['Detected']==True)])
STALTA_T = len(err_df_eq[(err_df_eq['Earthquake Level(MMI)']=='Moderate(V-VI)') 
                    & (err_df_eq['Detector']=='STALTA')])
M_TPR.append(STALTA_TP/STALTA_T)


MER_TP = len(err_df_eq[(err_df_eq['Earthquake Level(MMI)']=='Severe(> VI)') 
                    & (err_df_eq['Detector']=='MER')
                    & (err_df_eq['Detected']==True)])
MER_T = len(err_df_eq[(err_df_eq['Earthquake Level(MMI)']=='Severe(> VI)') 
                    & (err_df_eq['Detector']=='MER')])
S_TPR.append(MER_TP/MER_T)

MWA_TP = len(err_df_eq[(err_df_eq['Earthquake Level(MMI)']=='Severe(> VI)') 
                    & (err_df_eq['Detector']=='MWA')
                    & (err_df_eq['Detected']==True)])
MWA_T = len(err_df_eq[(err_df_eq['Earthquake Level(MMI)']=='Severe(> VI)') 
                    & (err_df_eq['Detector']=='MWA')])
S_TPR.append(MWA_TP/MWA_T)

STALTA_TP = len(err_df_eq[(err_df_eq['Earthquake Level(MMI)']=='Severe(> VI)') 
                    & (err_df_eq['Detector']=='STALTA')
                    & (err_df_eq['Detected']==True)])
STALTA_T = len(err_df_eq[(err_df_eq['Earthquake Level(MMI)']=='Severe(> VI)') 
                    & (err_df_eq['Detector']=='STALTA')])
S_TPR.append(STALTA_TP/STALTA_T)

width = 0.8  # the width of the bars

labels = ["MER", "MWA", "STA/LTA"]
x = np.arange(len(labels))  # the label locations

fig, ax = plt.subplots()
rects1 = ax.bar(x-width/3, L_TPR, width/3,color=(0.05,1,0.03,0.4), label='Light(I-IV)')
rects2 = ax.bar(x, M_TPR, width/3, color=(1,0.65,0, 0.8),label='Moderate(V-VI)')
rects3 = ax.bar(x+width/3, S_TPR, width/3,color=(1,0.21,0.03, 0.8), label='Severe(> VI)')

def autolabel(rects):
    """Attach a text label above each bar in *rects*, displaying its height."""
    for rect in rects:
        height = rect.get_height()
        ax.annotate('{}'.format(np.round(height,2)),
                    xy=(rect.get_x() + rect.get_width() / 3, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom',rotation = 90)


autolabel(rects1)
autolabel(rects2)
autolabel(rects3)

# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_ylabel('Normalized Sensitivity (TPR)')
ax.set_title('Sensitivity of detection techniques\n categorized by earthquake intensity ')
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.legend()

plt.legend(bbox_to_anchor=(1.05, 1.0),loc = 'upper left',title = 'Earthquake Intensity(MMI)',framealpha = 0.2)
plt.grid(which = 'major', alpha = 0.7)
ax.set(ylim=(0, 1.1))

plt.tight_layout()

plt.show()
#%%
client = MongoClient()
db_mer_noise = client.mer_error_noise_all
db_mwa_noise= client.mwa_error_noise_all
db_stalta_noise= client.stalta_error_noise_all

df_mer_noise = pd.DataFrame(list(db_mer_noise.error_p.find({})))
df_mwa_noise = pd.DataFrame(list(db_mwa_noise.error_p.find({})))
df_stalta_noise = pd.DataFrame(list(db_stalta_noise.error_p.find({})))

client.close()
err_mer_noise = df_mer_noise.drop(['_id','PGA_pos',], axis = 1)
err_mwa_noise = df_mwa_noise.drop(['_id','PGA_pos',], axis = 1)
err_stalta_noise = df_stalta_noise.drop(['_id','PGA_pos',], axis = 1)

err_mer_noise['Detector'] = 'MER'
err_mwa_noise['Detector'] = 'MWA'
err_stalta_noise['Detector'] = 'STALTA'
err_df_noise = pd.concat([err_mer_noise,err_mwa_noise,err_stalta_noise])
#%%
err_df_noise['Detected'] = False
err_df_noise['Error'] = err_df_noise['Error'].replace(np.nan, True)
err_df_noise.loc[err_df_noise['Error']==True ,'Detected'] = True
err_df_noise['Error'] = err_df_noise['Error'].replace(True, np.nan)


TNR = []


MER_TN = len(err_df_noise[(err_df_noise['Detector']=='MER')
                    & (err_df_noise['Detected']==True)])
MER_N = len(err_df_noise[(err_df_noise['Detector']=='MER')])
TNR = [MER_TN/MER_N]

MWA_TN = len(err_df_noise[(err_df_noise['Detector']=='MWA')
                    & (err_df_noise['Detected']==True)])
MWA_N = len(err_df_noise[(err_df_noise['Detector']=='MWA')])
TNR.append(MWA_TN/MWA_N)

STALTA_TN = len(err_df_noise[(err_df_noise['Detector']=='STALTA')
                    & (err_df_noise['Detected']==True)])
STALTA_N = len(err_df_noise[(err_df_noise['Detector']=='STALTA')])
TNR.append(STALTA_TN/STALTA_N)


width = 0.8  # the width of the bars

labels = ["MER", "MWA", "STA/LTA"]
x = np.arange(len(labels))  # the label locations

TPR =[np.mean([L_TPR[0],M_TPR[0],S_TPR[0]]),np.mean([L_TPR[1],M_TPR[1],S_TPR[1]]),
      np.mean([L_TPR[2],M_TPR[2],S_TPR[2]])]
BA = [np.mean([TNR[0],TPR[0]]), np.mean([TNR[1],TPR[1]]), np.mean([TNR[2],TPR[2]])]

plt.rcParams['figure.figsize'] = (6, 4)
plt.rcParams["figure.dpi"] = 120
fig, ax = plt.subplots()
rects1 = ax.bar(x-width/3, TPR, width/3, label='Sensitivity')
rects2 = ax.bar(x, TNR, width/3,label='Specificity')
rects3 = ax.bar(x+width/3, BA, width/3, label='Balanced Accuracy')


def autolabel(rects):
    """Attach a text label above each bar in *rects*, displaying its height."""
    for rect in rects:
        height = rect.get_height()
        ax.annotate('{}'.format(np.round(height,3)),
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom',rotation = 90)


autolabel(rects1)
autolabel(rects2)
autolabel(rects3)

# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_ylabel('Normalized performance measures')
ax.set_title('Performance comparision of \nevent detection techniques')
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.legend()

plt.legend(bbox_to_anchor=(1.05, 1.0),loc = 'upper left',title = 'BA based optimized \nperformance measures',framealpha = 0.2)
plt.grid(which = 'major', alpha = 0.7)
ax.set(ylim=(0, 1.15))

plt.tight_layout()

plt.show()
