# -*- coding: utf-8 -*-
"""
Created on Thu Feb 17 10:27:30 2022

@author: Anubrata Roy
"""
import pandas as pd
from pymongo import MongoClient
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
#%%
plt.rcParams['font.family'] = 'DeJavu Serif'
plt.rcParams['font.serif'] = ['Times New Roman']
plt.rcParams['font.size'] = '10'
plt.rcParams["figure.dpi"] = 120
#%%
client = MongoClient()
db_mer = client.mer_error_eq_all
db_mwa= client.mwa_error_eq_all
db_stalta= client.stalta_error_eq_all

df_mer = pd.DataFrame(list(db_mer.error_p.find({})))
df_mwa = pd.DataFrame(list(db_mwa.error_p.find({})))
df_stalta = pd.DataFrame(list(db_stalta.error_p.find({})))

client.close()
err_mer = df_mer.drop(['_id','PGA_pos',], axis = 1)
err_mwa = df_mwa.drop(['_id','PGA_pos',], axis = 1)
err_stalta = df_stalta.drop(['_id','PGA_pos',], axis = 1)
err_mer['Detector'] = 'MER'
err_mwa['Detector'] = 'MWA'
err_stalta['Detector'] = 'STALTA'
err_df = pd.concat([err_mer,err_mwa,err_stalta])
#%%
round_off = -1
err_mer.rename(columns={'PGA_gal':'PGA (gal)'}, inplace=True)
err_mwa.rename(columns={'PGA_gal':'PGA (gal)'}, inplace=True)
err_stalta.rename(columns={'PGA_gal':'PGA (gal)'}, inplace=True)
decimals = pd.Series([round_off], index=['PGA (gal)'])
err_mer = err_mer.round(decimals)
err_mwa = err_mwa.round(decimals)
err_stalta = err_stalta.round(decimals)
err_df = pd.concat([err_mer,err_mwa,err_stalta])
#%%
err_df['Detected'] = True
err_df['Error'] = err_df['Error'].replace(np.nan, False)
err_df.loc[err_df['Error']==False ,'Detected'] = False
err_df['Error'] = err_df['Error'].replace(False, np.nan)
depth_bins = [0,70,300]
depth_labels = ['Shallow(< 70km)','Deep(>= 70km)']
pga_bins = [0,38.245935,176.5197,637.43225]
pga_labels = ['Light(I-IV)','Moderate(V-VI)','Severe(> VI)']
err_df['Earthquake Type'] = pd.cut(err_df['Depth_km'], bins=depth_bins, labels = depth_labels)
err_df['Earthquake Level(MMI)'] = pd.cut(err_df['PGA (gal)'], bins=pga_bins,labels = pga_labels)
#%%
plt.rcParams['figure.figsize'] = (6,4.5)
eq_size_pal = [(0.05,1,0.03,0.4),(1,0.65,0, 0.8),(1,0.21,0.03, 0.8)]
palette = sns.color_palette(eq_size_pal)
plt.figure()
ax = sns.boxenplot(x="Detector", y="Error",
              hue ='Earthquake Level(MMI)',hue_order = ['Light(I-IV)','Moderate(V-VI)','Severe(> VI)'],
              palette = palette,
              data=err_df)
# ax = sns.boxenplot(x="Detector", y="Error",
#               data=err_df)
# median_detector_mmi = err_df.groupby(['Detector','Earthquake Level(MMI)'])['Error'].median()
# median_detector = err_df.groupby(['Detector'])['Error'].median()
plt.legend(loc = 'upper right',title = 'Earthquake Level(MMI)',framealpha = 0.2)
ax.axhline(y=0,c = 'k',linewidth = '0.5')
plt.grid(which = 'major', alpha = 0.8)
ax.set_title('Detection Delay distribution by \nEarthquake intensity and detection techniques')
# ax.set_title('Detection Delay distribution of detection techniques')
ax.set(ylim=(-1, 5))
ax.set_ylabel('Detection Delay (sec)')
ax.set_xlabel('Detection techniques')
plt.tight_layout()