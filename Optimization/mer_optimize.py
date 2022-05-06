# -*- coding: utf-8 -*-
"""
Created on Thu April 14 12:27:45 2022

@author: Anubrata Roy
"""
from __future__ import division
from pymongo import MongoClient
import numpy as np
import math
import random
from scipy.signal import butter,filtfilt
from joblib import Parallel, delayed
import multiprocessing
import winsound
import pandas as pd
import matplotlib.pyplot as plt
#%%
lowcut = 32
alphas = np.logspace(math.log2(5),math.log2(500), num = 16, base = 2)  # thresholds
durations  = np.logspace(math.log2(0.5),math.log2(3), num = 16, base = 2)  # leading window durations

def LPF(data, lowcut, fs, order=5):
    if data.ndim <2:  # data should be in [channel,data] matrix format
        data = data.reshape(1,-1)  # for filtering single channel data
    filtered_signal = []
    nyq = 0.5 * fs
    low = lowcut / nyq
    b, a = butter(order, low, btype='low')
    for channel in data:      
        output = filtfilt(b, a, channel)
        filtered_signal.append(output)
    return np.array(filtered_signal)

def sliding_window(arr, window_len = 4, shift_index = 2, copy = False): # create sliding window from given data
    sh = (arr.size - window_len + 1, window_len)
    st = arr.strides * 2
    view = np.lib.stride_tricks.as_strided(arr, strides = st, shape = sh)[0::shift_index]
    if copy:
        return view.copy()
    else:
        return view
    
def rescale(arr, factor=2): 
    n = len(arr)
    return np.interp(np.linspace(0, n, factor*n+1), np.arange(n), arr)
#%%
client=MongoClient("mongodb://localhost:27017/")
db_eq=client.knet_1530
header_eq = db_eq.header
db_op = client.mer_sensitivity

query = {"$and":
                    [ {"Magnitude": {'$gt':3,"$lt":8.1}},
                          {"PGA_gal" :{'$gt':1}}
                          
                      ]}

selectedEvents_eq = header_eq.find(query)
nof_selection_eq = selectedEvents_eq.count()
record_list = list(zip(list(range(nof_selection_eq)), selectedEvents_eq))
sample_ratio = 0.25
rand_rec_list_eq = random.sample(record_list,int(len(record_list)*sample_ratio))
nof_selection_sample = len(rand_rec_list_eq)

db_noise=client.DMRC_till_201902
header_noise = db_noise.header
db_op_noise = client.mer_specificity
selectedEvents_noise = header_noise.find()
nof_selection_noise = selectedEvents_noise.count()
record_list_n = list(zip(list(range(nof_selection_noise)), selectedEvents_noise))
sample_ratio = 0.25
rand_rec_list_noise = random.sample(record_list_n,int(len(record_list_n)*sample_ratio))
nof_selection_sample_n = len(rand_rec_list_noise)
#%%
def gridsearch_eq(c,nof_selection_eq,alphas):
    x = c[1]
    client = MongoClient()
    db_eq=client.knet_1530
    accelerogram_eq = db_eq.accelerogram
    db_op = client.mer_sensitivity
    f_name = x["_id"]
    sps = x["SamplingRate_Hz"]
    data_eq = []
    data_eq.append(accelerogram_eq.find({ "_id": f_name+".UD"})[0]["accelerogram"])
    data_eq.append(accelerogram_eq.find({ "_id": f_name+".NS"})[0]["accelerogram"])
    data_eq.append(accelerogram_eq.find({ "_id": f_name+".EW"})[0]["accelerogram"])
    acc = np.array(data_eq)
    nof_datapoints = int(acc.shape[1])
    dc_offset = np.mean(acc,axis=1).reshape((3,1))
    acc-= dc_offset
    acc_filtered = LPF(acc,lowcut,sps,order=5)
    acc_f = np.sqrt(np.sum(acc_filtered*acc_filtered,axis = 0)) # triaxial vector sum of 3 channel data
    acc_abs = np.abs(acc_f)
    global_max_acc = acc_abs.max()
    indices = np.where(acc_abs == global_max_acc)
    pga_index = indices[0][0]
    overlap = 0.2
    stride = int(sps*overlap)
    error_p_mer = []
    error_s_mer = []
    for d in durations:
        dur = int(sps * d)
        win = sliding_window(acc_f,2*dur-1,stride)
        window1 = np.sum(win[:,:dur]**2,axis =1)
        window2 = np.sum(win[:,-dur:]**2,axis =1)
        er = window2/window1
        mer = (er*np.abs(win[:,dur-1]))**3
        # mer_mod = mer[1:]/mer[:-1]
        fact = int(np.ceil(len(acc_f)/len(mer)))
        mer_mod = rescale(mer, factor = fact)
        for alfa in alphas:
            indx = [index for index,value in enumerate(mer_mod[:nof_datapoints]) if value > alfa]
            if len(indx)> 0:
                tg_p_phase = indx[0]+dur-stride//2 # minimum delay to detect p is window length
                tg_p = tg_p_phase+dur
                if  tg_p > x["pIndex"] - 1*sps and tg_p < pga_index:
                    if x["p"] and tg_p <(x["pIndex"] + 4.5*sps) and tg_p <x["sIndex"] :
                        error_p_mer.append([d,(round(alfa, 2)),(tg_p-x["pIndex"])/sps])
                    elif x["p"] and ((tg_p >= (x["pIndex"] + 4.5*sps)) or tg_p >= x["sIndex"]) :
                        error_p_mer.append([d,(round(alfa, 2)),np.nan]) 
                        error_s_mer.append([d,(round(alfa, 2)),(tg_p-x["sIndex"])/sps])
                    else:
                        error_s_mer.append([d,(round(alfa, 2)),(tg_p-x["sIndex"])/sps])
                else:
                    if x["p"]:
                        error_p_mer.append([d,(round(alfa, 2)),np.nan])
                    else:
                        error_s_mer.append([d,(round(alfa, 2)),np.nan])
            else:
                if x["p"]:
                    error_p_mer.append([d,(round(alfa, 2)),np.nan])
                else:
                    error_s_mer.append([d,(round(alfa, 2)),np.nan]) #extra brackets around round for str


    op = {}
    for errs in error_p_mer :
        op["_id"] = f_name+'_'+str(errs[0])+'_'+str(errs[1])
        op["Magnitude"] = x["Magnitude"]
        op["Distance_km"] = x["Distance"]
        op["Depth_km"] = x["Depth_km"]
        op["PGA_gal"] = x["PGA_gal"]
        op["PGA_pos"] = int(pga_index)
        op["Th"] = errs[1]
        op['Error'] = errs[2] #delay in detection
        op["Win_len"] = errs[0]
        if x["p"]:
            op["p-s_time"] = (x["sIndex"]-x["pIndex"])/sps
        db_op["error_p"].insert_one(op)


    op = {}
    for errs in error_s_mer :
        op["_id"] = f_name+'_'+str(errs[0])+'_'+str(errs[1])
        op["Magnitude"] = x["Magnitude"]
        op["Distance_km"] = x["Distance"]
        op["Depth_km"] = x["Depth_km"]
        op["PGA_gal"] = x["PGA_gal"]
        op["PGA_pos"] = int(pga_index)
        op["Th"] = errs[1]
        op['Error'] = errs[2] #delay in detection
        op["Win_len"] = errs[0]
        db_op["error_s"].insert_one(op)

#%%
db_op["error_p"].delete_many({})
db_op["error_s"].delete_many({})
num_cores = multiprocessing.cpu_count()
results = Parallel(n_jobs= int(num_cores*1) ,prefer = 'processes',
                      verbose=1
                      #backend="multiprocessing" #use multiprocessing on unix systems specially
                      )(delayed(gridsearch_eq)(
                        c,
                        nof_selection_sample,
                        alphas
                        )
                        for c in rand_rec_list_eq)
#%%
def gridsearch_n(c,nof_selection_eq,alphas):
    x = c[1]
    client = MongoClient()
    db_noise=client.DMRC_till_201902
    accelerogram_noise = db_noise.accelerogram
    db_op_noise = client.mer_specificity

    f_name = x["_id"]
    sps = x["SamplingRate_Hz"] 
    try:
        data_noise = []
        data_noise.append(accelerogram_noise.find({ "_id": f_name+".UD"})[0]["accelerogram"])
        data_noise.append(accelerogram_noise.find({ "_id": f_name+".NS"})[0]["accelerogram"])
        data_noise.append(accelerogram_noise.find({ "_id": f_name+".EW"})[0]["accelerogram"])
        acc = np.array(data_noise)
        nof_datapoints = int(acc.shape[1])
        acc_f = np.sqrt(np.sum(acc*acc,axis = 0))
        acc_abs = np.abs(acc_f)
        global_max_acc = acc_abs.max()
        indices = np.where(acc_abs == global_max_acc)
        pga_index = indices[0][0]
        error_noise_mer = []
        for d in durations:
            dur = int(sps * d)
            win = sliding_window(acc_f,2*dur-1,1)
            window1 = np.sum(win[:,:dur]**2,axis =1)
            window2 = np.sum(win[:,-dur:]**2,axis =1)
            er = window2/window1
            mer = (er*np.abs(win[:,dur-1]))**3
            # mer_mod = mer[1:]/mer[:-1]
            fact = int(np.ceil(len(acc_f)/len(mer)))
            mer_mod = rescale(mer, factor = fact)
            for alfa in alphas:
                indx = [index for index,value in enumerate(mer_mod[:nof_datapoints]) if value > alfa]
                if len(indx)> 0: 
                    tg_p = indx[0] # minimum delay to detect p is window length
                    error_noise_mer.append([d,(round(alfa, 2)),(tg_p)/sps])
                else:
                    error_noise_mer.append([d,(round(alfa, 2)),np.nan]) #extra brackets around round for str
    
    
        op = {}
        for errs in error_noise_mer :
            op["_id"] = f_name+'_'+str(errs[0])+'_'+str(errs[1])
            op["PGA_gal"] = x["PGA_gal"]
            op["PGA_pos"] = int(pga_index)
            op["Th"] = errs[1]
            op['Error'] = errs[2]
            op["Win_len"] = errs[0]
            db_op_noise["error_p"].insert_one(op)
    
    except:
        print("error\n\n",f_name)
        winsound.Beep(1000,200)
        db_op_noise["error_file"].insert_one({"_id":f_name,"error": False})
#%%
from joblib import Parallel, delayed
import multiprocessing

db_op_noise["error_p"].delete_many({})
db_op_noise["error_file"].delete_many({})
num_cores = multiprocessing.cpu_count()
results = Parallel(n_jobs= int(num_cores*1) ,prefer = 'processes', 
                      verbose=1
                      #backend="multiprocessing" #use multiprocessing on unix systems specially
                      )(delayed(gridsearch_n)(
                        c,
                        nof_selection_sample_n,
                        alphas
                        )
                        for c in rand_rec_list_noise)

#%%
df_eq = pd.DataFrame(list(db_op.error_p.find({})))
df_nan = df_eq[df_eq['Error'].isnull()]

df_noise = pd.DataFrame(list(db_op_noise.error_p.find({})))
df_noise_nan = df_noise[df_noise['Error'].isnull()]
#%%
err_mer_nan = df_nan.drop(['_id','Distance_km','Depth_km','p-s_time','PGA_pos','Magnitude'], axis = 1)
th_unique_eq = list(err_mer_nan.Th.unique())
th_unique_eq.sort()
win_unique_eq = list(err_mer_nan.Win_len.unique())
win_unique_eq.sort()

def sensitivity(x):
    y = np.zeros(x[0].shape)

    for row in np.arange(x[0].shape[0]):
        for col in np.arange(x[0].shape[1]):
            y[row,col] = (nof_selection_sample -
                          len(err_mer_nan[(err_mer_nan['Win_len']==x[0][row][col])
                                          &(err_mer_nan['Th']==x[1][row][col])]))/nof_selection_sample
    return y

err_mer_noise_nan = df_noise_nan.drop(['_id','PGA_gal','PGA_pos'], axis = 1)
th_unique_noise = list(err_mer_noise_nan.Th.unique())
th_unique_noise.sort()
win_unique_noise = list(err_mer_noise_nan.Win_len.unique())
win_unique_noise.sort()

def specificity(x):
    y = np.zeros(x[0].shape)

    for row in np.arange(x[0].shape[0]):
        for col in np.arange(x[0].shape[1]):
            y[row,col] =len(err_mer_noise_nan[(err_mer_noise_nan['Win_len']==x[0][row][col])
                                          &(err_mer_noise_nan['Th']==x[1][row][col])])/nof_selection_sample_n
    return y
#%%
# plt.rcParams['figure.figsize'] = (7,6)
plt.rcParams['font.family'] = 'DeJavu Serif'
plt.rcParams['font.serif'] = ['Times New Roman']
plt.rcParams['font.size'] = '10'
#%%
plt.rcParams['figure.figsize'] = (4,3)
plt.rcParams["figure.dpi"] = 120
nof_levels = 8
vmax=1
vmin=0.3

win_arr_e = np.array(win_unique_eq)
th_arr_e = np.array(th_unique_eq)
win_grd_e, th_grd_e = np.meshgrid(win_arr_e, th_arr_e)
TPR = sensitivity([win_grd_e, th_grd_e])

fig, ax = plt.subplots()
contour_e = ax.contour(win_grd_e, th_grd_e,TPR,
                          nof_levels,colors = 'k')
clabel_e = ax.clabel(contour_e,colors = 'k', fmt = '%2.2f', fontsize=10)
cpf_e = ax.contourf(win_grd_e, th_grd_e, TPR,12,cmap ='autumn',vmax = vmax,vmin = 0.45,alpha =0.8)
cbar_e = plt.colorbar(cpf_e)
cbar_e.set_label('Normalized Sensitivity')
ax.set_title('Distribution of Sensitivity of MER')
ax.set_xlabel('Window_duration (s)')
ax.set_ylabel('Threshold')
plt.tight_layout()

win_arr_n = np.array(win_unique_noise)
th_arr_n = np.array(th_unique_noise)
win_grd_n, th_grd_n = np.meshgrid(win_arr_n, th_arr_n)
TNR = specificity([win_grd_n, th_grd_n])

fig, ax = plt.subplots()
contour_n = ax.contour(win_grd_n, th_grd_n,TNR,
                          nof_levels,colors = 'k')
clabel_n = ax.clabel(contour_n, fmt = '%2.2f',colors = 'k', fontsize=10)
cpf_n = ax.contourf(win_grd_n, th_grd_n, TNR,nof_levels,cmap ='autumn',vmax = vmax,vmin = 0.1 ,alpha = 0.8)
cbar_n = plt.colorbar(cpf_n)
# ax.yaxis.set_major_formatter(fake_log)
cbar_n.set_label('Normalized Specificity')
ax.set_title('Distribution of Specificity of MER')
ax.set_xlabel('Window_duration (s)')
ax.set_ylabel('Threshold')
plt.tight_layout()

plt.rcParams['figure.figsize'] = (5.5,3)
plt.rcParams["figure.dpi"] = 120

BA = (TPR + TNR)/2 #balanced accuracy for imbalanced data

fig, ax = plt.subplots()
contour = ax.contour(win_grd_n, th_grd_n,BA,
                          nof_levels,colors = 'k')
clabel = ax.clabel(contour, fmt = '%2.2f',colors = 'k', fontsize=10)
cpf = ax.contourf(win_grd_n, th_grd_n, BA,nof_levels,cmap ='seismic',vmax = vmax,vmin = vmin,alpha = 0.7)
cbar = plt.colorbar(cpf)
max_BA = BA.max()
indices = np.where(BA == max_BA)
ax.scatter(win_arr_e[indices[1]],th_arr_e[indices[0]],marker = 'x',color = 'k')
print(win_arr_e[indices[1]],th_arr_e[indices[0]])
cbar.set_label('Normalized Accuracy')
ax.set_title('Distribution of Balanced Accuracy of MER')
ax.set_xlabel('Window_duration (s)')
ax.set_ylabel('Threshold')
plt.tight_layout()
plt.show()                          
