# -*- coding: utf-8 -*-
"""
Created on Thu April 18 15:31:45 2022

@author: Anubrata Roy
"""
from __future__ import division
from pymongo import MongoClient
import numpy as np
from scipy.signal import hilbert,butter,filtfilt
import math
import random
from joblib import Parallel, delayed
import multiprocessing
import winsound
import pandas as pd
import matplotlib.pyplot as plt
#%%
alpha = 3
lowcut = 32

alphas = np.logspace(np.log2(1.5),np.log2(50), num = 16, base = 2) # thresholds
dta_durations  = np.logspace(math.log2(0.5),math.log2(4.5), num = 16, base = 2) # leading window durations

def sliding_window(arr, window_len = 4, shift_index = 2, copy = False): # create sliding window from given data
    sh = (arr.size - window_len + 1, window_len)
    st = arr.strides * 2
    view = np.lib.stride_tricks.as_strided(arr, strides = st, shape = sh)[0::shift_index]
    if copy:
        return view.copy()
    else:
        return view
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

def rescale(arr, factor=2):
    n = len(arr)
    return np.interp(np.linspace(0, n, factor*n+1), np.arange(n), arr)
#%%
client = MongoClient()
db_eq=client.knet_1530
header_eq = db_eq.header

db_op = client.mwa_sensitivity
query = {"$and":
                    [ {"Magnitude": {'$gt':3,"$lt":8.1}},
                          {"PGA_gal" :{'$gt':1}},

                      ]}

selectedEvents_eq = header_eq.find(query)
nof_selection_eq = selectedEvents_eq.count()
record_list = list(zip(list(range(nof_selection_eq)), selectedEvents_eq))
sample_ratio = 0.25
rand_rec_list_eq = random.sample(record_list,int(len(record_list)*sample_ratio))
nof_selection_sample = len(rand_rec_list_eq)

db_noise = client.DMRC_till_201902
header_noise = db_noise.header
db_op_noise = client.mwa_specificity 
selectedEvents_noise = header_noise.find()
nof_selection_noise = selectedEvents_noise.count()
record_list_n = list(zip(list(range(nof_selection_noise)), selectedEvents_noise))
sample_ratio = 0.25
rand_rec_list_noise = random.sample(record_list_n,int(len(record_list_n)*sample_ratio))
nof_selection_sample_n = len(rand_rec_list_noise)
#%%
def gridsearch_eq(c,nof_selection_eq,alphas):
    count = c[0]
    x = c[1]
    client = MongoClient()
    db_eq=client.knet_1530
    accelerogram_eq = db_eq.accelerogram

    db_op = client.mwa_sensitivity
    f_name = x["_id"]
    sps = x["SamplingRate_Hz"]
    data_eq = []
    data_eq.append(accelerogram_eq.find({ "_id": f_name+".UD"})[0]["accelerogram"])
    data_eq.append(accelerogram_eq.find({ "_id": f_name+".NS"})[0]["accelerogram"])
    data_eq.append(accelerogram_eq.find({ "_id": f_name+".EW"})[0]["accelerogram"])

    u = np.array(data_eq)
    nof_datapoints = int(u.shape[1])
    dc_offset = np.mean(u,axis=1).reshape((3,1))
    u-= dc_offset
    u_filtered = LPF(u,lowcut,sps,order=5)
    u_f = np.sqrt(np.sum(u_filtered*u_filtered,axis = 0)) # triaxial vector sum of 3 channel data
    u_abs = np.abs(u_f)
    global_max_acc = u_abs.max()
    indices = np.where(u_abs == global_max_acc)
    pga_index = indices[0][0]
    overlap = 0.2
    stride = int(sps*overlap)
    error_p_mwa = []
    error_s_mwa = []
    for dt in dta_durations:
        bt = 4*dt/6
        at = 0.5*dt
        tot_t = bt+dt
        dur = int(sps * tot_t)
        m = int(sps * bt)
        n = int (sps * at)
        q = int (sps* dt)
        win = sliding_window(u_f,dur-1,stride)
        bta_win = np.abs(win[:,0:m])
        ata_win = np.abs(win[:,m-1:m+n-1])
        dta_win = np.abs(win[:,m-1:m+n+q-1])
        envlop = np.abs(hilbert(bta_win))

        bta = np.mean(bta_win,axis = 1)
        ata = np.mean(ata_win,axis = 1)
        dta = np.mean(dta_win,axis = 1)
        r1 =  np.abs(bta_win[:,-1])
        r2 = ata/bta
        r3 = dta/bta
        envlop_std = np.std(envlop,axis =1)
        envlop_mean = np.mean(envlop,axis=1)
        h1 = envlop_mean + (alpha * envlop_std)
        h1 = h1[:-1]
        fact = int(np.ceil(len(u_f)/len(r1)))
        h1_mod = rescale(h1, factor = fact)
        r1_mod = rescale(r1, factor = fact)
        r2_mod = rescale(r2, factor = fact)
        r3_mod = rescale(r3 ,factor = fact)
        tg_p = 0
        for alfa in alphas:
            indx = [index for index,value in enumerate(zip
                                                       (r1_mod[:nof_datapoints],
                                                        r2_mod[:nof_datapoints],
                                                        r3_mod[:nof_datapoints],
                                                        h1_mod[:nof_datapoints]))
                                                       if (value[0] > value[3] and
                                                           value[1] > alfa and
                                                           value[2] > alfa)
                                                       ]
            if len(indx)> 0:
                tg_p_phase = indx[0]+m-stride//2 # minimum delay to detect p is window length
                tg_p = tg_p_phase+q
                if  tg_p > x["pIndex"] - 1*sps and tg_p < pga_index:
                    if x["p"] and tg_p <(x["pIndex"] + 4.5*sps) and tg_p <x["sIndex"] :
                        error_p_mwa.append([dt,(round(alfa, 2)),(tg_p-x["pIndex"])/sps])
                    elif x["p"] and ((tg_p >= (x["pIndex"] + 4.5*sps)) or tg_p >= x["sIndex"]) :
                        error_p_mwa.append([dt,(round(alfa, 2)),np.nan]) 
                        error_s_mwa.append([dt,(round(alfa, 2)),(tg_p-x["sIndex"])/sps])
                    else:
                        error_s_mwa.append([dt,(round(alfa, 2)),(tg_p-x["sIndex"])/sps])
                else:
                    if x["p"]:
                        error_p_mwa.append([dt,(round(alfa, 2)),np.nan])
                    else:
                        error_s_mwa.append([dt,(round(alfa, 2)),np.nan])
            else:
                if x["p"]:
                    error_p_mwa.append([dt,(round(alfa, 2)),np.nan])
                else:
                    error_s_mwa.append([dt,(round(alfa, 2)),np.nan])


    op = {}
    for errs in error_p_mwa :
        op["_id"] = f_name+'_'+str(errs[0])+'_'+str(errs[1])
        op["Eventid"] = x["_id"]
        op["Magnitude"] = x["Magnitude"]
        op["Distance_km"] = x["Distance"]
        op["Depth_km"] = x["Depth_km"]
        op["PGA_gal"] = x["PGA_gal"]
        op["PGA_pos"] = int(pga_index)
        op["Th"] = errs[1]
        op['Error'] = errs[2] #delay in detection
        op["DTA_len"] = errs[0]
        if x["p"]:
            op["p-s_time"] = (x["sIndex"]-x["pIndex"])/sps
        db_op["error_p"].insert_one(op)


    op = {}
    for errs in error_s_mwa :
        op["_id"] = f_name+'_'+str(errs[0])+'_'+str(errs[1])
        op["Eventid"] = x["_id"]
        op["Magnitude"] = x["Magnitude"]
        op["Distance_km"] = x["Distance"]
        op["Depth_km"] = x["Depth_km"]
        op["PGA_gal"] = x["PGA_gal"]
        op["PGA_pos"] = int(pga_index)
        op["Th"] = errs[1]
        op['Error'] = errs[2] #delay in detection
        op["DTA_len"] = errs[0]
        db_op["error_s"].insert_one(op)


#%%
db_op["error_p"].delete_many({})
db_op["error_s"].delete_many({})
num_cores = multiprocessing.cpu_count()
results = Parallel(n_jobs=num_cores,prefer = 'processes',
                      verbose=1,
                      #backend="multiprocessing" #use multiprocessing on unix systems specially
                      )(delayed(gridsearch_eq)(
                        c,
                        nof_selection_sample,
                        alphas
                        )
                        for c in rand_rec_list_eq)
#%%
def gridsearch_n(c,nof_selection_eq,alphas):
    count = c[0]
    x = c[1]
    client = MongoClient()
    db_noise = client.DMRC_till_201902s
    accelerogram_noise = db_noise.accelerogram
    db_op_noise = client.mwa_specificity
    f_name = x["_id"]
    sps = x["SamplingRate_Hz"]
    try:
        data_noise = []
        data_noise.append(accelerogram_noise.find({ "_id": f_name+".UD"})[0]["accelerogram"])
        data_noise.append(accelerogram_noise.find({ "_id": f_name+".NS"})[0]["accelerogram"])
        data_noise.append(accelerogram_noise.find({ "_id": f_name+".EW"})[0]["accelerogram"])
        
        u = np.array(data_noise)
        nof_datapoints = int(u.shape[1])
        u_f = np.sqrt(np.sum(u*u,axis = 0))
        u_abs = np.abs(u)
        
        global_max_acc = u_abs.max()
        indices = np.where(u_abs == global_max_acc)
        pga_index = indices[0][0]
        # overlap = 0.2
        # stride = int(sps*overlap)        
        error_noise_mwa = []
        for dt in dta_durations:
            bt = (4*dt/6)
            at = 0.5*dt
            tot_t = bt+dt
            dur = int(sps * tot_t)
            m = int(sps * bt)
            n = int (sps * at)
            q = int (sps* dt)
            win = sliding_window(u_f,dur-1,1)
            bta_win = np.abs(win[:,0:m])
            ata_win = np.abs(win[:,m-1:m+n-1])
            dta_win = np.abs(win[:,m-1:m+n+q-1])
            envlop = np.abs(hilbert(bta_win))
    
            bta = np.mean(bta_win,axis = 1)
            ata = np.mean(ata_win,axis = 1)
            dta = np.mean(dta_win,axis = 1)
            r1 =  np.abs(bta_win[:,-1])
            r2 = ata/bta
            r3 = dta/bta
            
            envlop_std = np.std(envlop,axis =1)
            envlop_mean = np.mean(envlop,axis=1)
            
            h1 = envlop_mean + (alpha * envlop_std)
            h1 = h1[:-1]
            fact = int(np.ceil(len(u_f)/len(r1)))
            h1_mod = rescale(h1, factor = fact)
            r1_mod = rescale(r1, factor = fact)
            r2_mod = rescale(r2, factor = fact)
            r3_mod = rescale(r3 ,factor = fact)
            tg_p = 0
            for alfa in alphas:
                indx = [index for index,value in enumerate(zip
                                                           (r1_mod[:nof_datapoints],
                                                            r2_mod[:nof_datapoints],
                                                            r3_mod[:nof_datapoints],
                                                            h1_mod[:nof_datapoints]))
                                                           if (value[0] > value[3] and
                                                               value[1] > alfa and
                                                               value[2] > alfa)
                                                           ]
                if len(indx)> 0: 
                    tg_p = indx[0] # minimum delay to detect p is window length
                    error_noise_mwa.append([dt,(round(alfa, 2)),(tg_p)/sps])
                else:
                    error_noise_mwa.append([dt,(round(alfa, 2)),np.nan])
    
        
        op = {}
        for errs in error_noise_mwa :
            op["_id"] = f_name+'_'+str(errs[0])+'_'+str(errs[1])
            op["Eventid"] = x["_id"]
            op["PGA_gal"] = x["PGA_gal"]
            op["PGA_pos"] = int(pga_index)
            op["Th"] = errs[1]
            op['Error'] = errs[2]
            op["DTA_len"] = errs[0]
            db_op_noise["error_p"].insert_one(op)
    

    except Exception as e:
        print(e)
        print("error\n\n",f_name)
        winsound.Beep(1000,200)
        db_op_noise["error_file"].insert_one({"_id":f_name,"error": False})

#%%

db_op_noise["error_p"].delete_many({})
db_op_noise["error_file"].delete_many({})
num_cores = multiprocessing.cpu_count()
results = Parallel(n_jobs=num_cores,prefer = 'processes',
                      verbose=1,
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
err_mwa_nan = df_nan.drop(['_id','Distance_km','Depth_km','p-s_time','PGA_pos','Magnitude'], axis = 1)
th_unique_eq = list(err_mwa_nan.Th.unique())
th_unique_eq.sort()
win_unique_eq = list(err_mwa_nan.DTA_len.unique())
win_unique_eq.sort()

def sensitivity(x):
    y = np.zeros(x[0].shape)

    for row in np.arange(x[0].shape[0]):
        for col in np.arange(x[0].shape[1]):
            y[row,col] = (nof_selection_sample -
                          len(err_mwa_nan[(err_mwa_nan['DTA_len']==x[0][row][col])
                                          &(err_mwa_nan['Th']==x[1][row][col])]))/nof_selection_sample
    return y

err_mwa_noise_nan = df_noise_nan.drop(['_id','PGA_gal','PGA_pos'], axis = 1)
th_unique_noise = list(err_mwa_noise_nan.Th.unique())
th_unique_noise.sort()
win_unique_noise = list(err_mwa_noise_nan.DTA_len.unique())
win_unique_noise.sort()

def specificity(x):
    y = np.zeros(x[0].shape)

    for row in np.arange(x[0].shape[0]):
        for col in np.arange(x[0].shape[1]):
            y[row,col] =len(err_mwa_noise_nan[(err_mwa_noise_nan['DTA_len']==x[0][row][col])
                                         &(err_mwa_noise_nan['Th']==x[1][row][col])])/nof_selection_sample_n
    return y

#%%
# plt.rcParams['figure.figsize'] = (7,6)
plt.rcParams['font.family'] = 'DeJavu Serif'
plt.rcParams['font.serif'] = ['Times New Roman']
plt.rcParams['font.size'] = '10'
#%%
import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = (4,3)
plt.rcParams["figure.dpi"] = 120
nof_levels = 12
vmax=1
vmin=0.3

win_arr_e = np.array(win_unique_eq)
th_arr_e = np.array(th_unique_eq)
win_grd_e, th_grd_e = np.meshgrid(win_arr_e, th_arr_e)
TPR = sensitivity([win_grd_e, th_grd_e])

plt.figure()
contour_e = plt.contour(win_grd_e, th_grd_e,TPR,
                         nof_levels,colors = 'k')
clabel_e = plt.clabel(contour_e,colors = 'k', fmt = '%2.2f', fontsize=10)
cpf_e = plt.contourf(win_grd_e, th_grd_e, TPR,nof_levels,cmap ='autumn',vmax = vmax,vmin = 0.3,alpha = 0.8)
cbar_e = plt.colorbar(cpf_e)
cbar_e.set_label('Normalized Sensitivity')
plt.title('Distribution of Sensitivity of MWA')
plt.xlabel('DTA_duration (s)')
plt.ylabel('Threshold')
plt.tight_layout()

win_arr_n = np.array(win_unique_noise)
th_arr_n = np.array(th_unique_noise)
win_grd_n, th_grd_n = np.meshgrid(win_arr_n, th_arr_n)
TNR = specificity([win_grd_n, th_grd_n])

plt.figure()
contour_n = plt.contour(win_grd_n, th_grd_n,TNR,
                         nof_levels,colors = 'k')
clabel_n = plt.clabel(contour_n, fmt = '%2.2f',colors = 'k', fontsize=10)
cpf_n = plt.contourf(win_grd_n, th_grd_n, TNR,nof_levels,cmap ='autumn',vmax = vmax,vmin = 0.3,alpha = 0.8)
cbar_n = plt.colorbar(cpf_n)
cbar_n.set_label('Normalized Specificity')
plt.title('Distribution of Specificity of MWA')
plt.xlabel('DTA_duration (s)')
plt.ylabel('Threshold')
plt.tight_layout()
plt.show()

BA = (TPR + TNR)/2 #balanced accuracy for imbalanced data

plt.rcParams['figure.figsize'] = (5.5,3)
plt.rcParams["figure.dpi"] = 120
plt.figure()

contour = plt.contour(win_grd_e, th_grd_e,BA,
                         nof_levels,colors = 'k')
clabel = plt.clabel(contour, fmt = '%2.2f',colors = 'k', fontsize=10)
cpf = plt.contourf(win_grd_e, th_grd_e, BA,nof_levels,vmax = vmax,vmin = vmin,cmap ='seismic',alpha = 0.7)
cbar = plt.colorbar(cpf)
# cbar_n.solids.set(alpha=0.2)
max_BA = BA.max()
indices = np.where(BA == max_BA)
plt.scatter(win_arr_e[indices[1]],th_arr_e[indices[0]],marker = 'x',color = 'k')
print(win_arr_e[indices[1]],th_arr_e[indices[0]])
cbar.set_label('Normalized Accuracy')
plt.title('Distribution of Balanced Accuracy of MWA')
plt.xlabel('DTA_duration (s)')
plt.ylabel('Threshold')
plt.tight_layout()

plt.show()
              
