# -*- coding: utf-8 -*-
"""
Created on Sat May 16 18:23:59 2020

@author: Anubrata Roy
"""
from pymongo import MongoClient
import numpy as np
import time
from scipy.signal import butter,filtfilt
from joblib import Parallel, delayed
import multiprocessing
import winsound
#%%
lowcut = 32
SNR_alphas = [8.11] #optimized threshold from gridsearch
durations = [0.54] #optimized leading window duration from gridsearch

def LPF(data, lowcut, fs, order=5):
    if data.ndim <2:  #data should be in [channel,data] matrix format
        data = data.reshape(1,-1)  # for filtering single channel data
    filtered_signal = []
    nyq = 0.5 * fs
    low = lowcut / nyq
    b, a = butter(order, low, btype='low')
    for channel in data:      
        output = filtfilt(b, a, channel)
        filtered_signal.append(output)
    return np.array(filtered_signal)

def sliding_window(arr, window_len = 4, shift_index = 2, copy = False):
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
client = MongoClient()
db_eq=client.knet_1530
header_eq = db_eq.header
db_op = client.stalta_error_eq_all
query = {"$and":
                    [ {"Magnitude": {'$gt':3,"$lt":8.1}},
                          {"PGA_gal" :{'$gt':1}}

                      ]}
selectedEvents_eq = header_eq.find(query)
nof_selection_eq = selectedEvents_eq.count()
record_list_eq = list(zip(list(range(nof_selection_eq)), selectedEvents_eq))
client.close()

db_noise=client.DMRC_till_201902
header_noise = db_noise.header
db_op_noise = client.stalta_error_noise_all
selectedEvents_noise = header_noise.find()
nof_selection_noise = selectedEvents_noise.count()
record_list_noise = list(zip(list(range(nof_selection_noise)), selectedEvents_noise))
client.close()
#%%
def performance_eq(c,nof_selection,SNR_alphas):
    count = c[0]
    x = c[1]
    client1 = MongoClient()
    db_eq=client1.knet_1530
    accelerogram_eq = db_eq.accelerogram
    db_op = client1.stalta_error_eq_all
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
    error_p_stalta = []
    error_s_stalta = []
    time.sleep(0.1)
    for st in durations:
        lt = st*5
        long_time = int(sps * lt)
        short_time = int(sps * st)
        dur = long_time+short_time
        win = sliding_window(acc_f,dur-1,stride)
        sta_win = win[:,-short_time:]
        lta_win = win[:,:long_time]
        sta = np.mean(sta_win,axis = 1)
        lta = np.mean(lta_win,axis = 1)
        SNR_alpha = sta/lta
        fact = int(np.ceil(len(acc_f)/len(sta)))
        SNR_mod = rescale(SNR_alpha, factor = fact)
        tg_p = 0
        for alfa in SNR_alphas:
            indx = [index for index,value in enumerate(SNR_mod[:nof_datapoints]) if value > alfa]
            if len(indx)> 0:
                tg_p_phase = indx[0]+long_time-stride//2# minimum delay to detect p is window length
                tg_p = tg_p_phase+short_time
                if  tg_p > x["pIndex"] - 1*sps and tg_p < pga_index:
                    if x["p"] and tg_p <(x["pIndex"] + 4.5*sps) and tg_p <x["sIndex"] :
                        error_p_stalta.append([st,(round(alfa, 2)),(tg_p-x["pIndex"])/sps,(tg_p_phase-x["pIndex"])/sps])
                    elif x["p"] and ((tg_p >= (x["pIndex"] + 4.5*sps)) or tg_p >= x["sIndex"]) :
                        error_p_stalta.append([st,(round(alfa, 2)),np.nan,np.nan]) 
                        error_s_stalta.append([st,(round(alfa, 2)),(tg_p-x["sIndex"])/sps,(tg_p_phase-x["sIndex"])/sps])
                    else:
                        error_s_stalta.append([st,(round(alfa, 2)),(tg_p-x["sIndex"])/sps,(tg_p_phase-x["sIndex"])/sps])
                else:
                    if x["p"]:
                        error_p_stalta.append([st,(round(alfa, 2)),np.nan,np.nan])
                    else:
                        error_s_stalta.append([st,(round(alfa, 2)),np.nan,np.nan])
            else:
                if x["p"]:
                    error_p_stalta.append([st,(round(alfa, 2)),np.nan,np.nan])
                else:
                    error_s_stalta.append([st,(round(alfa, 2)),np.nan,np.nan])


    op = {}
    for errs in error_p_stalta :
        op["_id"] = f_name+'_'+str(errs[0])+'_'+str(errs[1])
        op["Magnitude"] = x["Magnitude"]
        op["Distance_km"] = x["Distance"]
        op["Depth_km"] = x["Depth_km"]
        op["PGA_gal"] = x["PGA_gal"]
        op["PGA_pos"] = int(pga_index)
        op["Th"] = errs[1]
        op['Error'] = errs[2] #delay in detection
        op['Error_p_phase'] = errs[3] #error in p phase picking
        op["STA_len"] = errs[0]
        if x["p"]:
            op["p-s_time"] = (x["sIndex"]-x["pIndex"])/sps
        db_op["error_p"].insert_one(op)


    op = {}
    for errs in error_s_stalta :
        op["_id"] = f_name+'_'+str(errs[0])+'_'+str(errs[1])
        op["Magnitude"] = x["Magnitude"]
        op["Distance_km"] = x["Distance"]
        op["Depth_km"] = x["Depth_km"]
        op["PGA_gal"] = x["PGA_gal"]
        op["PGA_pos"] = int(pga_index)
        op["Th"] = errs[1]
        op['Error'] = errs[2] #delay in detection
        op['Error_s_phase'] = errs[3] #error in s phase picking
        op["STA_len"] = errs[0]
        db_op["error_s"].insert_one(op)

    client1.close()
    # time.sleep(0.1)
    if(count%500 == 0):
        print(str(count)+"/"+str(nof_selection))#,end='\r'

#%%
db_op["error_p"].delete_many({})
db_op["error_s"].delete_many({})
num_cores = multiprocessing.cpu_count()
results = Parallel(n_jobs= int(num_cores*0.8),prefer = 'processes',
                      verbose=1
                      #backend="multiprocessing" #use multiprocessing on unix systems specially
                      )(delayed(performance_eq)(
                        c,
                        nof_selection_eq,
                        SNR_alphas
                        )
                        for c in record_list_eq)
#%%
def performance_noise(c,nof_selection,SNR_alphas):
    x = c[1]
    client1 = MongoClient()
    db_noise=client1.DMRC_till_201902
    accelerogram_noise = db_noise.accelerogram
    db_op_noise = client1.stalta_error_noise_all

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
        error_noise_stalta = []
        time.sleep(0.1)
        for st in durations:
            lt = st*5
            long_time = int(sps * lt)
            short_time = int(sps * st)
            dur = long_time+short_time
            win = sliding_window(acc_f,dur-1,1)
            sta_win = win[:,-short_time:]
            lta_win = win[:,:long_time]
            sta = np.mean(sta_win,axis = 1)
            lta = np.mean(lta_win,axis = 1)
            SNR_alpha = sta/lta
            fact = int(np.ceil(len(acc_f)/len(sta)))
            SNR_mod = rescale(SNR_alpha, factor = fact)
            tg_p = 0
            for alfa in SNR_alphas:
                indx = [index for index,value in enumerate(SNR_mod[:nof_datapoints]) if value > alfa]
                if len(indx)> 0: 
                    tg_p = indx[0]# minimum delay to detect p is window length
                    error_noise_stalta.append([st,(round(alfa, 2)),(tg_p-x["pIndex"])/sps])
                else:
                    error_noise_stalta.append([st,(round(alfa, 2)),np.nan])

    
        op = {}
        for errs in error_noise_stalta :
            op["_id"] = f_name+'_'+str(errs[0])+'_'+str(errs[1])
            op["PGA_gal"] = x["PGA_gal"]
            op["PGA_pos"] = int(pga_index)
            op["Th"] = errs[1]
            op['Error'] = errs[2] #delay in detection
            op["STA_len"] = errs[0]
            db_op_noise["error_p"].insert_one(op)
        client1.close()
        time.sleep(0.001)
    except Exception as e :
        # print("error\n\n",f_name)
        # print("\n",e)
        winsound.Beep(1000,200)
        db_op_noise["error_file"].insert_one({"_id":f_name,"error": str(e)})
#%%
from joblib import Parallel, delayed
import multiprocessing
 
db_op_noise["error_p"].delete_many({})
db_op_noise["error_file"].delete_many({})

results = Parallel(n_jobs=int(num_cores*0.8),prefer = 'processes',
                      verbose= 1,
                      #backend="multiprocessing" #use multiprocessing on unix systems specially
                      )(delayed(performance_noise)(
                        c,
                        nof_selection_noise,
                        SNR_alphas
                        )
                        for c in record_list_noise)