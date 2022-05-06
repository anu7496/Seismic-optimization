# -*- coding: utf- -*-
"""
Created on Sat May 16 17:59:41 2020

@author: Anubrata Roy
"""
from __future__ import division
from pymongo import MongoClient
import numpy as np
from scipy.signal import hilbert,butter,filtfilt
import time
from joblib import Parallel, delayed
import multiprocessing
import winsound
#%%
alpha = 3
lowcut = 32

alphas = [6.1] #optimized threshold from gridsearch
dta_durations = [2.9] #optimized leading window duration from gridsearch

def sliding_window(arr, window_len = 4, shift_index = 2, copy = False):
    sh = (arr.size - window_len + 1, window_len)
    st = arr.strides * 2
    view = np.lib.stride_tricks.as_strided(arr, strides = st, shape = sh)[0::shift_index]
    if copy:
        return view.copy()
    else:
        return view
    
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

def rescale(arr, factor=2):
    n = len(arr)
    return np.interp(np.linspace(0, n, factor*n+1), np.arange(n), arr)
#%%
client = MongoClient()

db_eq=client.knet_1530
header_eq = db_eq.header

db_op = client.mwa_error_eq_all
query = {"$and":
                    [ {"Magnitude": {'$gt':3,"$lt":8.1}},
                          {"PGA_gal" :{'$gt':1}}

                      ]}

selectedEvents_eq = header_eq.find(query)
nof_selection_eq = selectedEvents_eq.count()
record_list_eq = list(zip(list(range(nof_selection_eq)), selectedEvents_eq))
client.close()

db_noise = client.DMRC_till_201902
header_noise = db_noise.header
db_op_noise = client.mwa_error_noise_all
selectedEvents_noise = header_noise.find()
nof_selection_noise = selectedEvents_noise.count()
record_list_noise = list(zip(list(range(nof_selection_noise)), selectedEvents_noise))
client.close()
#%%
def performance_eq(c,nof_selection_eq,alphas):
    count = c[0]
    x = c[1]
    client1 = MongoClient()
    db_eq=client1.knet_1530
    accelerogram_eq = db_eq.accelerogram
    db_op = client1.mwa_error_eq_all
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
    time.sleep(0.1)
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
                tg_p = tg_p_phase + q
                if  tg_p > x["pIndex"] - 1*sps and tg_p < pga_index:
                    if x["p"] and tg_p <(x["pIndex"] + 4.5*sps) and tg_p <x["sIndex"] :
                        error_p_mwa.append([dt,(round(alfa, 2)),(tg_p-x["pIndex"])/sps,(tg_p_phase-x["pIndex"])/sps])
                    elif x["p"] and ((tg_p >= (x["pIndex"] + 4.5*sps)) or tg_p >= x["sIndex"]) :
                        error_p_mwa.append([dt,(round(alfa, 2)),np.nan,np.nan]) 
                        error_s_mwa.append([dt,(round(alfa, 2)),(tg_p-x["sIndex"])/sps,(tg_p_phase-x["sIndex"])/sps])
                    else:
                        error_s_mwa.append([dt,(round(alfa, 2)),(tg_p-x["sIndex"])/sps,(tg_p_phase-x["sIndex"])/sps])
                else:
                    if x["p"]:
                        error_p_mwa.append([dt,(round(alfa, 2)),np.nan,np.nan])
                    else:
                        error_s_mwa.append([dt,(round(alfa, 2)),np.nan,np.nan])
            else:
                if x["p"]:
                    error_p_mwa.append([dt,(round(alfa, 2)),np.nan,np.nan])
                else:
                    error_s_mwa.append([dt,(round(alfa, 2)),np.nan,np.nan])


    op = {}
    for errs in error_p_mwa :
        op["_id"] = f_name+'_'+str(errs[0])+'_'+str(errs[1])
        op["Magnitude"] = x["Magnitude"]
        op["Distance_km"] = x["Distance"]
        op["Depth_km"] = x["Depth_km"]
        op["PGA_gal"] = x["PGA_gal"]
        op["PGA_pos"] = int(pga_index)
        op["Th"] = errs[1]
        op['Error'] = errs[2] #delay in detection
        op['Error_p_phase'] = errs[3] #error in p phase picking
        op["DTA_len"] = errs[0]
        if x["p"]:
            op["p-s_time"] = (x["sIndex"]-x["pIndex"])/sps
        db_op["error_p"].insert_one(op)


    op = {}
    for errs in error_s_mwa :
        op["_id"] = f_name+'_'+str(errs[0])+'_'+str(errs[1])
        op["Magnitude"] = x["Magnitude"]
        op["Distance_km"] = x["Distance"]
        op["Depth_km"] = x["Depth_km"]
        op["PGA_gal"] = x["PGA_gal"]
        op["PGA_pos"] = int(pga_index)
        op["Th"] = errs[1]
        op['Error'] = errs[2] #delay in detection
        op['Error_s_phase'] = errs[3] #error in s phase picking
        op["DTA_len"] = errs[0]
        db_op["error_s"].insert_one(op)

    client1.close()
    if(count%500 == 0):
        print(str(count)+"/"+str(nof_selection_eq))

#%%
db_op["error_p"].delete_many({})
db_op["error_s"].delete_many({})
num_cores = multiprocessing.cpu_count()
results = Parallel(n_jobs= int(num_cores*0.9),prefer = 'processes',
                      verbose=1
                      #backend="multiprocessing" #use multiprocessing on unix systems specially
                      )(delayed(performance_eq)(
                        c,
                        nof_selection_eq,
                        alphas
                        )
                        for c in record_list_eq)
#%%
def performance_noise(c,nof_selection_eq,alphas):
    count = c[0]
    x = c[1]
    client1 = MongoClient()
    db_noise = client1.DMRC_till_201902
    accelerogram_noise = db_noise.accelerogram
    db_op_noise = client1.mwa_error_noise_all
    
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
        u_tg = np.ones(u_f.shape)
        u_abs = np.abs(u_f)
    
        global_max_acc = u_abs.max()
        indices = np.where(u_abs == global_max_acc)
        pga_index = indices[0][0]
        error_noise_mwa = []
        time.sleep(0.1)
        for dt in dta_durations:
            bt = 4*dt/6
            at = 0.5*dt
            tot_t = bt+dt
            dur = int(sps * tot_t)
            m = int(sps * bt)
            n = int (sps * at)
            q = int (sps* dt)
            win = sliding_window(u_f,dur-1,1)
            bta_win = win[:,0:m]
            ata_win = win[:,m-1:m+n-1]
            dta_win = win[:,m-1:m+n+q-1]
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
                    error_noise_mwa.append([dt,(round(alfa, 2)),(tg_p-x["pIndex"])/sps])
                else:
                    error_noise_mwa.append([dt,(round(alfa, 2)),np.nan])
    
        
        op = {}
        for errs in error_noise_mwa :
            op["_id"] = f_name+'_'+str(errs[0])+'_'+str(errs[1])
            op["PGA_gal"] = x["PGA_gal"]
            op["PGA_pos"] = int(pga_index)
            op["Th"] = errs[1]
            op['Error'] = errs[2]
            op["DTA_len"] = errs[0]
            db_op_noise["error_p"].insert_one(op)
        client1.close()    
    except:
        print("error\n\n",f_name)
        winsound.Beep(1000,200)
        db_op_noise["error_file"].insert_one({"_id":f_name,"error": False})
        
#%%
db_op_noise["error_p"].delete_many({})
db_op_noise["error_file"].delete_many({})
num_cores = multiprocessing.cpu_count()
results = Parallel(n_jobs=int(num_cores*0.9),prefer = 'processes',
                      verbose=1,
                      #backend="multiprocessing" #use multiprocessing on unix systems specially
                      )(delayed(performance_noise)(
                        c,
                        nof_selection_noise,
                        alphas
                        )
                        for c in record_list_noise)