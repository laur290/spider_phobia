import os
import numpy as np
import neurokit2 as nk
import pandas as pd
import matplotlib.pyplot as plt

# legend
# patient_data = dictionary with unprocessed data
# processed_data = dictionary with processed data (neurokit2)
# br = for all .txt respiration files
# ecg = for all .txt heart activity files
# eda = for all .txt electrodermal activity files
# patient_csv_data = dictionary containing HR+BR imported from .csv
# patient_csv_data_2 = dictionary containing EDA imported from .csv

root_folder = r"D:\erasmus\work"

# —————————————————————————————
# 1) DATA EXTRACTION
# —————————————————————————————

#1a) dedicated functions 

def load_deflection(path):
    df = pd.read_csv(path, sep="\t", header=None,
                     names=["Deflection","Timestamp","Drop"])
    df = df.drop(columns="Drop")
    df["Timestamp"] = pd.to_datetime(df["Timestamp"])
    return df

def load_voltage(path):
    df = pd.read_csv(path, sep="\t", header=None,
                     names=["Voltage","Timestamp","Drop"])
    df = df.drop(columns="Drop")
    df["Timestamp"] = pd.to_datetime(df["Timestamp"])
    return df

def load_conductance(path):
    df = pd.read_csv(path, sep="\t", header=None,
                     names=["Conductance","Timestamp","Drop"])
    df = df.drop(columns="Drop")
    df["Timestamp"] = pd.to_datetime(df["Timestamp"])
    return df

#1b) dictionary with raw data
patient_data = {}
for pid in os.listdir(root_folder):
    folder = os.path.join(root_folder, pid)
    if not os.path.isdir(folder):
        continue
    patient_data[pid] = {}
    br = os.path.join(folder, "BitalinoBR.txt")
    ecg = os.path.join(folder, "BitalinoECG.txt")
    eda = os.path.join(folder, "BitalinoGSR.txt")
    if os.path.isfile(br):
        patient_data[pid]["breath"] = load_deflection(br)
    if os.path.isfile(ecg):
        patient_data[pid]["ecg"] = load_voltage(ecg)
    if os.path.isfile(eda):
        patient_data[pid]["eda"] = load_conductance(eda)

# —————————————————————————————
# 2) PROCESSING + CSV EXPORT
# —————————————————————————————

#2a) dictionary with processed data
processed_data = {} 

for pid, metrics in patient_data.items():
    processed_data[pid] = {}
    
    #Respiration (BR)
    if "breath" in metrics:
        rsp_s, rsp_i = nk.rsp_process(metrics["breath"]["Deflection"], sampling_rate=100)
        processed_data[pid]["breath"] = rsp_s  # keep only signals
        # Export per-second CSV
        df_b = rsp_s[["RSP_Rate"]].rename(columns={"RSP_Rate":"BR"})
        df_b["Time"] = df_b.index // 100
    else:
        df_b = pd.DataFrame(columns=["Time","BR"])
    
    #ECG (HR)
    if "ecg" in metrics:
        ecg_s, ecg_i = nk.ecg_process(metrics["ecg"]["Voltage"], sampling_rate=100)
        processed_data[pid]["ecg"] = {
            "signals": ecg_s,
            "info":    ecg_i
            }
        df_h = ecg_s[["ECG_Rate"]].rename(columns={"ECG_Rate":"HR"})
        df_h["Time"] = df_h.index // 100
    else:
        df_h = pd.DataFrame(columns=["Time","HR"])
    
    #merge & aggregate BR+HR
    df = pd.merge(df_b, df_h, on="Time", how="outer").fillna(method="ffill")
    df_sec = df.groupby("Time")[["BR","HR"]].max().reset_index()
    df_sec.to_csv(os.path.join(root_folder, pid, "BR_HR.csv"), index=False)

    #EDA
    if "eda" in metrics:
        eda_s, eda_i = nk.eda_process(metrics["eda"]["Conductance"], sampling_rate=100)
        processed_data[pid]["eda"] = eda_s
        eda_s["Time"] = eda_s.index // 100
        df_eda = (
            eda_s.groupby("Time")
                 .agg(Tonic=("EDA_Clean","max"), SCR_Count=("SCR_Peaks","sum"))
                 .reset_index()
        )
        df_eda.to_csv(os.path.join(root_folder, pid, "EDA.csv"), index=False)

# —————————————————————————————
# 3) LOAD CSVS FOR PLOTTING
# —————————————————————————————

#BR+HR
patient_csv_data = {
    pid: pd.read_csv(os.path.join(root_folder, pid, "BR_HR.csv"))
    for pid in patient_data
    if os.path.isfile(os.path.join(root_folder, pid, "BR_HR.csv"))
}

#EDA
patient_csv_data_2 = {
    pid: pd.read_csv(os.path.join(root_folder, pid, "EDA.csv"))
    for pid in patient_data
    if os.path.isfile(os.path.join(root_folder, pid, "EDA.csv"))
}

# —————————————————————————————
# 4) PLOTTING FUNCTIONS
# —————————————————————————————

def plot_br_hr(pid1):
    df = patient_csv_data.get(pid)
    if df is None:
        raise ValueError(f"No BR/HR data for {pid}")
    fig, ax = plt.subplots()
    ax.plot(df["Time"], df["BR"], color="blue",  label="Breathing Rate [resp/min]")
    ax.plot(df["Time"], df["HR"], color="red",  label="Heart Rate [bpm]")
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Rate")
    ax.set_title(f"BR & HR over Time for {pid}")
    ax.grid(True)
    ax.legend()
    plt.show()

def plot_eda(pid):
    df = patient_csv_data_2.get(pid)
    if df is None:
        raise ValueError(f"No EDA summary for {pid}")
    fig, ax1 = plt.subplots()
    ax1.plot(df["Time"], df["Tonic"], color='green', label='Tonic SCL')
    ax1.set_xlabel("Time [s]")
    ax1.set_ylabel("SCL [μS]", color='green')
    ax2 = ax1.twinx()
    ax2.bar(df["Time"], df["SCR_Count"], width=1, alpha=0.4,
            color='purple', label='SCR Count')
    ax2.set_ylabel("SCR Count", color='purple')
    ax1.legend(loc='upper left')
    ax2.legend(loc='upper right')
    plt.title(f"EDA Summary for {pid}")
    plt.show()

# —————————————————————————————
# 5) ADVANCED ECG PROCESSING
# —————————————————————————————

fs = 100 # sampling rate
interval_summary=[] #list of dictionaries, one for every patient, regarding ECG intervals
for pid, metrics in processed_data.items():
    row={"PID":pid}
    if "ecg" in metrics:
        ecg_s = metrics["ecg"]["signals"]   
        ecg_i = metrics["ecg"]["info"] 
        r_onsets=np.array(ecg_i['ECG_R_Onsets'])
        r_offsets=np.array(ecg_i['ECG_R_Offsets'])
        valid_mask = (~np.isnan(r_onsets)) & (~np.isnan(r_offsets))
        r_onsets  = r_onsets[valid_mask].astype(int)
        r_offsets = r_offsets[valid_mask].astype(int)
        rr_samples=np.diff(r_onsets)
        rr_secs=rr_samples/fs
        row["RR_mean_s"] = rr_secs.mean() #average RR interval duration for each patient
        
        amps = []
        for onset, offset in zip(r_onsets, r_offsets):
            segment = ecg_s["ECG_Clean"].iloc[onset:offset]
            amps.append(segment.max())
        row["R_amp_mean_uv"] = np.nanmean(amps) #average R peak amplitude for each patient
        
        ecg_d, delineate_i = nk.ecg_delineate(
            ecg_s["ECG_Clean"],
            rpeaks=r_onsets,
            sampling_rate=fs
        )
        q_peaks  = np.array(delineate_i["ECG_Q_Peaks"])
        t_onsets = np.array(delineate_i["ECG_T_Onsets"])
        valid_qt = (~np.isnan(q_peaks)) & (~np.isnan(t_onsets))
        qt_samps = (t_onsets[valid_qt] - q_peaks[valid_qt]) / fs
        row["QT_mean_s"] = np.nanmean(qt_samps) #average QT interval duration for each patient
        
        s_offsets = np.array(delineate_i["ECG_S_Peaks"])
        valid_st  = (~np.isnan(s_offsets)) & (~np.isnan(t_onsets))
        st_samps = (t_onsets[valid_st] - s_offsets[valid_st]) / fs
        row["ST_mean_s"] = np.nanmean(st_samps) #average ST interval duration for each patient
    interval_summary.append(row)
df_interval_summary=pd.DataFrame(interval_summary).set_index('PID')
