import os
import numpy as np
from sklearn.cluster import KMeans
import neurokit2 as nk
import pandas as pd
import matplotlib.pyplot as plt

#defining functions that extract the data from each type of txt file into a data frame

def load_deflection(path):
    df=pd.read_csv(
               path,
               sep="\t", 
               header=None,  
               names=["Deflection", "Timestamp", "Drop"])
    df=df.drop(columns='Drop') # to be ignored
    df["Timestamp"] = pd.to_datetime(df["Timestamp"]) # clearer time moments
    return df

def load_voltage(path):
    df=pd.read_csv(
               path,
               sep="\t", 
               header=None,  
               names=["Voltage", "Timestamp", "Drop"])
    df=df.drop(columns='Drop') # to be ignored
    df["Timestamp"] = pd.to_datetime(df["Timestamp"]) # clearer time moments
    return df
    
def load_conductance(path):
    df=pd.read_csv(
               path,
               sep="\t", 
               header=None,  
               names=["Conductance", "Timestamp", "Drop"])
    df=df.drop(columns='Drop') # to be ignored
    df["Timestamp"] = pd.to_datetime(df["Timestamp"]) # clearer time moments
    return df
    
# data extraction

root_folder=r"D:\erasmus\work"
patient_data = {}

for patient_folder in os.listdir(root_folder):
    FOLDER_PATH = os.path.join(root_folder, patient_folder)
    if not os.path.isdir(FOLDER_PATH):
        continue
    
    # initialize sub‑dictionary
    patient_data[patient_folder] = {}
    
    # breath rate file
    br_file = os.path.join(FOLDER_PATH, "BitalinoBR.txt")
    if os.path.isfile(br_file):
        patient_data[patient_folder]["breath"] = load_deflection(br_file)
    
    # ECG file
    ecg_file = os.path.join(FOLDER_PATH, "BitalinoECG.txt")
    if os.path.isfile(ecg_file):
        patient_data[patient_folder]["ecg"] = load_voltage(ecg_file)
    
    # GSR file
    gsr_file = os.path.join(FOLDER_PATH, "BitalinoGSR.txt")
    if os.path.isfile(gsr_file):
        patient_data[patient_folder]["gsr"] = load_conductance(gsr_file)
        
# data processing using neurokit2

processed_data={}

for patient_folder in patient_data:
    processed_data[patient_folder]={}
    
    #breath rate
    signal1=patient_data[patient_folder]["breath"]
    if signal1 is not None:
        rsp_signals, rsp_info = nk.rsp_process(signal1["Deflection"],
                                               sampling_rate=100)
        processed_data[patient_folder]["breath"]={"signals": rsp_signals, "info":rsp_info}
     
    #ECG
    signal2=patient_data[patient_folder]["ecg"]
    if signal2 is not None:
        ecg_signals, ecg_info=nk.ecg_process(signal2["Voltage"],
                                             sampling_rate=100)
        processed_data[patient_folder]['ecg']={"signals":ecg_signals, "info":ecg_info}

    #GSR
    signal3=patient_data[patient_folder]['gsr'] 
    if signal3 is not None:
        gsr_signals, gsr_info=nk.eda_process(signal3["Conductance"],
                                             sampling_rate=100)
        processed_data[patient_folder]['gsr']={"signals": gsr_signals, "info": gsr_info}

# sampling rate is 100 since that is the value of the recording frequency (in Hz) of the Bitalino devices 

# EXPORT TO CSV

for pid, metrics in processed_data.items():
    # build a DataFrame with Time, BR, HR
    df_b = metrics["breath"]["signals"][["RSP_Rate"]].rename(columns={"RSP_Rate":"BR"})
    df_h = metrics["ecg"]["signals"][["ECG_Rate"]].rename(columns={"ECG_Rate":"HR"})
    
    # assume index is sample number at 100 Hz
    df_b["Time"] = df_b.index // 100
    df_h["Time"] = df_h.index // 100
    
    # merge them into one DF
    df = pd.merge(df_b, df_h, on="Time", how="outer")

    # aggregate per second
    df_sec = df.groupby("Time")[["BR","HR"]].max().reset_index()
    
    # creates a destination
    filename   = "BR_HR.csv"
    # export to csv
      
    save_path = os.path.join(root_folder, pid, "BR_HR.csv")

    # write the CSV, without the pandas index column
    df_sec.to_csv(save_path, index=False)
    
# CREATE SEPARATE DATAFRAMES FOR EACH PATIENT'S BR AND HR OVER TIME + PLOT


    
patient_csv_data = {}
for pid in os.listdir(root_folder):
    csv_path = os.path.join(root_folder, pid, "BR_HR.csv")
    if os.path.isfile(csv_path):
        # read it once, store under that patient’s ID
        patient_csv_data[pid] = pd.read_csv(csv_path)
        
# now patient_csv_data is a dict: { "VP01": DataFrame, "VP02": DataFrame, … }

# define a plotting function I can call on demand

def graph(patient_id):
   
    df = patient_csv_data.get(patient_id)
    if df is None:
        raise ValueError(f"No data for patient {patient_id}")
    
    fig, ax = plt.subplots()
    ax.plot(df["Time"], df["BR"], color="red",  label="BR")
    ax.plot(df["Time"], df["HR"], color="blue", label="HR")
    
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Value")
    ax.set_title(f"Breathing & Heart Rate for {patient_id}")
    ax.grid(True)
    ax.legend()
    plt.show()
    
# EDA ANALYSIS 
#(the data have already been extracted by load_conductance function, and processed with the neurokit2 function eda_process )

# 1) exporting to csv
for pid, metrics in processed_data.items():
    gsr_signals = metrics['gsr']['signals'].copy()
    gsr_signals['Time'] = gsr_signals.index // 100
    df_eda_sec = gsr_signals.groupby('Time').agg({
    'EDA_Clean': 'max',
    'SCR_Peaks': 'sum'
    }).rename(columns={'EDA_Clean':'Tonic','SCR_Peaks':'SCR_Count'}).reset_index()
    save_path = os.path.join(root_folder, pid, "EDA.csv")
    df_eda_sec.to_csv(save_path, index=False)
 
        
# 2) creating dictionary from the csv data
patient_csv_data_2={}
for pid in os.listdir(root_folder):
    csv_path_2=os.path.join(root_folder, pid, "EDA.csv")
    if os.path.isfile(csv_path_2):
        patient_csv_data_2[pid]=pd.read_csv(csv_path_2)
    
# 3) defining plotting function for a. EDA signal and b. EDA stress_related feature
def graph_eda(patient_id):
    signals = processed_data[patient_id]['gsr']['signals']
    fig, ax = plt.subplots()
    ax.plot(signals.index / 100, signals['EDA_Clean'], color='green')
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("EDA_clean (μS)")
    ax.set_title(f"Cleaned EDA for {patient_id}")
    plt.show()

def graph_eda_2(patient_id):
    df = patient_csv_data_2.get(patient_id)
    if df is None:
        raise ValueError(f"No data for patient {patient_id}")
    fig, ax1 = plt.subplots()
    ax1.plot(df['Time'], df['Tonic'], color='green', label='Tonic EDA')
    ax1.set_xlabel("Time [s]")
    ax1.set_ylabel("Tonic Level", color='green')
    ax1.legend(loc='upper left')

    ax2 = ax1.twinx()
    ax2.bar(df['Time'], df['SCR_Count'], width=1, alpha=0.4,
            color='purple', label='SCR Count')
    ax2.set_ylabel("SCR Count", color='purple')
    ax2.legend(loc='upper right')

    plt.title(f"EDA Summary for {patient_id}")
    plt.show()
