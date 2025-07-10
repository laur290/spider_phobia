import os
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





