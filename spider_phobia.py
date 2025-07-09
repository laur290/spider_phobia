import os
from sklearn.cluster import KMeans
import neurokit2 as nk
import pandas as pd

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
    

root_folder="D:\erasmus\work"
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

