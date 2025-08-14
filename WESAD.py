import os
import numpy as np
import neurokit2 as nk
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer

wesad_folder = r"D:\erasmus\WESAD"

# -------------
# 1) extraction 
# -------------

# scaling frequencies, in Hz
scf_hr  = 1
scf_eda = 4
scf_bvp = 64

def load_eda(path):
    df = pd.read_csv(path, header=None, names=['Conductance [uS]'])
    df = df.drop([0, 1], axis=0).reset_index(drop=True)
    n = len(df)
    df.insert(0, 'Time[s]', np.arange(n) / scf_eda)
    return df

def load_bvp(path):
    df = pd.read_csv(path, header=None, names=['BVP'])
    df = df.drop([0, 1], axis=0).reset_index(drop=True)
    n = len(df)
    df.insert(0, 'Time[s]', np.arange(n) / scf_bvp)
    return df

def load_ibi(path):
    df = pd.read_csv(path, sep=',', header=None, names=['Time[s]', 'Duration [s]'])
    df = df.drop([0], axis=0).reset_index(drop=True)
    df['Duration [ms]'] = df['Duration [s]'].astype(float) * 1000
    return df

def load_hr(path):
    df = pd.read_csv(path, header=None, names=['HR [bpm]'])
    df = df.drop([0, 1], axis=0).reset_index(drop=True)
    n = len(df)
    df.insert(0, 'Time[s]', np.arange(n) / scf_hr)
    return df

patient_data = {}
for pid in os.listdir(wesad_folder):
    folder = os.path.join(wesad_folder, pid)
    if not os.path.isdir(folder):
        continue
    patient_data[pid] = {}
    eda_f = os.path.join(folder, "EDA.csv")
    bvp_f = os.path.join(folder, "BVP.csv")
    ibi_f = os.path.join(folder, "IBI.csv")
    hr_f  = os.path.join(folder, "HR.csv")
    
    if os.path.isfile(eda_f):
        patient_data[pid]["eda"] = load_eda(eda_f)
        
    if os.path.isfile(bvp_f):
        patient_data[pid]["bvp"] = load_bvp(bvp_f)
        
    if os.path.isfile(ibi_f):
        patient_data[pid]["ibi"] = load_ibi(ibi_f)
        
    if os.path.isfile(hr_f):
        patient_data[pid]["hr"] = load_hr(hr_f)


# ---------------------------------------------------
# 2) extracting the necessary parameters + processing
# ---------------------------------------------------


processed_data={}

for pid, metrics in patient_data.items():
    
    processed_data[pid] = {}
    window_size =  60  # 1-minute seconds samples
    
    if "hr" in metrics:
        hr_df=metrics['hr'].copy()
        hr_df["window"]=(hr_df["Time[s]"] // window_size).astype(int)
        hr_means = (
            hr_df
            .groupby("window", as_index=False)["HR [bpm]"]
            .mean()
            .rename(columns={"HR [bpm]": "mean_HR"}) 
            ) #group by + mean calculation
        hr_means["window_start_time_[s]"] = hr_means["window"] * window_size
        processed_data[pid]["HR_mean_per_1min"] = hr_means # parameter 1: mean heart rate per 1-minute sample
        
    if "eda" in metrics:
        eda_df = metrics['eda'].copy()
        eda_signal, eda_info = nk.eda_process(eda_df['Conductance [uS]'], sampling_rate=scf_eda)
        eda_signal['Time[s]']=eda_signal.index/scf_eda
        eda_signal['window']=(eda_signal['Time[s]']//window_size).astype(int)
        
        tonic_means = (
            eda_signal
            .groupby('window', as_index=False)['EDA_Tonic']
            .mean()
            .rename(columns={'EDA_Tonic':"mean_SCL"})
            )
        tonic_means["window_start_time_[s]"] = tonic_means["window"] * window_size
        processed_data[pid]["SCL_means_per_1min"]=tonic_means # parameter 2: mean tonic level per 1-minute sample
        
        phasic_counts=(
            eda_signal
            .groupby('window', as_index=False)['SCR_Peaks']
            .sum(['SCR_Peaks']==1)
            .rename(columns={'SCR_Peaks':'SCR_Count'})
            )
        phasic_counts["window_start_time_[s]"] = tonic_means["window"] * window_size
        processed_data[pid]["SCR_Counts_per_1min"]=phasic_counts # parameter 3: SCR peaks number per 1-minute sample
        
        
        '''
    if "bvp" in metrics:
        bvp_df=metrics['bvp'].copy()
        bvp_signal, bvp_info = nk.ppg_process(bvp_df['BVP'], sampling_rate=scf_bvp)
        bvp_signal['Time[s]']=bvp_signal.index/scf_bvp
        bvp_signal['window']=(bvp_signal['Time[s]']//window_size).astype(int)
        ...
        processed_data[pid]['bvp_signal']=bvp_signal
        '''

    if "ibi" in metrics:
        ibi_df = metrics["ibi"].copy()
        ibi_df["window"] = (ibi_df["Time[s]"] // window_size).astype(int)

        window_hrv = []
        for w, sub in ibi_df.groupby("window"):
            durations = sub["Duration [ms]"].astype(float).values
            R_peaks = nk.intervals_to_peaks(durations, sampling_rate=scf_bvp)
            hrv = nk.hrv_time(R_peaks, sampling_rate=scf_bvp)
            window_hrv.append({
                "window":w,
                "HRV_RMSSD":hrv["HRV_RMSSD"].iloc[0], # parameter 4: RMSSD per 1-minute sample
                "HRV_pNN50":hrv["HRV_pNN50"].iloc[0] # parameter 5: pNN50 per 1-minute sample
                                })

        window_hrv = pd.DataFrame(window_hrv)
        window_hrv["window_start_time_[s]"] = window_hrv["window"] * window_size
        processed_data[pid]["HRV_per_1min"] = window_hrv

# dropping unnecessary data (affects stress identification loop)
for pid, metrics in processed_data.items():
    for key in ["HR_mean_per_1min", "HRV_per_1min", "SCL_means_per_1min", "SCR_Counts_per_1min"]:
        if key in metrics and not metrics[key].empty:
            metrics[key]=metrics[key].iloc[25:].reset_index(drop=True)

    

# ---------------------
# 3) plotting functions
# ---------------------

def plot_pNN50_RMSSD(): # for checking that the two indices are proportional
    fig, ax = plt.subplots()
    
    for pid, pdata in processed_data.items():
        df_h = pdata.get("HRV_per_1min")
        if df_h is None or df_h.empty:
            continue
        X = df_h["HRV_RMSSD"].values.reshape(-1, 1)
        y = df_h["HRV_pNN50"].values
        mask = np.isfinite(X.flatten()) & np.isfinite(y)
        X, y = X[mask], y[mask]
        if len(X) < 2: # a regression line needs at least 2 point; caution measure
            continue
        ax.scatter(X.flatten(), y, label=pid)
        pipeline = make_pipeline(
            SimpleImputer(strategy="mean"),
            LinearRegression()
        )
        
        pipeline.fit(X, y)
        x_grid = np.linspace(X.min(), X.max(), 100).reshape(-1, 1)
        y_pred = pipeline.predict(x_grid)
        
        ax.plot(x_grid.flatten(), y_pred, label=f"{pid} fit")
    
    ax.set_xlabel("HRV_RMSSD [ms]")
    ax.set_ylabel("HRV_pNN50 [%]")
    ax.set_title("RMSSD vs pNN50 per 1-min Window")
    ax.grid(True)
    ax.legend(title="Patient ID", bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.tight_layout()
    plt.show()

def plot_eda(pid):
    df = processed_data.get(pid)
    if df is None:
        raise ValueError(f"No EDA summary for {pid}")
    fig, ax1 = plt.subplots()
    ax1.plot(df['SCL_means_per_1min']['window'], df['SCL_means_per_1min']["mean_SCL"], color='green', label='Tonic EDA')
    ax1.set_xlabel("Time [s]")
    ax1.set_ylabel("SCL [μS]", color='green')
    ax2 = ax1.twinx()
    ax2.bar(df['SCR_Counts_per_1min']["window"], df['SCR_Counts_per_1min']["SCR_Count"], width=1, alpha=0.4,
            color='purple', label='SCR Count')
    ax2.set_ylabel("Phasic EDA", color='purple')
    ax1.legend(loc='upper left')
    ax2.legend(loc='upper right')
    plt.title(f"EDA Summary for {pid}")
    plt.show()

def plot_hr(pid):
    df = processed_data.get(pid)
    if df is None:
        raise ValueError(f"No HR data for {pid}")
    fig, ax = plt.subplots()
    ax.plot(df['HR_mean_per_1min']["window"], df['HR_mean_per_1min']["mean_HR"], color="blue")
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Rate (bpm)")
    ax.set_title(f" HR over Time for {pid}")
    ax.grid(True)
    ax.legend()
    plt.show()


# ------------------------------------
# 4) classification based on thresolds 
# ------------------------------------



THRESHOLDS={
    "mean_HR": 82,
    "RMSSD": 35 ,
    "pNN50": 15,
    "SCR": 3,
}

def count_violations(pid):
    violations = 0
    win_hr=win_rmssd = win_pnn50 = win_scr = win_scl = None
    
    df_hrv=processed_data[pid].get("HRV_per_1min")
    df_scl=processed_data[pid].get("SCL_means_per_1min")
    df_scr=processed_data[pid].get("SCR_Counts_per_1min")
    df_hr=processed_data[pid].get("HR_mean_per_1min")

    if df_hr is not None and not df_hr.empty:
        max_hr=pd.to_numeric(df_hr["mean_HR"], errors="coerce").max()
        idx_hr=df_hr["mean_HR"].idxmax()
        if pd.notna(idx_hr):
            win_hr=df_hr.loc[idx_hr, "window_start_time_[s]"]
        if pd.notna(max_hr) and max_hr>THRESHOLDS["mean_HR"]:
            violations += 1

    if df_hrv is not None and not df_hrv.empty:
        rmssd_series=pd.to_numeric(df_hrv["HRV_RMSSD"], errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
        if not rmssd_series.empty:
            min_rmssd=rmssd_series.min()
            idx_rmssd=rmssd_series.idxmin()
            win_rmssd=df_hrv.loc[idx_rmssd, "window_start_time_[s]"]
            if min_rmssd<THRESHOLDS["RMSSD"]:
                violations += 1
        else:
            min_rmssd=np.nan

        pnn_series=pd.to_numeric(df_hrv["HRV_pNN50"], errors="coerce").replace([np.inf, -np.inf], np.nan)
        pnn_series=pnn_series[pnn_series > 0].dropna()
        if not pnn_series.empty:
            min_pnn50=pnn_series.min()
            idx_pnn=pnn_series.idxmin()
            win_pnn50=df_hrv.loc[idx_pnn, "window_start_time_[s]"]
            if min_pnn50<THRESHOLDS["pNN50"]:
                violations += 1
        else:
            min_pnn50=np.nan

    if df_scr is not None and not df_scr.empty:
        scr_series=pd.to_numeric(df_scr["SCR_Count"], errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
        if not scr_series.empty:
            max_scr=scr_series.max()
            idx_scr=scr_series.idxmax()
            win_scr=df_scr.loc[idx_scr, "window_start_time_[s]"]
            if max_scr>THRESHOLDS["SCR"]:
                violations += 1
        else:
            max_scr=np.nan

    if df_scl is not None and not df_scl.empty:
        scl_series = pd.to_numeric(df_scl["mean_SCL"], errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
        if not scl_series.empty:
            max_scl=scl_series.max()
            idx_scl=scl_series.idxmax()
            win_scl=df_scl.loc[idx_scl, "window_start_time_[s]"]
            global_mean_scl=scl_series.mean()
            if max_scl>1.48 * global_mean_scl:
                violations += 1
        else:
            max_scl=np.nan
            global_mean_scl=np.nan

    if violations >= 3:
        print(f"critical windows for patient {pid} are the following:  HR: {win_hr}, RMSSD: {win_rmssd}, PNN50: {win_pnn50}, SCR: {win_scr}, SCL: {win_scl}")
    return violations

        
        
        
        
        