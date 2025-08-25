import os
import numpy as np
import neurokit2 as nk
import pandas as pd
from pathlib import Path
import pickle 

wesad_folder = r"D:\erasmus\WESAD"

scf_hr  = 1
scf_eda = 4
scf_bvp = 64

OFFSET_SECONDS = 1000
WINDOW_SIZE = 60

THRESHOLDS = {
    "mean_HR": 82,
    "HRV_RMSSD": 46,
    "HRV_pNN50": 26,
    "SCR_Count": 3
}

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

processed_data = {}

for pid, metrics in patient_data.items():
    processed_data[pid] = {}
    window_size = WINDOW_SIZE

    if "hr" in metrics:
        hr_df = metrics['hr'].copy()
        hr_df["window"] = (hr_df["Time[s]"] // window_size).astype(int)
        hr_means = (
            hr_df
            .groupby("window", as_index=False)["HR [bpm]"]
            .mean()
            .rename(columns={"HR [bpm]": "mean_HR"})
        )
        hr_means["window_start_time_[s]"] = hr_means["window"] * window_size
        hr_means.loc[hr_means["window_start_time_[s]"] < OFFSET_SECONDS, "mean_HR"] = 0.0
        processed_data[pid]["HR_mean_per_1min"] = hr_means

    if "eda" in metrics:
        eda_df = metrics['eda'].copy()
        eda_signal, eda_info = nk.eda_process(eda_df['Conductance [uS]'], sampling_rate=scf_eda)
        eda_signal['Time[s]'] = eda_signal.index / scf_eda
        eda_signal['window'] = (eda_signal['Time[s]'] // window_size).astype(int)

        tonic_means = (
            eda_signal
            .groupby('window', as_index=False)['EDA_Tonic']
            .mean()
            .rename(columns={'EDA_Tonic': "mean_SCL"})
        )
        tonic_means["window_start_time_[s]"] = tonic_means["window"] * window_size
        tonic_means.loc[tonic_means["window_start_time_[s]"] < OFFSET_SECONDS, "mean_SCL"] = 0.0
        processed_data[pid]["SCL_means_per_1min"] = tonic_means

        phasic_counts = (
            eda_signal
            .groupby('window', as_index=False)['SCR_Peaks']
            .sum()
            .rename(columns={'SCR_Peaks': 'SCR_Count'})
        )
        phasic_counts["window_start_time_[s]"] = phasic_counts["window"] * window_size
        phasic_counts.loc[phasic_counts["window_start_time_[s]"] < OFFSET_SECONDS, "SCR_Count"] = 0.0
        processed_data[pid]["SCR_Counts_per_1min"] = phasic_counts

    if "ibi" in metrics:
        ibi_df = metrics["ibi"].copy()
        ibi_df["window"] = (ibi_df["Time[s]"] // window_size).astype(int)
        window_hrv = []
        for w, sub in ibi_df.groupby("window"):
            start_time = w * window_size
            durations = sub["Duration [ms]"].astype(float).values
            if len(durations) < 2:
                rmssd_val = np.nan
                pnn_val = np.nan
            else:
                R_peaks = nk.intervals_to_peaks(durations, sampling_rate=scf_bvp)
                hrv = nk.hrv_time(R_peaks, sampling_rate=scf_bvp)
                try:
                    rmssd_val = float(hrv["HRV_RMSSD"].iloc[0])
                except Exception:
                    rmssd_val = np.nan
                try:
                    pnn_val = float(hrv["HRV_pNN50"].iloc[0])
                except Exception:
                    pnn_val = np.nan
            if start_time < OFFSET_SECONDS:
                rmssd_val = 0.0
                pnn_val = 0.0
            window_hrv.append({
                "window": w,
                "HRV_RMSSD": rmssd_val,
                "HRV_pNN50": pnn_val
            })
        if window_hrv:
            window_hrv = pd.DataFrame(window_hrv)
            window_hrv["window_start_time_[s]"] = window_hrv["window"] * window_size
            processed_data[pid]["HRV_per_1min"] = window_hrv
        else:
            processed_data[pid]["HRV_per_1min"] = pd.DataFrame(columns=["window","HRV_RMSSD","HRV_pNN50","window_start_time_[s]"])

def get_true_stress_windows_from_pkl(pkl_path, window_size=WINDOW_SIZE, offset_seconds=OFFSET_SECONDS):
    pkl_path = Path(pkl_path)
    if not pkl_path.exists():
        raise FileNotFoundError(pkl_path)
    with open(pkl_path, "rb") as f:
        data = pickle.load(f, encoding="latin1")
    labels = data.get("label", None)
    ecg = None
    try:
        ecg = data["signal"]["chest"]["ECG"]
    except Exception:
        ecg = None
    if labels is None:
        return set(), []
    FS_ECG = 700.0
    fs_label = FS_ECG if (ecg is not None and labels.shape[0] == ecg.shape[0]) else FS_ECG
    stress_idx = np.where(np.asarray(labels) == 2)[0]
    if stress_idx.size == 0:
        return set(), []
    stress_secs = stress_idx.astype(float) / fs_label - float(offset_seconds)
    window_idxs = np.floor(stress_secs / window_size).astype(int)
    window_idxs = window_idxs[stress_secs >= 0]
    stress_windows = set(window_idxs.tolist())
    segs = []
    if stress_idx.size:
        changes = np.where(np.diff(stress_idx) != 1)[0] + 1
        starts = np.r_[stress_idx[0], stress_idx[changes]] if changes.size > 0 else np.array([stress_idx[0]])
        ends = np.r_[stress_idx[changes - 1], stress_idx[-1]] if changes.size > 0 else np.array([stress_idx[-1]])
        for s_i, e_i in zip(starts, ends):
            s_sec = s_i / fs_label - float(offset_seconds)
            e_sec = e_i / fs_label - float(offset_seconds)
            segs.append((max(0.0, s_sec), max(0.0, e_sec)))
    return stress_windows, segs

def detect_stress_windows_from_processed(pid, processed_data, THRESHOLDS, scl_factor=1.48, min_violations=2, window_size=WINDOW_SIZE):
    if pid not in processed_data:
        return set()
    pdata = processed_data[pid]
    hr_df = pdata.get("HR_mean_per_1min")
    hrv_df = pdata.get("HRV_per_1min")
    scl_df = pdata.get("SCL_means_per_1min")
    scr_df = pdata.get("SCR_Counts_per_1min")
    dfs = []
    if hr_df is not None and not hr_df.empty:
        dfs.append(hr_df[["window", "mean_HR"]].copy())
    if hrv_df is not None and not hrv_df.empty:
        dfs.append(hrv_df[["window", "HRV_RMSSD", "HRV_pNN50"]].copy())
    if scl_df is not None and not scl_df.empty:
        dfs.append(scl_df[["window", "mean_SCL"]].copy())
    if scr_df is not None and not scr_df.empty:
        dfs.append(scr_df[["window", "SCR_Count"]].copy())
    if not dfs:
        return set()
    df = dfs[0].copy()
    for other in dfs[1:]:
        df = df.merge(other, on="window", how="outer")
    df = df.sort_values("window").reset_index(drop=True)
    global_mean_scl = df["mean_SCL"].mean(skipna=True) if "mean_SCL" in df.columns else np.nan
    detected = set()
    for _, row in df.iterrows():
        violations = 0
        w = int(row["window"])
        if "mean_HR" in THRESHOLDS and not pd.isna(row.get("mean_HR")):
            try:
                if float(row["mean_HR"]) > float(THRESHOLDS["mean_HR"]):
                    violations += 1
            except Exception:
                pass
        if "HRV_RMSSD" in THRESHOLDS and not pd.isna(row.get("HRV_RMSSD")):
            try:
                if float(row["HRV_RMSSD"]) < float(THRESHOLDS["HRV_RMSSD"]):
                    violations += 1
            except Exception:
                pass
        if "HRV_pNN50" in THRESHOLDS and not pd.isna(row.get("HRV_pNN50")):
            try:
                p = float(row["HRV_pNN50"])
                if p > 0 and p < float(THRESHOLDS["HRV_pNN50"]):
                    violations += 1
            except Exception:
                pass
        if "SCR_Count" in THRESHOLDS and not pd.isna(row.get("SCR_Count")):
            try:
                if float(row["SCR_Count"]) > float(THRESHOLDS["SCR_Count"]):
                    violations += 1
            except Exception:
                pass
        if "mean_SCL" in row and not pd.isna(row.get("mean_SCL")) and not np.isnan(global_mean_scl):
            try:
                if float(row["mean_SCL"]) > scl_factor * float(global_mean_scl):
                    violations += 1
            except Exception:
                pass
        if violations >= min_violations:
            detected.add(w)
    return detected

def compare_detected_vs_true(
    pid,
    processed_data,
    pkl_path,
    THRESHOLDS,
    scl_factor=1.48,
    min_violations=2,
    window_size=WINDOW_SIZE,
    offset_seconds=OFFSET_SECONDS,
    tolerance_windows=1
):
    true_windows, true_times = get_true_stress_windows_from_pkl(
        pkl_path, window_size=window_size, offset_seconds=offset_seconds
    )
    detected_windows = detect_stress_windows_from_processed(
        pid, processed_data, THRESHOLDS, scl_factor, min_violations, window_size
    )

    pdata = processed_data.get(pid, {})
    all_windows = set()
    for k in ("HR_mean_per_1min", "HRV_per_1min", "SCL_means_per_1min", "SCR_Counts_per_1min"):
        df = pdata.get(k)
        if df is not None and not df.empty and "window" in df.columns:
            all_windows.update(df["window"].dropna().astype(int).unique().tolist())

    T = set(true_windows)
    D = set(detected_windows)

    matched_true = set()
    matched_detected = set()
    for d in D:
        for t in T:
            if abs(int(d) - int(t)) <= tolerance_windows:
                matched_detected.add(d)
                matched_true.add(t)
                break

    TP = len(matched_detected)
    FP = len(D - matched_detected)
    FN = len(T - matched_true)

    total = len(all_windows)
    if total == 0:
        TN = 0
    else:
        TN = total - (TP + FP + FN)
        if TN < 0:
            TN = 0

    TPR = TP / (TP + FN) if (TP + FN) > 0 else 0.0
    FPR = FP / (FP + TN) if (FP + TN) > 0 else 0.0

    summary = {
        "PID": pid,
        "detected_windows": sorted(list(D)),
        "true_windows": sorted(list(T)),
        "TP": TP,
        "FP": FP,
        "FN": FN,
        "TN": TN,
        "TPR": TPR,
        "FPR": FPR,
        "true_times": true_times
    }

    print(f"PID {pid}: TP={TP}, FP={FP}, FN={FN}, TN={TN}, TPR={TPR:.3f}, FPR={FPR:.3f}")
    print("True windows (indices):", sorted(list(T)))
    print("Detected windows (indices):", sorted(list(D)))
    print("True time segments (sec):", true_times)
    return summary


# AUC

# x1, y1 - min_violations=2, S10
# x2, y2 - min_violations=3, S10

x1=[0.603960396039603,
0.584158415841584,
0.574257425742574,
0.554455445544554,
0.544554455445544,
0.534653465346534,
0.534653465346534,
0.524752475247524,
0.524752475247524,
0.524752475247524,
0.524752475247524,
0.524752475247524,
0.514851485148514,
0.514851485148514,
0.504950495049505,
0.504950495049505,
0.485148514851485,
0.475247524752475,
0.465346534653465,
0.465346534653465,
0.455445544554455,
0.455445544554455,
0.455445544554455,
0.435643564356435,
0.435643564356435,
0.415841584158415,
0.415841584158415,
0.405940594059405,
0.386138613861386,
0.366336633663366,
0.356435643564356,
0.336633663366336,
0.326732673267326,
0.306930693069306,
0.277227722772277,
0.267326732673267,
0.267326732673267,
0.267326732673267,
0.237623762376237,
0.217821782178217,
0.217821782178217,
0.188118811881188,
0.188118811881188,
0,
]

y1=[0.846153846153846,
0.846153846153846,
0.846153846153846,
0.846153846153846,
0.846153846153846,
0.769230769230769,
0.769230769230769,
0.769230769230769,
0.769230769230769,
0.769230769230769,
0.769230769230769,
0.692307692307692,
0.692307692307692,
0.615384615384615,
0.538461538461538,
0.538461538461538,
0.538461538461538,
0.461538461538461,
0.461538461538461,
0.461538461538461,
0.461538461538461,
0.461538461538461,
0.461538461538461,
0.461538461538461,
0.461538461538461,
0.461538461538461,
0.461538461538461,
0.384615384615384,
0.384615384615384,
0.384615384615384,
0.384615384615384,
0.384615384615384,
0.384615384615384,
0.384615384615384,
0.384615384615384,
0.307692307692307,
0.23076923076923,
0.153846153846153,
0.153846153846153,
0.153846153846153,
0.0769230769230769,
0.0769230769230769,
0.0769230769230769,
0.0178217821782178,
]

x2=[0.366336633663366,
0.366336633663366,
0.356435643564356,
0.346534653465346,
0.346534653465346,
0.336633663366336,
0.336633663366336,
0.336633663366336,
0.326732673267326,
0.297029702970297,
0.277227722772277,
0.277227722772277,
0.277227722772277,
0.267326732673267,
0.257425742574257,
0.247524752475247,
0.227722772277227,
0.207920792079207,
0.198019801980198,
0.188118811881188,
0.168316831683168,
0.148514851485148,
0.138613861386138,
0.138613861386138,
0.138613861386138,
0.128712871287128,
0.108910891089108,
0.099009900990099,
0.099009900990099,
0.0891089108910891,
0.0792079207920792,
0
]


y2=[0.384615384615384,
0.384615384615384,
0.384615384615384,
0.384615384615384,
0.384615384615384,
0.384615384615384,
0.384615384615384,
0.384615384615384,
0.384615384615384,
0.384615384615384,
0.384615384615384,
0.384615384615384,
0.384615384615384,
0.384615384615384,
0.384615384615384,
0.384615384615384,
0.307692307692307,
0.307692307692307,
0.307692307692307,
0.307692307692307,
0.307692307692307,
0.307692307692307,
0.307692307692307,
0.307692307692307,
0.153846153846153,
0.153846153846153,
0.153846153846153,
0.153846153846153,
0.153846153846153,
0.0769230769230769,
0.0769230769230769,
0.0792079207920792,
]



def compute_and_print_auc_sorted(x1, y1, x2, y2):
    def area(x, y):
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)
        order = np.argsort(x)
        xs = x[order]
        ys = y[order]
        return float(np.trapz(ys, xs))

    a1 = area(x1, y1)
    a2 = area(x2, y2)
    print(f"AUC for min_violations=2 is {a1}")
    print(f"AUC for min_violations=3 is {a2}")
    return a1, a2