import os
import sys
import numpy as np
import neurokit2 as nk
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
import pickle
from pathlib import Path
from sklearn.metrics import roc_auc_score, roc_curve


# -------------
# 0) Settings
# -------------

wesad_folder = r"D:\erasmus\WESAD"

# scaling frequencies, in Hz
scf_hr  = 1
scf_eda = 4
scf_bvp = 64

# drop initial seconds (calibration) 
OFFSET_SECONDS = 1000

# window size in seconds (used for all windowing)
WINDOW_SIZE = 60

# threshold dictionary (to be changed as needed for ROC)
THRESHOLDS = {
    "mean_HR": 82,
    "HRV_RMSSD": 35,
    "HRV_pNN50": 18,
    "SCR_Count": 3
}

# ---------------
# 1) data loaders
# ---------------

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

# --------------------
# 2) extract raw files
# --------------------

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
# 3) extracting the necessary parameters + processing
# ---------------------------------------------------

processed_data = {}

for pid, metrics in patient_data.items():
    processed_data[pid] = {}
    window_size = WINDOW_SIZE
    
    # HR mean per window
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
        hr_means = hr_means[hr_means["window_start_time_[s]"] >= OFFSET_SECONDS].reset_index(drop=True)
        processed_data[pid]["HR_mean_per_1min"] = hr_means

    # EDA: tonic means + SCR counts per window
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
        tonic_means = tonic_means[tonic_means["window_start_time_[s]"] >= OFFSET_SECONDS].reset_index(drop=True)
        processed_data[pid]["SCL_means_per_1min"] = tonic_means

        phasic_counts = (
            eda_signal
            .groupby('window', as_index=False)['SCR_Peaks']
            .sum()
            .rename(columns={'SCR_Peaks': 'SCR_Count'})
        )
        phasic_counts["window_start_time_[s]"] = phasic_counts["window"] * window_size
        phasic_counts = phasic_counts[phasic_counts["window_start_time_[s]"] >= OFFSET_SECONDS].reset_index(drop=True)
        processed_data[pid]["SCR_Counts_per_1min"] = phasic_counts

   
    if "ibi" in metrics:
        ibi_df = metrics["ibi"].copy()
        ibi_df["window"] = (ibi_df["Time[s]"] // window_size).astype(int)
        window_hrv = []
        for w, sub in ibi_df.groupby("window"):
            start_time = w * window_size
            if start_time < OFFSET_SECONDS:
                continue
            durations = sub["Duration [ms]"].astype(float).values
            if len(durations) < 2:
                # not enough intervals
                continue
            R_peaks = nk.intervals_to_peaks(durations, sampling_rate=scf_bvp)
            hrv = nk.hrv_time(R_peaks, sampling_rate=scf_bvp)
            # guard: sometimes hrv_time returns empty series
            try:
                rmssd_val = float(hrv["HRV_RMSSD"].iloc[0])
            except Exception:
                rmssd_val = np.nan
            try:
                pnn_val = float(hrv["HRV_pNN50"].iloc[0])
            except Exception:
                pnn_val = np.nan
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

# ---------------------
# 4) plotting functions
# ---------------------

def plot_pNN50_RMSSD():
    fig, ax = plt.subplots()
    for pid, pdata in processed_data.items():
        df_h = pdata.get("HRV_per_1min")
        if df_h is None or df_h.empty:
            continue
        X = df_h["HRV_RMSSD"].values.reshape(-1, 1)
        y = df_h["HRV_pNN50"].values
        mask = np.isfinite(X.flatten()) & np.isfinite(y)
        X, y = X[mask], y[mask]
        if len(X) < 2:
            continue
        ax.scatter(X.flatten(), y, label=pid)
        pipeline = make_pipeline(SimpleImputer(strategy="mean"), LinearRegression())
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
    ax1.plot(df['SCL_means_per_1min']['window_start_time_[s]'], df['SCL_means_per_1min']["mean_SCL"], color='green', label='Tonic EDA')
    ax1.set_xlabel("Time [s]")
    ax1.set_ylabel("SCL [μS]", color='green')
    ax2 = ax1.twinx()
    ax2.bar(df['SCR_Counts_per_1min']["window_start_time_[s]"], df['SCR_Counts_per_1min']["SCR_Count"], width=20, alpha=0.4,
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
    ax.plot(df['HR_mean_per_1min']["window_start_time_[s]"], df['HR_mean_per_1min']["mean_HR"], color="blue")
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Rate (bpm)")
    ax.set_title(f" HR over Time for {pid}")
    ax.grid(True)
    plt.show()

# -----------------------
# 5) label utilities (.pkl)
# -----------------------

FS_ECG = 700.0

def load_wesad(pkl_path: Path) -> dict:
    with open(pkl_path, "rb") as f:
        return pickle.load(f, encoding="latin1")

def segments_from_labels(labels: np.ndarray, fs_label: float):
    labels=labels.astype(int)
    if labels.size==0:
        return []
    changes=np.where(np.diff(labels)!=0)[0]+1
    idx_starts=np.r_[0, changes]
    idx_ends=np.r_[changes, labels.size]
    return [(s / fs_label, e / fs_label, int(labels[s])) for s, e in zip(idx_starts, idx_ends)]

def infer_label_fs(labels: np.ndarray, ecg: np.ndarray):
    if labels.shape[0]==ecg.shape[0]:
        return FS_ECG
    return FS_ECG


def explore_wesad_pkl(file_path):
    file_path=Path(file_path)
    if not file_path.exists():
        print(f"File {file_path} does not exist.")
        return

    # load pickle
    with open(file_path, "rb") as f:
        data=pickle.load(f, encoding="latin1")  # WESAD is stored in Python2 compatibility

    print("\n=== Keys in the root object ===")
    if isinstance(data, dict):
        for k, v in data.items():
            if hasattr(v, "shape"):
                print(f"{k:20s} -> type: {type(v)}, shape: {v.shape}")
            else:
                print(f"{k:20s} -> type: {type(v)}, size: {len(v) if hasattr(v,'__len__') else 'N/A'}")
    else:
        print("Unexpected file format:", type(data))

    return data


def describe_labels(labels):
    """Print a summary of the label array from WESAD."""
    print("\n=== Structure of 'label' ===")

    # Known label mapping from WESAD documentation
    label_map = {
        0: "not defined",
        1: "baseline",
        2: "stress",
        3: "amusement",
        4: "meditation",   # not used in all subjects
        5: "ignore",
        6: "ignore",
        7: "ignore",
        8: "ignore",
        9: "ignore"
    }

    if isinstance(labels, np.ndarray):
        unique, counts = np.unique(labels, return_counts=True)
        print(f"Label array shape: {labels.shape}")
        print("Unique values and their counts:")
        for val, cnt in zip(unique, counts):
            desc = label_map.get(val, "unknown")
            print(f"  {val}: {cnt} samples ({desc})")
    else:
        print("Unexpected type for label:", type(labels))


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python explore_wesad.py <file.pkl>")
        sys.exit(1)

    file_path = sys.argv[1]
    data = explore_wesad_pkl(file_path)

    if "signal" in data:
        print("\n=== Structure of 'signal' ===")
        for sensor, values in data["signal"].items():
            if hasattr(values, "shape"):
                print(f"{sensor:15s} -> {values.shape}")
            elif isinstance(values, dict):
                print(f"{sensor:15s} -> dict with keys: {list(values.keys())}")
            else:
                print(f"{sensor:15s} -> {type(values)}")

    # Show details about "label"
    if "label" in data:
        describe_labels(data["label"])


# -------------------------------------------
# 6) detection on processed_data (per-window)
# -------------------------------------------



def get_true_stress_windows_from_pkl(pkl_path, window_size=60, offset_seconds=0):
    pkl_path=Path(pkl_path)
    if not pkl_path.exists():
        raise FileNotFoundError(pkl_path)
    with open(pkl_path, "rb") as f:
        data=pickle.load(f, encoding="latin1")
    labels=data.get("label", None)
    ecg=None
    try:
        ecg=data["signal"]["chest"]["ECG"]
    except Exception:
        ecg = None

    if labels is None:
        return set(), []
    if ecg is not None and labels.shape[0]==ecg.shape[0]:
        fs_label = FS_ECG
    else:
        fs_label = FS_ECG

    # find indices where label==2 (stress)
    stress_idx = np.where(np.asarray(labels) == 2)[0]
    if stress_idx.size == 0:
        return set(), []

    stress_secs = stress_idx.astype(float) / fs_label - float(offset_seconds)
    window_idxs = np.floor(stress_secs / window_size).astype(int)
    window_idxs = window_idxs[stress_secs >= 0]
    stress_windows = set(window_idxs.tolist())

    segs = []
    if stress_idx.size:
        # collapse contiguous index ranges to segments (in seconds, before offset)
        changes = np.where(np.diff(stress_idx) != 1)[0] + 1
        starts = np.r_[stress_idx[0], stress_idx[changes]]
        ends = np.r_[stress_idx[changes - 1], stress_idx[-1]]
        for s_i, e_i in zip(starts, ends):
            s_sec = s_i / fs_label - float(offset_seconds)
            e_sec = e_i / fs_label - float(offset_seconds)
            segs.append((max(0.0, s_sec), max(0.0, e_sec)))
    return stress_windows, segs


def detect_stress_windows_from_processed(pid, processed_data, THRESHOLDS,
                                         scl_factor=1.48, min_violations=2, window_size=60):
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


def compare_detected_vs_true(pid, processed_data, pkl_path, THRESHOLDS,
                             scl_factor=1.48, min_violations=2, window_size=60, offset_seconds=0, tolerance_windows=1):
    # get true windows from pkl
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
        true_w = set()
        true_times = []
    else:
        # infer fs_label (WESAD chest ECG labels generally match 700 Hz)
        FS_ECG = 700.0
        fs_label = FS_ECG if (ecg is not None and labels.shape[0] == ecg.shape[0]) else FS_ECG
        stress_idx = np.where(np.asarray(labels) == 2)[0]
        if stress_idx.size == 0:
            true_w = set()
            true_times = []
        else:
            stress_secs = stress_idx.astype(float) / fs_label - float(offset_seconds)
            window_idxs = np.floor(stress_secs / window_size).astype(int)
            window_idxs = window_idxs[stress_secs >= 0]
            true_w = set(window_idxs.tolist())
            # build contiguous time segments (for diagnostics)
            segs = []
            changes = np.where(np.diff(stress_idx) != 1)[0] + 1
            starts = np.r_[stress_idx[0], stress_idx[changes]] if stress_idx.size else np.array([])
            ends = np.r_[stress_idx[changes - 1], stress_idx[-1]] if stress_idx.size else np.array([])
            for s_i, e_i in zip(starts, ends):
                s_sec = s_i / fs_label - float(offset_seconds)
                e_sec = e_i / fs_label - float(offset_seconds)
                segs.append((max(0.0, s_sec), max(0.0, e_sec)))
            true_times = segs

    # detected
    detected_w = detect_stress_windows_from_processed(pid, processed_data, THRESHOLDS,
                                                      scl_factor=scl_factor, min_violations=min_violations, window_size=window_size)

    # tolerant matching (+/- tolerance_windows)
    matched_true = set()
    matched_detected = set()
    for d in detected_w:
        for t in true_w:
            if abs(int(d) - int(t)) <= tolerance_windows:
                matched_true.add(t)
                matched_detected.add(d)
                break

    tp = len(matched_detected)
    fp = len(detected_w - matched_detected)
    fn = len(true_w - matched_true)

    # define universe of observed windows for patient (for TN calc)
    all_windows = set()
    pdata = processed_data.get(pid, {})
    for key in ("HR_mean_per_1min", "HRV_per_1min", "SCL_means_per_1min", "SCR_Counts_per_1min"):
        df = pdata.get(key)
        if df is not None and not df.empty and "window" in df.columns:
            all_windows.update(df["window"].dropna().astype(int).unique().tolist())
    tn = len(all_windows - true_w - detected_w)

    TPR= tp / (tp + fn) if (tp + fn) > 0 else 0.0
    FPR= fp / (fp + tn) if (tp + fp + fn) > 0 else 0.0

    summary = {
        "PID": pid,
        "detected_windows": sorted(list(detected_w)),
        "true_windows": sorted(list(true_w)),
        "TP": tp, "FP": fp, "FN": fn, "TN": tn,
        "TPR": TPR, "FPR": FPR,
        "true_time_segments": true_times
    }
    print(f"PID {pid}: TP={tp}, FP={fp}, FN={fn}, TN={tn}, TPR={TPR:.3f}, FPR={FPR:.3f}")
    print("True windows (indices):", sorted(list(true_w)))
    print("Detected windows (indices):", sorted(list(detected_w)))
    print("True time segments (sec):", true_times)
    return summary

'''
def compare_detected_vs_true(pid, processed_data, pkl_path, THRESHOLDS, scl_factor=1.48, min_violations=2, window_size=60, offset_seconds=0):
    true_windows, true_times = get_true_stress_windows_from_pkl(pkl_path, window_size=window_size, offset_seconds=offset_seconds)
    detected_windows = detect_stress_windows_from_processed(pid, processed_data, THRESHOLDS, scl_factor, min_violations, window_size)
    true_w = set(true_windows)
    detected_w = set(detected_windows)

    # tolerant matching: consider a detected window a true positive if it lies within +/-1 window of any true window
    tp=0
    matched_true=set()
    for d in detected_w:
        candidates= {t for t in true_w if abs(t - d) <= 1}
        if candidates:
            tp += 1
            matched_true.update(candidates)

    fp=len(detected_w) - tp
    fn=len(true_w - matched_true)
    tn=len(detected_w) - tp
    
    TPR= tp / (tp + fn) if (tp + fn) > 0 else 0.0
    FPR= fp / (fp + tn) if (tp + fp + fn) > 0 else 0.0

  
    
    summary = {
        "PID": pid,
        "detected_windows": sorted(list(detected_w)),
        "true_windows": sorted(list(true_w)),
        "TP": tp,
        "FP": fp,
        "FN": fn,
        "TN": tn,
        "TPR": TPR,
        "FPR": FPR,
        "true_times": true_times
    }
    

    print(f"PID {pid}: TP={tp}, FP={fp}, FN={fn}, TN={tn}, TPR={TPR:.3f}, FPR={FPR:.3f}")
    print("True windows (indices):", sorted(list(true_w)))
    print("Detected windows (indices):", sorted(list(detected_w)))
    print("True tim
          '''