
import io
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

st.set_page_config(page_title="RS4D Near-Real-Time SHM Dashboard", layout="wide")

DEFAULT_FILES = {
    "ENE": "/mnt/data/AM.RA909.ENE_lagu2_20260318T130200_180s.csv",
    "EHZ": "/mnt/data/AM.RA909.EHZ_lag20260318T130200_180s.csv",
    "ENN": "/mnt/data/lag3 ENN.csv",
}

def find_numeric_signal_column(df: pd.DataFrame) -> str:
    preferred = [
        "accel_m_per_s2", "velocity_m_per_s", "counts", "value", "amplitude",
        "data", "signal", "sample"
    ]
    lower_map = {c.lower(): c for c in df.columns}
    for p in preferred:
        if p.lower() in lower_map:
            return lower_map[p.lower()]
    numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    if not numeric_cols:
        raise ValueError("No numeric signal column found in CSV.")
    return max(numeric_cols, key=lambda c: float(df[c].astype(float).var()))

def find_time_column(df: pd.DataFrame):
    candidates = ["time", "timestamp", "datetime", "date", "t", "seconds", "sec"]
    lower_map = {c.lower(): c for c in df.columns}
    for c in candidates:
        if c in lower_map:
            return lower_map[c]
    return None

def read_component_csv(uploaded_file=None, fallback_path=None):
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        source_name = uploaded_file.name
    else:
        df = pd.read_csv(fallback_path)
        source_name = Path(fallback_path).name

    sig_col = find_numeric_signal_column(df)
    time_col = find_time_column(df)

    y = pd.to_numeric(df[sig_col], errors="coerce").dropna().to_numpy(dtype=float)

    if time_col is not None:
        time_series = pd.to_numeric(df[time_col], errors="coerce")
        if time_series.notna().sum() == len(y):
            t = time_series.to_numpy(dtype=float)
            dt = np.median(np.diff(t)) if len(t) > 1 else np.nan
            fs = 1.0 / dt if np.isfinite(dt) and dt > 0 else None
        else:
            t = None
            fs = None
    else:
        t = None
        fs = None

    return {
        "df": df,
        "signal": y,
        "signal_col": sig_col,
        "time": t,
        "fs_from_file": fs,
        "source_name": source_name,
    }

def detrend_signal(y: np.ndarray) -> np.ndarray:
    x = np.arange(len(y), dtype=float)
    coeff = np.polyfit(x, y, 1)
    trend = coeff[0] * x + coeff[1]
    return y - trend

def taper_signal(y: np.ndarray, pct: float = 0.05) -> np.ndarray:
    n = len(y)
    m = max(1, int(n * pct))
    win = np.ones(n)
    ramp = 0.5 * (1 - np.cos(np.linspace(0, np.pi, m)))
    win[:m] = ramp
    win[-m:] = ramp[::-1]
    return y * win

def moving_average(y: np.ndarray, k: int) -> np.ndarray:
    if k <= 1:
        return y.copy()
    kernel = np.ones(k) / k
    return np.convolve(y, kernel, mode="same")

def bandpass_fft(y: np.ndarray, fs: float, fmin: float, fmax: float) -> np.ndarray:
    n = len(y)
    Y = np.fft.rfft(y)
    f = np.fft.rfftfreq(n, d=1.0 / fs)
    mask = (f >= fmin) & (f <= fmax)
    Yf = np.where(mask, Y, 0)
    return np.fft.irfft(Yf, n=n)

def preprocess(y: np.ndarray, fs: float, fmin: float, fmax: float, smooth_pts: int) -> np.ndarray:
    yp = detrend_signal(y)
    yp = taper_signal(yp, 0.05)
    if smooth_pts > 1:
        yp = moving_average(yp, smooth_pts)
    yp = bandpass_fft(yp, fs, fmin, min(fmax, 0.49 * fs))
    return yp

def fft_spectrum(y: np.ndarray, fs: float):
    n = len(y)
    freqs = np.fft.rfftfreq(n, d=1.0 / fs)
    amp = np.abs(np.fft.rfft(y))
    return freqs, amp

def dominant_frequency(y: np.ndarray, fs: float, fmin: float, fmax: float):
    freqs, amp = fft_spectrum(y, fs)
    mask = (freqs >= fmin) & (freqs <= fmax)
    if not np.any(mask):
        return np.nan, freqs, amp
    ff = freqs[mask]
    aa = amp[mask]
    i = np.argmax(aa)
    return float(ff[i]), freqs, amp

def sharpness_score(freqs, amp, peak_freq, band=0.35):
    if not np.isfinite(peak_freq):
        return 0.0
    local = (freqs >= max(0, peak_freq - band)) & (freqs <= peak_freq + band)
    if local.sum() < 3:
        return 0.0
    local_amp = amp[local]
    peak = local_amp.max()
    med = np.median(local_amp) + 1e-12
    ratio = peak / med
    return float(np.clip((ratio - 1.0) / 9.0, 0, 1))

def sliding_window_frequency(y: np.ndarray, fs: float, win_sec: float, step_sec: float, fmin: float, fmax: float):
    n = len(y)
    w = max(8, int(win_sec * fs))
    s = max(1, int(step_sec * fs))
    rows = []
    for start in range(0, n - w + 1, s):
        seg = y[start:start + w]
        f_dom, freqs, amp = dominant_frequency(seg, fs, fmin, fmax)
        t_mid = (start + w / 2) / fs
        rows.append({"time_s": t_mid, "freq_hz": f_dom})
    return pd.DataFrame(rows)

def compute_status(current_freq: float, baseline_freq: float):
    if not np.isfinite(current_freq) or not np.isfinite(baseline_freq) or baseline_freq <= 0:
        return "⚪ UNKNOWN", "Insufficient data", np.nan
    shift_pct = (baseline_freq - current_freq) / baseline_freq * 100.0
    if shift_pct < 5:
        return "🟢 NORMAL", "Building Status: NORMAL", shift_pct
    elif shift_pct < 15:
        return "🟡 WARNING", "WARNING: Frequency Shift Detected", shift_pct
    else:
        return "🔴 ALERT", "ALERT: Possible Structural Change", shift_pct

def overall_confidence(f_ene, f_enn, sharp_ene, sharp_enn):
    if not (np.isfinite(f_ene) and np.isfinite(f_enn) and f_ene > 0 and f_enn > 0):
        return 0
    agreement = 1.0 - abs(f_ene - f_enn) / max((f_ene + f_enn) / 2.0, 1e-9)
    agreement = np.clip(agreement, 0, 1)
    conf = 100 * (0.6 * agreement + 0.2 * sharp_ene + 0.2 * sharp_enn)
    return int(round(np.clip(conf, 0, 100)))

def fig_time_history(t, y, title):
    fig, ax = plt.subplots(figsize=(10, 3.2))
    if t is None:
        t = np.arange(len(y))
        ax.set_xlabel("Sample")
    else:
        ax.set_xlabel("Time (s)")
    ax.plot(t, y)
    ax.set_title(title)
    ax.set_ylabel("Amplitude")
    ax.grid(alpha=0.3)
    return fig

def fig_fft(freqs, amp, peak_freq, title, xlim):
    fig, ax = plt.subplots(figsize=(10, 3.2))
    ax.plot(freqs, amp)
    if np.isfinite(peak_freq):
        ax.axvline(peak_freq, linestyle="--")
    ax.set_xlim(0, xlim)
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Amplitude")
    ax.set_title(title)
    ax.grid(alpha=0.3)
    return fig

def fig_trend(df, baseline_freq):
    fig, ax = plt.subplots(figsize=(10, 3.2))
    ax.plot(df["time_s"], df["freq_hz"], marker="o", linewidth=1)
    if np.isfinite(baseline_freq):
        ax.axhline(baseline_freq, linestyle="--")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Dominant frequency (Hz)")
    ax.set_title("Sliding-window frequency trend")
    ax.grid(alpha=0.3)
    return fig

st.title("RS4D Near-Real-Time SHM Dashboard")
st.caption("Event-driven frequency tracking with status output for ENE, ENN, and EHZ components.")

st.sidebar.header("Input data")
use_uploaded = st.sidebar.toggle("Upload my own CSV files", value=False)

if use_uploaded:
    ene_file = st.sidebar.file_uploader("ENE CSV", type=["csv"], key="ene")
    enn_file = st.sidebar.file_uploader("ENN CSV", type=["csv"], key="enn")
    ehz_file = st.sidebar.file_uploader("EHZ CSV", type=["csv"], key="ehz")
else:
    ene_file = enn_file = ehz_file = None

st.sidebar.header("Processing settings")
fs = st.sidebar.number_input("Sampling rate (Hz)", min_value=1.0, max_value=500.0, value=100.0, step=1.0)
fmin = st.sidebar.number_input("Band-pass low cutoff (Hz)", min_value=0.01, max_value=50.0, value=0.5, step=0.05)
fmax = st.sidebar.number_input("Band-pass high cutoff (Hz)", min_value=0.1, max_value=49.0, value=15.0, step=0.5)
smooth_pts = st.sidebar.number_input("Moving-average points", min_value=1, max_value=51, value=3, step=2)

st.sidebar.header("Monitoring settings")
baseline_mode = st.sidebar.radio("Baseline frequency source", ["Manual", "From current ENE-ENN mean"], index=0)
manual_baseline = st.sidebar.number_input("Manual baseline frequency (Hz)", min_value=0.1, max_value=20.0, value=5.40, step=0.01)
win_sec = st.sidebar.number_input("Sliding window (sec)", min_value=5.0, max_value=120.0, value=20.0, step=1.0)
step_sec = st.sidebar.number_input("Window step (sec)", min_value=1.0, max_value=60.0, value=5.0, step=1.0)

run = st.sidebar.button("Run dashboard", type="primary")

if not run:
    st.info("Click Run dashboard from the sidebar to process the ENE, ENN, and EHZ files.")
    st.stop()

try:
    ene = read_component_csv(ene_file, DEFAULT_FILES["ENE"])
    enn = read_component_csv(enn_file, DEFAULT_FILES["ENN"])
    ehz = read_component_csv(ehz_file, DEFAULT_FILES["EHZ"])

    ene_y = preprocess(ene["signal"], fs, fmin, fmax, smooth_pts)
    enn_y = preprocess(enn["signal"], fs, fmin, fmax, smooth_pts)
    ehz_y = preprocess(ehz["signal"], fs, fmin, fmax, smooth_pts)

    f_ene, ene_freqs, ene_amp = dominant_frequency(ene_y, fs, fmin, fmax)
    f_enn, enn_freqs, enn_amp = dominant_frequency(enn_y, fs, fmin, fmax)
    f_ehz, ehz_freqs, ehz_amp = dominant_frequency(ehz_y, fs, fmin, fmax)

    sharp_ene = sharpness_score(ene_freqs, ene_amp, f_ene)
    sharp_enn = sharpness_score(enn_freqs, enn_amp, f_enn)

    current_freq = np.nanmean([f_ene, f_enn])

    baseline_freq = manual_baseline if baseline_mode == "Manual" else current_freq
    status_icon, status_text, shift_pct = compute_status(current_freq, baseline_freq)
    confidence = overall_confidence(f_ene, f_enn, sharp_ene, sharp_enn)
    period = 1.0 / current_freq if np.isfinite(current_freq) and current_freq > 0 else np.nan

    trend_df = sliding_window_frequency((ene_y + enn_y) / 2.0, fs, win_sec, step_sec, fmin, fmax)

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Status", status_icon)
    c2.metric("Current frequency", f"{current_freq:.3f} Hz" if np.isfinite(current_freq) else "N/A")
    c3.metric("Period", f"{period:.3f} s" if np.isfinite(period) else "N/A")
    c4.metric("Baseline", f"{baseline_freq:.3f} Hz" if np.isfinite(baseline_freq) else "N/A")
    c5.metric("Confidence", f"{confidence}%")

    st.markdown(f"## {status_text}")
    if np.isfinite(shift_pct):
        st.write(f"Estimated frequency shift relative to baseline: {shift_pct:.2f}%")

    summary = pd.DataFrame([
        {"Component": "ENE", "Source file": ene["source_name"], "Signal column": ene["signal_col"], "Dominant frequency (Hz)": round(float(f_ene), 4), "Peak sharpness": round(float(sharp_ene), 3)},
        {"Component": "ENN", "Source file": enn["source_name"], "Signal column": enn["signal_col"], "Dominant frequency (Hz)": round(float(f_enn), 4), "Peak sharpness": round(float(sharp_enn), 3)},
        {"Component": "EHZ", "Source file": ehz["source_name"], "Signal column": ehz["signal_col"], "Dominant frequency (Hz)": round(float(f_ehz), 4), "Peak sharpness": np.nan},
    ])
    st.subheader("Component summary")
    st.dataframe(summary, use_container_width=True)

    st.subheader("Frequency trend")
    st.pyplot(fig_trend(trend_df, baseline_freq), clear_figure=True, use_container_width=True)

    tab1, tab2, tab3, tab4 = st.tabs(["ENE", "ENN", "EHZ", "Export"])

    with tab1:
        st.pyplot(fig_time_history(None, ene_y, "ENE filtered signal"), clear_figure=True, use_container_width=True)
        st.pyplot(fig_fft(ene_freqs, ene_amp, f_ene, "ENE FFT spectrum", xlim=max(10, fmax)), clear_figure=True, use_container_width=True)

    with tab2:
        st.pyplot(fig_time_history(None, enn_y, "ENN filtered signal"), clear_figure=True, use_container_width=True)
        st.pyplot(fig_fft(enn_freqs, enn_amp, f_enn, "ENN FFT spectrum", xlim=max(10, fmax)), clear_figure=True, use_container_width=True)

    with tab3:
        st.pyplot(fig_time_history(None, ehz_y, "EHZ filtered signal"), clear_figure=True, use_container_width=True)
        st.pyplot(fig_fft(ehz_freqs, ehz_amp, f_ehz, "EHZ FFT spectrum", xlim=max(10, fmax)), clear_figure=True, use_container_width=True)

    with tab4:
        export_df = trend_df.copy()
        export_df["baseline_freq_hz"] = baseline_freq
        export_df["current_freq_hz"] = current_freq
        export_df["period_s"] = period
        export_df["status"] = status_text
        csv_bytes = export_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            "Download monitoring results CSV",
            data=csv_bytes,
            file_name="rs4d_monitoring_results.csv",
            mime="text/csv",
        )

        report = io.StringIO()
        report.write("RS4D Near-Real-Time SHM Dashboard Summary\n")
        report.write(f"Status: {status_text}\n")
        report.write(f"Current frequency (ENE-ENN mean): {current_freq:.4f} Hz\n")
        report.write(f"Period: {period:.4f} s\n")
        report.write(f"Baseline frequency: {baseline_freq:.4f} Hz\n")
        report.write(f"Frequency shift: {shift_pct:.2f}%\n")
        report.write(f"Confidence: {confidence}%\n\n")
        report.write("Per-component frequencies:\n")
        report.write(summary.to_string(index=False))
        st.download_button(
            "Download text summary",
            data=report.getvalue().encode("utf-8"),
            file_name="rs4d_dashboard_summary.txt",
            mime="text/plain",
        )

    with st.expander("Recommended thesis interpretation"):
        st.markdown(
            """
- NORMAL means the current mean horizontal frequency is close to the baseline.
- WARNING means a moderate drop in frequency.
- ALERT means a larger drop in frequency and needs further investigation.
- This dashboard is a near-real-time screening tool, not a final damage diagnosis system.
            """
        )

except Exception as e:
    st.error(f"Processing failed: {e}")
    st.exception(e)
