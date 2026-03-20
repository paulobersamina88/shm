import io

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from obspy.clients.fdsn import Client
from obspy import UTCDateTime

st.set_page_config(page_title="RS4D Near-Real-Time SHM Dashboard", layout="wide")


# =========================
# DATA FETCH
# =========================
def fetch_rs4d_waveforms(network: str, station: str, start_time: str, duration: int):
    client = Client("RASPISHAKE")
    t1 = UTCDateTime(start_time)
    t2 = t1 + duration

    st_obj = client.get_waveforms(
        network=network,
        station=station,
        location="*",
        channel="EN*,EH*",
        starttime=t1,
        endtime=t2,
        attach_response=False,
    )

    st_obj.merge(method=1, fill_value="interpolate")

    traces = {}
    sampling_rate = None

    for tr in st_obj:
        ch = tr.stats.channel.upper()
        traces[ch] = tr.data.astype(float)
        if sampling_rate is None:
            sampling_rate = float(tr.stats.sampling_rate)

    return traces, sampling_rate


def pick_channel(traces: dict, candidates: list[str]):
    for ch in candidates:
        if ch in traces:
            return traces[ch], ch
    return None, None


# =========================
# SIGNAL PROCESSING
# =========================
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
        f_dom, _, _ = dominant_frequency(seg, fs, fmin, fmax)
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


# =========================
# PLOTTING
# =========================
def fig_time_history(y, fs, title):
    fig, ax = plt.subplots(figsize=(10, 3.2))
    t = np.arange(len(y)) / fs
    ax.plot(t, y)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Amplitude")
    ax.set_title(title)
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


# =========================
# UI
# =========================
st.title("RS4D Near-Real-Time SHM Dashboard")
st.caption("Event-driven frequency tracking with status output for horizontal and vertical RS4D components.")

st.sidebar.header("Raspberry Shake fetch")
network = st.sidebar.text_input("Network", value="AM")
station = st.sidebar.text_input("Station", value="RA909")
start_time = st.sidebar.text_input("Start time (UTC)", value="2026-03-18T13:02:00")
duration = st.sidebar.number_input("Duration (seconds)", min_value=30, max_value=600, value=180, step=10)

st.sidebar.header("Processing settings")
manual_fs = st.sidebar.number_input("Sampling rate override (Hz, 0 = use downloaded)", min_value=0.0, max_value=500.0, value=0.0, step=1.0)
fmin = st.sidebar.number_input("Band-pass low cutoff (Hz)", min_value=0.01, max_value=50.0, value=0.5, step=0.05)
fmax = st.sidebar.number_input("Band-pass high cutoff (Hz)", min_value=0.1, max_value=49.0, value=15.0, step=0.5)
smooth_pts = st.sidebar.number_input("Moving-average points", min_value=1, max_value=51, value=3, step=2)

st.sidebar.header("Monitoring settings")
baseline_mode = st.sidebar.radio("Baseline frequency source", ["Manual", "From current horizontal mean"], index=0)
manual_baseline = st.sidebar.number_input("Manual baseline frequency (Hz)", min_value=0.1, max_value=20.0, value=5.40, step=0.01)
win_sec = st.sidebar.number_input("Sliding window (sec)", min_value=5.0, max_value=120.0, value=20.0, step=1.0)
step_sec = st.sidebar.number_input("Window step (sec)", min_value=1.0, max_value=60.0, value=5.0, step=1.0)

run = st.sidebar.button("Fetch & Analyze", type="primary")

if not run:
    st.info("Set station, start time, and duration in the sidebar, then click Fetch & Analyze.")
    st.stop()

try:
    traces, fs_downloaded = fetch_rs4d_waveforms(network, station, start_time, int(duration))

    ene_raw, ene_name = pick_channel(traces, ["ENE", "EHE"])
    enn_raw, enn_name = pick_channel(traces, ["ENN", "EHN"])
    ehz_raw, ehz_name = pick_channel(traces, ["EHZ"])

    if ene_raw is None or enn_raw is None:
        st.error("Could not find both horizontal components for this station/time window.")
        st.write("Available channels:", sorted(traces.keys()))
        st.stop()

    fs = manual_fs if manual_fs > 0 else fs_downloaded
    if fs is None or fs <= 0:
        st.error("Invalid sampling rate.")
        st.stop()

    min_len = min(len(ene_raw), len(enn_raw), len(ehz_raw) if ehz_raw is not None else len(ene_raw))
    ene_raw = ene_raw[:min_len]
    enn_raw = enn_raw[:min_len]
    if ehz_raw is not None:
        ehz_raw = ehz_raw[:min_len]

    ene_y = preprocess(ene_raw, fs, fmin, fmax, smooth_pts)
    enn_y = preprocess(enn_raw, fs, fmin, fmax, smooth_pts)
    ehz_y = preprocess(ehz_raw, fs, fmin, fmax, smooth_pts) if ehz_raw is not None else None

    f_ene, ene_freqs, ene_amp = dominant_frequency(ene_y, fs, fmin, fmax)
    f_enn, enn_freqs, enn_amp = dominant_frequency(enn_y, fs, fmin, fmax)

    if ehz_y is not None:
        f_ehz, ehz_freqs, ehz_amp = dominant_frequency(ehz_y, fs, fmin, fmax)
    else:
        f_ehz, ehz_freqs, ehz_amp = np.nan, np.array([]), np.array([])

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

    st.caption(f"Downloaded station: {network}.{station} | Sampling rate used: {fs:.2f} Hz | Horizontal channels: {ene_name}, {enn_name}" + (f" | Vertical: {ehz_name}" if ehz_name else ""))

    summary_rows = [
        {
            "Component": "Horizontal 1",
            "Channel": ene_name,
            "Dominant frequency (Hz)": round(float(f_ene), 4),
            "Peak sharpness": round(float(sharp_ene), 3),
        },
        {
            "Component": "Horizontal 2",
            "Channel": enn_name,
            "Dominant frequency (Hz)": round(float(f_enn), 4),
            "Peak sharpness": round(float(sharp_enn), 3),
        },
    ]

    if ehz_y is not None:
        summary_rows.append(
            {
                "Component": "Vertical",
                "Channel": ehz_name,
                "Dominant frequency (Hz)": round(float(f_ehz), 4),
                "Peak sharpness": np.nan,
            }
        )

    summary = pd.DataFrame(summary_rows)

    st.subheader("Component summary")
    st.dataframe(summary, use_container_width=True)

    st.subheader("Frequency trend")
    st.pyplot(fig_trend(trend_df, baseline_freq), clear_figure=True, use_container_width=True)

    if ehz_y is not None:
        tab1, tab2, tab3, tab4 = st.tabs([ene_name, enn_name, ehz_name, "Export"])
    else:
        tab1, tab2, tab4 = st.tabs([ene_name, enn_name, "Export"])
        tab3 = None

    with tab1:
        st.pyplot(fig_time_history(ene_y, fs, f"{ene_name} filtered signal"), clear_figure=True, use_container_width=True)
        st.pyplot(fig_fft(ene_freqs, ene_amp, f_ene, f"{ene_name} FFT spectrum", xlim=max(10, fmax)), clear_figure=True, use_container_width=True)

    with tab2:
        st.pyplot(fig_time_history(enn_y, fs, f"{enn_name} filtered signal"), clear_figure=True, use_container_width=True)
        st.pyplot(fig_fft(enn_freqs, enn_amp, f_enn, f"{enn_name} FFT spectrum", xlim=max(10, fmax)), clear_figure=True, use_container_width=True)

    if tab3 is not None and ehz_y is not None:
        with tab3:
            st.pyplot(fig_time_history(ehz_y, fs, f"{ehz_name} filtered signal"), clear_figure=True, use_container_width=True)
            st.pyplot(fig_fft(ehz_freqs, ehz_amp, f_ehz, f"{ehz_name} FFT spectrum", xlim=max(10, fmax)), clear_figure=True, use_container_width=True)

    with tab4:
        export_df = trend_df.copy()
        export_df["network"] = network
        export_df["station"] = station
        export_df["start_time_utc"] = start_time
        export_df["duration_s"] = duration
        export_df["baseline_freq_hz"] = baseline_freq
        export_df["current_freq_hz"] = current_freq
        export_df["period_s"] = period
        export_df["status"] = status_text

        csv_bytes = export_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            "Download monitoring results CSV",
            data=csv_bytes,
            file_name=f"{network}_{station}_monitoring_results.csv",
            mime="text/csv",
        )

        report = io.StringIO()
        report.write("RS4D Near-Real-Time SHM Dashboard Summary\n")
        report.write(f"Network: {network}\n")
        report.write(f"Station: {station}\n")
        report.write(f"Start time (UTC): {start_time}\n")
        report.write(f"Duration (s): {duration}\n")
        report.write(f"Status: {status_text}\n")
        report.write(f"Current frequency (horizontal mean): {current_freq:.4f} Hz\n")
        report.write(f"Period: {period:.4f} s\n")
        report.write(f"Baseline frequency: {baseline_freq:.4f} Hz\n")
        report.write(f"Frequency shift: {shift_pct:.2f}%\n")
        report.write(f"Confidence: {confidence}%\n\n")
        report.write("Per-component frequencies:\n")
        report.write(summary.to_string(index=False))

        st.download_button(
            "Download text summary",
            data=report.getvalue().encode("utf-8"),
            file_name=f"{network}_{station}_dashboard_summary.txt",
            mime="text/plain",
        )

    with st.expander("Recommended thesis interpretation"):
        st.markdown(
            """
- NORMAL means the current mean horizontal frequency is close to the baseline.
- WARNING means a moderate drop in frequency.
- ALERT means a larger drop in frequency and needs further investigation.
- This dashboard is a near-real-time screening tool, not a final damage diagnosis system.
- Direct frequency shifts may also reflect excitation level or soil-structure interaction, not only damage.
            """
        )

except Exception as e:
    st.error(f"Processing failed: {e}")
    st.exception(e)
