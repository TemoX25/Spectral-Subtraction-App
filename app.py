import io
import numpy as np
import wave
import streamlit as st
import matplotlib.pyplot as plt
from scipy.signal import stft, istft, get_window
from scipy.io.wavfile import write

# Feste Defaults
NPERSEG, NOVERLAP = 1024, 512

st.title("Spektral-Subtraktions-Filter")

uploaded = st.file_uploader("WAV-Datei hochladen", type=["wav"])
NOISE_SEC = st.number_input("Rauschfenster am Anfang (Sekunden)", min_value=0.0, value=10.0, step=0.5)
ALPHA = st.slider("Alpha (Over-Subtraction)", min_value=0.0, max_value=5.0, value=1.2, step=0.05)

def spec_db(Z: np.ndarray) -> np.ndarray:
    return 20 * np.log10(np.maximum(np.abs(Z), 1e-12))

def plot_spectrogram(Z: np.ndarray, f: np.ndarray, t: np.ndarray, title: str) -> tuple[plt.Figure, bytes]:
    fig, ax = plt.subplots(figsize=(10, 4))
    pcm = ax.pcolormesh(t, f, spec_db(Z), shading="auto")
    ax.set_ylabel("Frequenz [Hz]")
    ax.set_xlabel("Zeit [s]")
    ax.set_title(title)
    fig.colorbar(pcm, ax=ax, label="dB")
    fig.tight_layout()

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    return fig, buf.getvalue()

if uploaded is not None:
    # WAV lesen
    wav_bytes = uploaded.read()
    with wave.open(io.BytesIO(wav_bytes), "rb") as wf:
        p = wf.getparams()
        raw = wf.readframes(p.nframes)

    # Auf Mono umwandeln
    dt = np.int16 if p.sampwidth == 2 else np.int32
    scale = 32768.0 if p.sampwidth == 2 else 2147483648.0

    x_i = np.frombuffer(raw, dtype=dt)
    if p.nchannels > 1:
        x_i = x_i.reshape(-1, p.nchannels)[:, 0]

    # Abtastrate & normalisieren
    fs = p.framerate
    x = x_i.astype(np.float64) / scale

    # STFT
    win = get_window("hann", NPERSEG)
    f, t, X = stft(x, fs=fs, window=win, nperseg=NPERSEG, noverlap=NOVERLAP)
    X_mag, X_phase = np.abs(X), np.angle(X)

    # Spektrogramm Original anzeigen + Download
    fig_orig, orig_png = plot_spectrogram(X, f, t, "Spektrogramm Original (dB)")
    st.pyplot(fig_orig)
    st.download_button(
        "Original-Spektrogramm herunterladen (PNG)",
        data=orig_png,
        file_name="spektrogramm_original.png",
        mime="image/png",
    )

    # Noise-Profil (mit Fallback)
    noise_cols = t < NOISE_SEC
    if not np.any(noise_cols):
        # falls NOISE_SEC zu groß/Datei zu kurz: nimm mindestens 1 Frame
        noise_cols = np.array([True] + [False] * (len(t) - 1))

    N_mag = np.mean(X_mag[:, noise_cols], axis=1, keepdims=True)

    # Spektral-Subtraktion
    Y_mag = np.maximum(X_mag - ALPHA * N_mag, 0.0)
    Y = Y_mag * np.exp(1j * X_phase)

    # iSTFT
    _, y = istft(Y, fs=fs, window=win, nperseg=NPERSEG, noverlap=NOVERLAP)

    # Export
    out = np.clip(y * scale, -scale, scale - 1).astype(dt)

    # Streamlit will Bytes zum Download -> in Memory schreiben
    out_buf = io.BytesIO()
    write(out_buf, fs, out)
    out_bytes = out_buf.getvalue()

    st.success("Fertig gefiltert.")
    st.audio(out_bytes, format="audio/wav")
    st.download_button(
        "Gefilterte Datei herunterladen",
        data=out_bytes,
        file_name="gefiltert.wav",
        mime="audio/wav",
    )

    # Spektrogramm Gefiltert anzeigen + Download
    fig_filt, filt_png = plot_spectrogram(Y, f, t, "Spektrogramm Gefiltert (dB)")
    st.pyplot(fig_filt)
    st.download_button(
        "Gefiltertes Spektrogramm herunterladen (PNG)",
        data=filt_png,
        file_name="spektrogramm_gefiltert.png",
        mime="image/png",
    )
