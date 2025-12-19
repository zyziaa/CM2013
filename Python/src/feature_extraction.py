from typing import Any
import numpy as np
import scipy
from scipy.stats import entropy, skew, kurtosis 
from scipy import signal
from mne_features.univariate import compute_hjorth_complexity
from spectrum import pburg
import pywt 
from src.preprocessing import compute_welch_psd, compute_bandpower
from tqdm import tqdm

def extract_time_domain_features(epoch):
    """
    Extract basic 16 time-domain features from a single epoch.

    Works for any signal type (EEG, EOG, EMG)

    Args:
        epoch (np.ndarray): A 1D array representing one epoch of signal data.

    Returns:
        dict: A dictionary of features.
    """
    features = {
        'mean': np.mean(epoch),
        'median': np.median(epoch),
        'std': np.std(epoch),
        'variance': np.var(epoch),
        'rms':np.sqrt(np.mean(epoch**2)),
        'min':np.min(epoch),
        'max': np.max(epoch),
        'range': np.max(epoch) - np.min(epoch),
        'skewness': scipy.stats.skew(epoch),
        'kurtosis': scipy.stats.kurtosis(epoch),
        'zero_crossings': np.sum(np.diff(np.sign(epoch)) != 0),
        'hjorth_activity': np.var(epoch),
        'hjorth_mobility': np.sqrt(np.var(np.diff(epoch)) / np.var(epoch)),
        'hjorth_complexity': compute_hjorth_complexity(epoch),
        'total_energy': np.sum(epoch**2),
        'mean_power': np.mean(epoch**2)
    }
    
    return features

def extract_frequency_domain_features(epoch, fs, AR_method, Welch_method, wavelet_method):

    features = {}
    AR_features = AR_method(epoch,fs)
    features.update(AR_features)
    Welch_features = Welch_method(epoch,fs)
    features.update(Welch_features)
    wavelet_features = wavelet_method(epoch,fs)
    features.update(wavelet_features)
    
    return features

# Precompute band masks OUTSIDE the epoch loop (huge speedup)

def prepare_freqs_masks(fs, nfft=512):
    """
    Prepare frequency axis and boolean masks for canonical bands.
    Returns: freqs, masks(dict), total_mask
    """
    freqs = np.linspace(0, fs/2, nfft//2 + 1)

    bands = {
        "delta": (0.5, 4),
        "theta": (4, 8),
        "alpha": (8, 13),
        "beta":  (13, 30),
        "gamma": (30, 50)
    }

    masks = {name: (freqs >= lo) & (freqs <= hi) for name, (lo, hi) in bands.items()}
    total_mask = (freqs >= 0.5) & (freqs <= 50)

    return freqs, masks, total_mask


def AR_method(epoch, fs, order=16, nfft=512, freqs=None, masks=None, total_mask=None):
    """
    Burg-based AR feature extraction.
    """
    #Ensure that masks exist (safe default)
    if masks is None or freqs is None or total_mask is None:
        freqs, masks, total_mask = prepare_freqs_masks(fs, nfft)

    #run Burg
    p = pburg(epoch, order=order, sampling=fs, NFFT=nfft)
    psd = np.array(p.psd)
    try:
        freqs_ar = np.array(p.frequencies())
    except Exception:
        # Some versions expose different API; if not available, assume pburg used same grid
        freqs_ar = freqs

    # Interpolate PSD to precomputed freq grid if needed
    if not np.allclose(freqs, freqs_ar):
        psd = np.interp(freqs, freqs_ar, psd)

    AR_features = {}

    # Band powers
    band_powers = {}
    for name, mask in masks.items():
        if np.any(mask):
            band_powers[name] = np.trapezoid(psd[mask], freqs[mask])
        else:
            band_powers[name] = 0.0

    total_power = np.trapezoid(psd[total_mask], freqs[total_mask]) + 1e-12

    # Relative powers
    for name in band_powers:
        AR_features[f"rel_{name}"] = band_powers[name] / total_power

    # Ratios
    AR_features["delta_alpha_ratio"] = band_powers["delta"] / (band_powers["alpha"] + 1e-12)
    AR_features["theta_beta_ratio"]  = band_powers["theta"] / (band_powers["beta"]  + 1e-12)
    AR_features["slow_fast_ratio"]   = (band_powers["delta"] + band_powers["theta"]) / \
                                      (band_powers["alpha"] + band_powers["beta"] + 1e-12)

    # Spectral edge (95%)
    cumulative = np.cumsum(psd)
    threshold = 0.95 * cumulative[-1]
    idx = np.searchsorted(cumulative, threshold)
    AR_features["edge_freq"] = freqs[min(idx, len(freqs)-1)]

    # Peak frequency inside total_mask
    tm = total_mask
    if np.any(tm):
        AR_features["ar_peak_freq"] = freqs[tm][np.argmax(psd[tm])]
    else:
        AR_features["ar_peak_freq"] = freqs[np.argmax(psd)]

    # Spectral entropy measures

    AR_features['entropy']=entropy(psd)
    
    return AR_features

def Welch_method(epoch, fs):
   
    Welch_features = {}
    
    # Welch parameters
    nperseg = int(4 * fs)
    noverlap = int(0.5 * nperseg)
    window = 'hann'
    
    bands = {
        'delta': (0, 4.0),
        'theta': (4.0, 8.0),
        'alpha': (8.0, 13.0),
        'beta':  (13.0, 30.0)
    }
    
    # Compute Welch PSD
    freqs, psd = compute_welch_psd(
        epoch,
        fs=fs,
        nperseg=nperseg,
        noverlap=noverlap,
        window=window,
        nfft=None,
        scaling='density'
    )
    
    # Calculate total power in 0.5-40 Hz band
    fmin, fmax = 0.5, 40.0
    m_an = (freqs >= fmin) & (freqs <= fmax)
    freqs_b, psd_b = freqs[m_an], psd[m_an]
    total_power = float(np.trapezoid(psd_b, freqs_b)) if freqs_b.size > 1 else 0.0
    
    # Calculate absolute band powers
    abs_powers = {}
    for name, (lo, hi) in bands.items():
        m = (freqs >= lo) & (freqs <= hi)
        abs_powers[name] = float(np.trapezoid(psd[m], freqs[m])) if np.any(m) else 0.0
    
    # Calculate relative band powers
    for name in bands.keys():
        Welch_features[f'pow_{name}'] = abs_powers[name]
        Welch_features[f'rel_{name}'] = (abs_powers[name] / total_power if total_power > 0 else 0.0)

    Welch_features['welch_entropy'] = entropy(psd)
    
    return Welch_features

def wavelet_method(epoch, fs, wavelet='db4', level=5):
    # this function is to extract wavelet based features from a EEG (epoch).
    # decomposiiton level = 5
    # feaures to be extract per level: energy, relative energy, entropy, mean, standard deviation, 
    # skewness and kurtosis

    coeffs = pywt.wavedec(epoch, wavelet, level=level)
    wavelet_features = {}
    total_energy = sum(np.sum(c ** 2) for c in coeffs)

    for i, coeff in enumerate(coeffs):
        band_name = f'wavelet_L{i}'
        energy = np.sum(coeff ** 2)
        rel_energy = energy / (total_energy + 1e-10)  #to avoid div by 0
        ent=entropy(coeff)

        # Compute per-coefficient statistics
        wavelet_features.update({
            f'{band_name}_energy': energy,
            f'{band_name}_rel_energy': rel_energy,
            f'{band_name}_entropy': ent,
            f'{band_name}_mean': np.mean(coeff),
            f'{band_name}_std': np.std(coeff),
            f'{band_name}_skew': skew(coeff),
            f'{band_name}_kurt': kurtosis(coeff),
        })

    return wavelet_features 

def entropy(psd):
    psd=np.array(psd)
    psd=np.abs(psd)
    psd_sum=psd.sum()
    if psd_sum==0:
        return 0.0
    norm_psd=psd/psd_sum

    mask=norm_psd>0
    norm_psd=norm_psd[mask]
    ent=-np.sum(norm_psd*np.log2(norm_psd))
    norm_ent=ent/np.log2(len(psd))
    return norm_ent
    
def extract_features(data, channel_info, config):
    """

    This function handles both single-channel (old format) and
    multi-channel data (new format with 2 EEG + 2 EOG + 1 EMG channels).

    Iteration 1: 16 time-domain features per EEG channel
    Iteration 2: 31+ features (time + frequency domain) per channel
    Iteration 3: Multi-signal features (EEG + EOG + EMG)
    Iteration 4: Optimized feature set (selected subset)

    Args:
        data: Either np.ndarray (single-channel) or dict (multi-channel)
        config (module): The configuration module.

    Returns:
        np.ndarray: A 2D array of features (n_epochs, n_features).
    """
    print(f"Extracting features for iteration {config.CURRENT_ITERATION}...")

    # Detect if we have multi-channel data structure
    is_multi_channel = isinstance(data, dict) and 'eeg' in data

    if is_multi_channel:
        print("Processing multi-channel data (EEG + EOG + EMG)")
        return extract_multi_channel_features(data, channel_info, config)
    else:
        print("Processing single-channel data (backward compatibility)")
        return extract_single_channel_features(data, channel_info, config)


def extract_multi_channel_features(multi_channel_data, channel_info, config):
    """
    Extract features from multi-channel data: 2 EEG + 2 EOG + 1 EMG channels.
    """
    eeg_fs = channel_info['eeg_fs']
    eog_fs = channel_info['eog_fs']
    emg_fs = channel_info['emg_fs']

    n_epochs = multi_channel_data['eeg'].shape[0]
    all_features = []

    # Precompute AR masks once for all epochs
    FREQS, MASKS, TOTAL_MASK = prepare_freqs_masks(eeg_fs, nfft=512)

    for epoch_idx in tqdm(range(n_epochs), desc="Extracting Features"):
        epoch_features = []
        eeg_energies=[]

        # EEG features (2 channels)
        for ch in range(multi_channel_data['eeg'].shape[1]):
            eeg_signal = multi_channel_data['eeg'][epoch_idx, ch, :]
            eeg_features = extract_time_domain_features(eeg_signal)
            epoch_features.extend(list(eeg_features.values()))
            
            if 'total_energy' in eeg_features:
                eeg_energies.append(eeg_features['total_energy'])

            # Iteration 2+: Add frequency domain features (AR + Welch + Wavelet)
            if config.CURRENT_ITERATION >= 2:
                eeg_freq_features = extract_frequency_domain_features(
                                    eeg_signal,
                                    eeg_fs,
                                    lambda ep, fs: AR_method(ep, fs, freqs=FREQS, masks=MASKS, total_mask=TOTAL_MASK),
                                    Welch_method,
                                    wavelet_method
                                )
                epoch_features.extend(list(eeg_freq_features.values()))
            
        if config.CURRENT_ITERATION >= 2:

            # Add EOG features (2 channels)
            if 'eog' in multi_channel_data:
                for ch in range(multi_channel_data['eog'].shape[1]):
                    eog_signal = multi_channel_data['eog'][epoch_idx, ch, :]
                    eog_features = extract_eog_features(eog_signal)
                    epoch_features.extend(list(eog_features.values()))

        if config.CURRENT_ITERATION >= 3:

            # Add EMG features (1 channel)
            emg_signal = multi_channel_data['emg'][epoch_idx, 0, :]
            emg_features = extract_emg_features(emg_signal, fs=emg_fs)
            epoch_features.extend(list(emg_features.values()))

            if 'emg_power' in emg_features:
                epoch_features.append(np.log1p(emg_features['emg_power']))
            else:
                epoch_features.append(0)
            
            #EMG / EEG Energy Ratio
            avg_eeg_energy = np.mean(eeg_energies) if eeg_energies else 1.0
            emg_p = emg_features.get('emg_power', 0)

            ratio = emg_p / (avg_eeg_energy + 1e-10)
            epoch_features.append(ratio)

        all_features.append(epoch_features)

    features = np.array(all_features)

    if config.CURRENT_ITERATION == 1:
        expected = 2 * 16  # 2 EEG channels × 3 features each
        print(f"Multi-channel Iteration 1: {features.shape[1]} features (target: {expected}+)")
        print("Students must implement remaining 13 time-domain features per EEG channel!")
    elif config.CURRENT_ITERATION >= 3:
        print(f"Multi-channel features extracted: {features.shape[1]} total")
        print("(2 EEG + 2 EOG + 1 EMG channels)")

    return features


def extract_single_channel_features(data, channel_info, config):
    """
    Backward compatibility for single-channel data.
    """
    if config.CURRENT_ITERATION == 1:
        # Iteration 1: Time-domain features
        all_features = []
        for epoch in data:
            features = extract_time_domain_features(epoch)
            all_features.append(list(features.values()))
        features = np.array(all_features)

        print(f"{features.shape[1]} features extracted")


    elif config.CURRENT_ITERATION >= 2:
        # Time domain + Frequency domain (AR + Welch + Wavelet)
        fs = channel_info['eeg_fs']  # Get sampling frequency from channel_info
        all_features = []

        # Precompute AR masks once
        FREQS, MASKS, TOTAL_MASK = prepare_freqs_masks(fs, nfft=512)

        epochs = data if data.ndim > 1 else data[None, :]
        for epoch in epochs:
            # Time domain features
            td = extract_time_domain_features(epoch)
            
            # Frequency domain features (AR + Welch + Wavelet)
            freq_features = extract_frequency_domain_features(
                            epoch,
                            fs,
                            lambda ep, fs: AR_method(ep, fs, freqs=FREQS, masks=MASKS, total_mask=TOTAL_MASK),
                            Welch_method,
                            wavelet_method
                        )
            
            all_features.append(list(td.values()) + list(freq_features.values()))

        features = np.array(all_features)

    else:
        raise ValueError(f"Invalid iteration: {config.CURRENT_ITERATION}")

    return features


def extract_eog_features(eog_signal):
    """
    Extract EOG-specific features for eye movement detection.
    EOG signals are used to detect:
    - Rapid eye movements (REM sleep indicator)
    - Slow eye movements
    - Eye blinks and artifacts
    """
    features = {
        'eog_mean': np.mean(eog_signal),
        'eog_std': np.std(eog_signal),
        'eog_peak': np.max(abs(eog_signal)),
        'eog_var':np.var(eog_signal),
        'eog_range': np.max(eog_signal) - np.min(eog_signal),
        'eog_kurtosis': kurtosis(eog_signal),
    }

    features['eog_energy'] = np.sum(eog_signal ** 2)
    features['eog_entropy'] = entropy(eog_signal)

    #Hjorth Mobility
    diff_signal = np.diff(eog_signal)
    var_signal = np.var(eog_signal)
    if var_signal > 0:
        features['eog_mobility'] = np.sqrt(np.var(diff_signal) / var_signal)
    else:
        features['eog_mobility'] = 0

    #REM detection
    abs_signal = np.abs(eog_signal)
    threshold = np.mean(abs_signal)+np.std(abs_signal) * 1.96
    peaks, _ = signal.find_peaks(abs_signal, height=threshold)
    features['eog_rem_score'] = len(peaks)

    return features


def extract_emg_features(emg_signal, fs=None):
    """
    Extract EMG-specific features for muscle tone detection.
    EMG signals are used to detect:
    - Muscle tone levels (high in wake, low in REM)
    - Muscle twitches and artifacts
    - Sleep-related muscle activity
    """
    features = {
        'emg_mean': np.mean(emg_signal),
        'emg_std': np.std(emg_signal),
        'emg_rms': np.sqrt(np.mean(emg_signal**2)),
        'emg_power': np.mean(emg_signal**2),
        'emg_var': np.var(emg_signal)
    }

    #High-frequency (20-40 Hz) power ratio
    if fs is not None:
        high_freq_band = (20,40)
        tot_power = np.mean(emg_signal**2) + 1e-12 #just to avoid dividing by 0
        high_freq_power = compute_bandpower(emg_signal, fs, band=high_freq_band)
        features['emg_high_freq_ratio'] = high_freq_power/tot_power

    return features
