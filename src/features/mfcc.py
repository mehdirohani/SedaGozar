"""
MFCC (Mel-Frequency Cepstral Coefficients) feature extraction.

This module extracts MFCC features from audio signals for use in the classic
speaker identification model. MFCCs capture the spectral envelope of speech,
representing vocal tract characteristics that are speaker-specific.

Scientific Background:
- MFCCs model the human auditory system's perception of sound
- They capture the spectral shape (formants) which reflect vocal tract geometry
- Mean and variance statistics provide a speaker-discriminative signature
- Less sensitive to pitch variations compared to raw spectral features
"""

import numpy as np
import librosa
from typing import Tuple


def extract_mfcc_features(audio: np.ndarray, sample_rate: int = 16000, 
                         n_mfcc: int = 40) -> np.ndarray:
    """
    Extract MFCC features with mean and variance statistics.
    
    This function computes MFCC coefficients and aggregates them over time
    using mean and variance statistics. This creates a fixed-size feature
    vector that captures speaker-specific characteristics.
    
    Process:
    1. Compute MFCCs (typically 40 coefficients)
    2. Calculate mean across time → 40 values
    3. Calculate variance across time → 40 values
    4. Concatenate → 80-dimensional feature vector
    
    Why mean and variance?
    - Mean captures the average spectral shape (vocal tract characteristics)
    - Variance captures speaking style variability and dynamics
    - Together, they provide a robust speaker signature
    
    Args:
        audio: Audio signal as numpy array (normalized to [-1, 1])
        sample_rate: Sample rate in Hz (default: 16000)
        n_mfcc: Number of MFCC coefficients to extract (default: 40)
        
    Returns:
        numpy.ndarray: Feature vector of shape (n_mfcc * 2,) containing
                      concatenated mean and variance statistics
                      
    Example:
        >>> audio = np.random.randn(16000)  # 1 second of audio
        >>> features = extract_mfcc_features(audio)
        >>> print(features.shape)
        (80,)
    """
    
    # Check if audio has sufficient length
    if len(audio) < sample_rate * 0.5:  # At least 0.5 seconds
        raise ValueError("Audio too short for MFCC extraction (minimum 0.5 seconds)")
    
    # Extract MFCCs
    # librosa.feature.mfcc returns shape (n_mfcc, time_frames)
    mfccs = librosa.feature.mfcc(
        y=audio,
        sr=sample_rate,
        n_mfcc=n_mfcc,
        n_fft=512,  # Frame size for FFT
        hop_length=256,  # Overlap between frames
        n_mels=128  # Number of Mel bands
    )
    
    # Compute statistics over time axis
    mfcc_mean = np.mean(mfccs, axis=1)  # Shape: (n_mfcc,)
    mfcc_var = np.var(mfccs, axis=1)    # Shape: (n_mfcc,)
    
    # Prevent zero variance (for numerical stability in classifiers)
    mfcc_var = np.maximum(mfcc_var, 1e-10)
    
    # Concatenate mean and variance
    features = np.concatenate([mfcc_mean, mfcc_var])
    
    return features


def extract_mfcc_sequence(audio: np.ndarray, sample_rate: int = 16000,
                         n_mfcc: int = 40) -> np.ndarray:
    """
    Extract MFCC sequence without aggregation (for deep learning models).
    
    This function returns the full MFCC sequence over time, which can be
    useful for temporal models or visualization.
    
    Args:
        audio: Audio signal as numpy array
        sample_rate: Sample rate in Hz (default: 16000)
        n_mfcc: Number of MFCC coefficients (default: 40)
        
    Returns:
        numpy.ndarray: MFCC sequence of shape (n_mfcc, time_frames)
    """
    mfccs = librosa.feature.mfcc(
        y=audio,
        sr=sample_rate,
        n_mfcc=n_mfcc,
        n_fft=512,
        hop_length=256,
        n_mels=128
    )
    
    return mfccs


def compute_delta_features(mfccs: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute delta (velocity) and delta-delta (acceleration) features.
    
    Delta features capture temporal dynamics of speech, improving
    speaker discrimination. They represent the rate of change of MFCCs.
    
    Note: This is optional and not used in the basic implementation,
    but can enhance performance if added to the feature vector.
    
    Args:
        mfccs: MFCC sequence of shape (n_mfcc, time_frames)
        
    Returns:
        Tuple of (delta, delta_delta) arrays, each of shape (n_mfcc, time_frames)
    """
    delta = librosa.feature.delta(mfccs, order=1)
    delta_delta = librosa.feature.delta(mfccs, order=2)
    
    return delta, delta_delta


def extract_extended_features(audio: np.ndarray, sample_rate: int = 16000) -> np.ndarray:
    """
    Extract extended feature set including MFCCs and deltas.
    
    This creates a more comprehensive feature vector by including:
    - 40 MFCC means
    - 40 MFCC variances
    - 40 Delta means
    - 40 Delta-Delta means
    Total: 160 dimensions
    
    This is more robust but also more complex. Use if the basic 80-dim
    features don't provide sufficient accuracy.
    
    Args:
        audio: Audio signal as numpy array
        sample_rate: Sample rate in Hz
        
    Returns:
        numpy.ndarray: Extended feature vector of shape (160,)
    """
    # Extract base MFCCs
    mfccs = extract_mfcc_sequence(audio, sample_rate, n_mfcc=40)
    
    # Compute deltas
    delta, delta_delta = compute_delta_features(mfccs)
    
    # Aggregate statistics
    mfcc_mean = np.mean(mfccs, axis=1)
    mfcc_var = np.var(mfccs, axis=1)
    delta_mean = np.mean(delta, axis=1)
    delta_delta_mean = np.mean(delta_delta, axis=1)
    
    # Prevent zero variance
    mfcc_var = np.maximum(mfcc_var, 1e-10)
    
    # Concatenate all features
    features = np.concatenate([
        mfcc_mean,
        mfcc_var,
        delta_mean,
        delta_delta_mean
    ])
    
    return features


# Example usage and testing
if __name__ == "__main__":
    # Generate test audio (1 second of noise)
    sample_rate = 16000
    duration = 5.0
    audio = np.random.randn(int(sample_rate * duration)) * 0.1
    
    # Test basic MFCC extraction
    print("Testing MFCC feature extraction...")
    features = extract_mfcc_features(audio, sample_rate)
    print(f"Feature shape: {features.shape}")  # Should be (80,)
    print(f"Feature range: [{features.min():.3f}, {features.max():.3f}]")
    print(f"Feature mean: {features.mean():.3f}")
    
    # Test MFCC sequence
    print("\nTesting MFCC sequence extraction...")
    mfcc_seq = extract_mfcc_sequence(audio, sample_rate)
    print(f"MFCC sequence shape: {mfcc_seq.shape}")  # Should be (40, time_frames)
    
    # Test extended features
    print("\nTesting extended features...")
    ext_features = extract_extended_features(audio, sample_rate)
    print(f"Extended features shape: {ext_features.shape}")  # Should be (160,)
    
    # Test with real audio if librosa example is available
    try:
        y, sr = librosa.load(librosa.ex('libri1'), sr=16000, duration=5.0)
        print("\nTesting with real speech audio...")
        real_features = extract_mfcc_features(y, sr)
        print(f"Real audio features: {real_features[:10]}")  # Show first 10 values
    except:
        print("\nNo example audio available, skipping real audio test")
