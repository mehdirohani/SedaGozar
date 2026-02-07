"""
Mel Spectrogram generation and visualization.

This module generates Mel spectrograms for visual representation of audio signals.
Mel spectrograms show the frequency content of audio over time, using a perceptually-
relevant frequency scale (Mel scale).

Scientific Background:
- Mel scale approximates human auditory perception (non-linear frequency perception)
- Spectrograms provide time-frequency representation of audio
- Visual inspection can reveal speaker characteristics (pitch patterns, formants)
- Useful for debugging and understanding what the models are processing
"""

import numpy as np
import librosa
import librosa.display
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for server environments
import matplotlib.pyplot as plt
from io import BytesIO
import base64
from typing import Tuple, Optional


def generate_mel_spectrogram(audio: np.ndarray, sample_rate: int = 16000,
                             n_mels: int = 128) -> np.ndarray:
    """
    Generate Mel spectrogram from audio signal.
    
    The Mel spectrogram represents the power spectrum of a sound signal on the
    Mel scale, which approximates human auditory perception. Lower frequencies
    get more resolution, matching how humans perceive sound.
    
    Args:
        audio: Audio signal as numpy array (normalized to [-1, 1])
        sample_rate: Sample rate in Hz (default: 16000)
        n_mels: Number of Mel frequency bands (default: 128)
        
    Returns:
        numpy.ndarray: Mel spectrogram in dB scale, shape (n_mels, time_frames)
        
    Example:
        >>> audio = np.random.randn(16000)  # 1 second
        >>> mel_spec = generate_mel_spectrogram(audio)
        >>> print(mel_spec.shape)
        (128, ~63)  # Depends on audio length
    """
    
    # Compute Mel spectrogram
    mel_spec = librosa.feature.melspectrogram(
        y=audio,
        sr=sample_rate,
        n_fft=2048,  # FFT window size (larger = better frequency resolution)
        hop_length=512,  # Number of samples between successive frames
        n_mels=n_mels,  # Number of Mel bands
        fmin=0,  # Minimum frequency
        fmax=sample_rate // 2  # Maximum frequency (Nyquist)
    )
    
    # Convert to dB scale for better visualization and dynamic range
    # Add small epsilon to prevent log(0)
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
    
    return mel_spec_db


def plot_spectrogram(mel_spec_db: np.ndarray, sample_rate: int = 16000,
                    hop_length: int = 512, figsize: Tuple[int, int] = (10, 4),
                    title: str = "Mel Spectrogram") -> plt.Figure:
    """
    Create a matplotlib figure of the Mel spectrogram.
    
    This generates a publication-quality heatmap visualization of the
    Mel spectrogram with proper axis labels and colorbar.
    
    Args:
        mel_spec_db: Mel spectrogram in dB scale
        sample_rate: Sample rate in Hz
        hop_length: Number of samples between frames (used for time axis)
        figsize: Figure size as (width, height) in inches
        title: Title for the plot
        
    Returns:
        matplotlib.figure.Figure: Figure object containing the plot
    """
    
    fig, ax = plt.subplots(figsize=figsize)
    
    # Create the spectrogram display
    img = librosa.display.specshow(
        mel_spec_db,
        sr=sample_rate,
        hop_length=hop_length,
        x_axis='time',
        y_axis='mel',
        ax=ax,
        cmap='viridis'  # Perceptually uniform colormap
    )
    
    # Add colorbar
    fig.colorbar(img, ax=ax, format='%+2.0f dB')
    
    # Add labels and title
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xlabel('Time (s)', fontsize=12)
    ax.set_ylabel('Frequency (Hz)', fontsize=12)
    
    # Tight layout for better spacing
    fig.tight_layout()
    
    return fig


def spectrogram_to_image_array(mel_spec_db: np.ndarray, sample_rate: int = 16000,
                               figsize: Tuple[int, int] = (10, 4)) -> np.ndarray:
    """
    Convert Mel spectrogram to image array for display in Gradio.
    
    This creates a visualization and converts it to a numpy array
    that can be displayed directly in Gradio's Image component.
    
    Args:
        mel_spec_db: Mel spectrogram in dB scale
        sample_rate: Sample rate in Hz
        figsize: Figure size
        
    Returns:
        numpy.ndarray: RGB image array of shape (height, width, 3)
    """
    
    # Create figure
    fig = plot_spectrogram(mel_spec_db, sample_rate, figsize=figsize)
    
    # Convert figure to image array
    fig.canvas.draw()
    
    # Get the RGBA buffer from the figure
    buf = fig.canvas.buffer_rgba()
    
    # Convert to numpy array
    img_array = np.frombuffer(buf, dtype=np.uint8)
    img_array = img_array.reshape(fig.canvas.get_width_height()[::-1] + (4,))
    
    # Convert RGBA to RGB
    img_array = img_array[:, :, :3]
    
    # Close figure to free memory
    plt.close(fig)
    
    return img_array


def spectrogram_to_base64(mel_spec_db: np.ndarray, sample_rate: int = 16000,
                         figsize: Tuple[int, int] = (10, 4)) -> str:
    """
    Convert Mel spectrogram to base64-encoded PNG image.
    
    This is useful for embedding images in HTML or sending via web APIs.
    
    Args:
        mel_spec_db: Mel spectrogram in dB scale
        sample_rate: Sample rate in Hz
        figsize: Figure size
        
    Returns:
        str: Base64-encoded PNG image
    """
    
    # Create figure
    fig = plot_spectrogram(mel_spec_db, sample_rate, figsize=figsize)
    
    # Save to BytesIO buffer
    buf = BytesIO()
    fig.savefig(buf, format='png', dpi=100, bbox_inches='tight')
    buf.seek(0)
    
    # Encode to base64
    img_base64 = base64.b64encode(buf.read()).decode('utf-8')
    
    # Close figure
    plt.close(fig)
    
    return img_base64


def save_spectrogram(mel_spec_db: np.ndarray, filepath: str,
                    sample_rate: int = 16000, figsize: Tuple[int, int] = (10, 4)):
    """
    Save Mel spectrogram as image file.
    
    Args:
        mel_spec_db: Mel spectrogram in dB scale
        filepath: Path to save image file (e.g., 'spectrogram.png')
        sample_rate: Sample rate in Hz
        figsize: Figure size
    """
    
    fig = plot_spectrogram(mel_spec_db, sample_rate, figsize=figsize)
    fig.savefig(filepath, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved spectrogram to {filepath}")


def generate_equalizer_image(audio_chunk: np.ndarray, num_bands: int = 20, 
                             width: int = 400, height: int = 100) -> np.ndarray:
    """
    Generate a fast equalizer visualization for live display.
    
    This function computes FFT and creates a simple bar chart image using numpy
    for high performance (avoids matplotlib overhead).
    
    Args:
        audio_chunk: Audio chunk (short duration, e.g. 0.1s)
        num_bands: Number of frequency bands
        width: Image width in pixels
        height: Image height in pixels
        
    Returns:
        numpy.ndarray: RGB image array (height, width, 3)
    """
    if len(audio_chunk) < 2:
        return np.zeros((height, width, 3), dtype=np.uint8)
        
    # Compute magnitude spectrum
    fft = np.fft.rfft(audio_chunk * np.hanning(len(audio_chunk)))
    mag = np.abs(fft)
    
    # Resample to desired number of bands (simple averaging)
    # We want to focus on 0-8kHz range
    valid_bins = len(mag)
    if valid_bins < num_bands:
        bands = np.interp(np.linspace(0, valid_bins, num_bands), np.arange(valid_bins), mag)
    else:
        # Group bins into bands
        indices = np.linspace(0, valid_bins, num_bands + 1, dtype=int)
        bands = np.array([np.mean(mag[indices[i]:indices[i+1]]) for i in range(num_bands)])
    
    # Normalize with some smoothing/scaling
    # Log scale is better for audio
    bands = np.log1p(bands) * 50
    bands = np.clip(bands, 0, height)
    
    # Create empty image (dark background)
    img = np.zeros((height, width, 3), dtype=np.uint8)
    img[:, :] = [30, 30, 30]  # Dark gray
    
    # Draw bars
    bar_width = width // num_bands - 2
    for i in range(num_bands):
        bar_height = int(bands[i])
        if bar_height > 0:
            x_start = i * (width // num_bands) + 1
            x_end = x_start + bar_width
            y_start = height - bar_height
            
            # Gradient color (Green to Red)
            color = [
                int(255 * (bar_height / height)),  # R
                int(255 * (1 - bar_height / height)),  # G
                50  # B
            ]
            
            img[y_start:, x_start:x_end] = color
            
    return img


def generate_and_visualize(audio: np.ndarray, sample_rate: int = 16000,
                          title: Optional[str] = None) -> np.ndarray:
    """
    Generate Mel spectrogram and create visualization in one step.
    
    This is a convenience function that combines spectrogram generation
    and visualization for use in Gradio UI.
    
    Args:
        audio: Audio signal
        sample_rate: Sample rate in Hz
        title: Optional title for the plot
        
    Returns:
        numpy.ndarray: RGB image array ready for Gradio display
    """
    
    # Generate Mel spectrogram
    mel_spec_db = generate_mel_spectrogram(audio, sample_rate)
    
    # Set default title with audio duration
    if title is None:
        duration = len(audio) / sample_rate
        title = f"Mel Spectrogram ({duration:.1f}s)"
    
    # Create visualization
    fig = plot_spectrogram(mel_spec_db, sample_rate, title=title)
    
    # Convert to image array
    fig.canvas.draw()
    buf = fig.canvas.buffer_rgba()
    img_array = np.frombuffer(buf, dtype=np.uint8)
    img_array = img_array.reshape(fig.canvas.get_width_height()[::-1] + (4,))
    img_array = img_array[:, :, :3]  # RGBA to RGB
    
    plt.close(fig)
    
    return img_array


# Example usage and testing
if __name__ == "__main__":
    import time
    
    print("Testing Mel Spectrogram generation...")
    
    # Generate test audio (5 seconds with some varying frequency content)
    sample_rate = 16000
    duration = 5.0
    t = np.linspace(0, duration, int(sample_rate * duration))
    
    # Create audio with multiple frequency components (simulating speech)
    freq1 = 200 + 50 * np.sin(2 * np.pi * 2 * t)  # Varying fundamental frequency
    freq2 = 800 + 100 * np.sin(2 * np.pi * 3 * t)  # Varying formant
    audio = 0.3 * np.sin(2 * np.pi * freq1 * t) + 0.2 * np.sin(2 * np.pi * freq2 * t)
    
    # Add some noise
    audio += 0.05 * np.random.randn(len(audio))
    
    # Test spectrogram generation
    start = time.time()
    mel_spec = generate_mel_spectrogram(audio, sample_rate)
    print(f"Generated Mel spectrogram: shape {mel_spec.shape}, took {time.time()-start:.3f}s")
    print(f"Spectrogram range: [{mel_spec.min():.1f}, {mel_spec.max():.1f}] dB")
    
    # Test visualization
    print("\nTesting visualization...")
    img_array = generate_and_visualize(audio, sample_rate, title="Test Audio")
    print(f"Image array shape: {img_array.shape}")
    
    # Save example
    print("\nSaving example spectrogram...")
    save_spectrogram(mel_spec, "test_spectrogram.png", sample_rate)
    
    # Test with real audio if available
    try:
        print("\nTesting with real speech audio...")
        y, sr = librosa.load(librosa.ex('libri1'), sr=16000, duration=5.0)
        mel_spec_real = generate_mel_spectrogram(y, sr)
        save_spectrogram(mel_spec_real, "real_speech_spectrogram.png", sr)
        print("Saved real speech spectrogram")
    except:
        print("No example audio available, skipping real audio test")
    
    print("\nAll tests completed!")
