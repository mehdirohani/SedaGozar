"""
Audio recording module for speaker registration.

This module provides simple audio recording functionality for capturing
speaker samples during registration. It handles fixed-duration recordings
and saves to WAV format.
"""

import pyaudio
import wave
import numpy as np
from pathlib import Path
from typing import Optional


class AudioRecorder:
    """
    Simple audio recorder for speaker registration.
    
    This class records fixed-duration audio samples from the microphone
    and saves them as WAV files. It uses the same audio parameters as
    the AudioStreamer for consistency.
    
    Attributes:
        SAMPLE_RATE (int): Audio sample rate in Hz (16000)
        CHANNELS (int): Number of audio channels (1 for mono)
        CHUNK_SIZE (int): Number of frames per buffer (1024)
        FORMAT: PyAudio format (paInt16)
    """
    
    SAMPLE_RATE = 16000  # 16kHz
    CHANNELS = 1  # Mono
    CHUNK_SIZE = 1024
    FORMAT = pyaudio.paInt16
    
    def __init__(self):
        """Initialize audio recorder."""
        self.audio = pyaudio.PyAudio()
    
    def record(self, duration: float = 5.0, device_index: Optional[int] = None) -> np.ndarray:
        """
        Record audio from microphone for specified duration.
        
        Args:
            duration: Recording duration in seconds (default: 5.0)
            device_index: Index of input device to use (None for default)
            
        Returns:
            numpy.ndarray: Recorded audio as float32 array, normalized to [-1, 1]
        """
        print(f"Recording for {duration} seconds...")
        
        # Calculate number of chunks needed
        num_chunks = int(self.SAMPLE_RATE / self.CHUNK_SIZE * duration)
        
        # Open stream
        stream = self.audio.open(
            format=self.FORMAT,
            channels=self.CHANNELS,
            rate=self.SAMPLE_RATE,
            input=True,
            input_device_index=device_index,
            frames_per_buffer=self.CHUNK_SIZE
        )
        
        frames = []
        
        try:
            # Record chunks
            for i in range(num_chunks):
                data = stream.read(self.CHUNK_SIZE, exception_on_overflow=False)
                frames.append(data)
                
                # Print progress
                progress = (i + 1) / num_chunks * 100
                if (i + 1) % 10 == 0:  # Update every 10 chunks
                    print(f"Recording: {progress:.0f}%")
        
        finally:
            # Clean up
            stream.stop_stream()
            stream.close()
        
        print("Recording complete!")
        
        # Convert to numpy array
        audio_data = b''.join(frames)
        audio_array = np.frombuffer(audio_data, dtype=np.int16)
        
        # Normalize to [-1, 1]
        audio_array = audio_array.astype(np.float32) / 32768.0
        
        return audio_array
    
    def save_wav(self, audio_data: np.ndarray, filepath: str):
        """
        Save audio data to WAV file.
        
        Args:
            audio_data: Audio data as numpy array (normalized to [-1, 1])
            filepath: Path to save WAV file
        """
        # Convert back to int16
        audio_int16 = (audio_data * 32768.0).astype(np.int16)
        
        # Ensure directory exists
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        
        # Write WAV file
        with wave.open(filepath, 'wb') as wf:
            wf.setnchannels(self.CHANNELS)
            wf.setsampwidth(self.audio.get_sample_size(self.FORMAT))
            wf.setframerate(self.SAMPLE_RATE)
            wf.writeframes(audio_int16.tobytes())
        
        print(f"Saved audio to {filepath}")
    
    def record_and_save(self, filepath: str, duration: float = 5.0, device_index: Optional[int] = None) -> np.ndarray:
        """
        Record audio and save to file in one operation.
        
        Args:
            filepath: Path to save WAV file
            duration: Recording duration in seconds (default: 5.0)
            device_index: Index of input device to use
            
        Returns:
            numpy.ndarray: Recorded audio data
        """
        audio_data = self.record(duration, device_index)
        self.save_wav(audio_data, filepath)
        return audio_data
    
    def load_wav(self, filepath: str) -> np.ndarray:
        """
        Load audio from WAV file.
        
        Args:
            filepath: Path to WAV file
            
        Returns:
            numpy.ndarray: Audio data normalized to [-1, 1]
        """
        with wave.open(filepath, 'rb') as wf:
            # Verify format
            if wf.getnchannels() != self.CHANNELS:
                raise ValueError(f"Expected {self.CHANNELS} channel(s), got {wf.getnchannels()}")
            if wf.getframerate() != self.SAMPLE_RATE:
                print(f"Warning: File sample rate {wf.getframerate()}Hz differs from expected {self.SAMPLE_RATE}Hz")
            
            # Read frames
            frames = wf.readframes(wf.getnframes())
            audio_data = np.frombuffer(frames, dtype=np.int16)
            
            # Normalize
            audio_data = audio_data.astype(np.float32) / 32768.0
        
        return audio_data
    
    def normalize_audio_level(self, audio_data: np.ndarray, target_rms: float = 0.1) -> np.ndarray:
        """
        Normalize audio to target RMS level.
        
        This helps ensure consistent audio levels across different recordings
        and microphones, improving model performance.
        
        Args:
            audio_data: Input audio data
            target_rms: Target RMS level (default: 0.1)
            
        Returns:
            numpy.ndarray: Normalized audio data
        """
        current_rms = np.sqrt(np.mean(audio_data ** 2))
        
        if current_rms < 1e-6:  # Avoid division by zero for silence
            return audio_data
        
        scaling_factor = target_rms / current_rms
        
        # Apply scaling with clipping to prevent overflow
        normalized = audio_data * scaling_factor
        normalized = np.clip(normalized, -1.0, 1.0)
        
        return normalized
    
    def __del__(self):
        """Cleanup on deletion."""
        if hasattr(self, 'audio'):
            self.audio.terminate()


# Example usage
if __name__ == "__main__":
    recorder = AudioRecorder()
    
    # Record 5 seconds
    print("Test recording...")
    audio = recorder.record(duration=5.0)
    
    # Save to file
    test_file = "test_recording.wav"
    recorder.save_wav(audio, test_file)
    
    # Load and verify
    loaded = recorder.load_wav(test_file)
    print(f"Original shape: {audio.shape}, Loaded shape: {loaded.shape}")
    print(f"Arrays equal: {np.allclose(audio, loaded)}")
    
    # Test normalization
    normalized = recorder.normalize_audio_level(audio)
    print(f"Original RMS: {np.sqrt(np.mean(audio**2)):.4f}")
    print(f"Normalized RMS: {np.sqrt(np.mean(normalized**2)):.4f}")
