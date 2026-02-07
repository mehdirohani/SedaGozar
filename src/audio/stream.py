"""
Beast Mode Audio Streaming - VAD-Gated with 3-Second Accumulation Buffer

This module implements the front-end audio capture pipeline:

1. CAPTURE: Microphone → PyAudio at 16kHz mono
2. VAD FILTERING: Each chunk passes through Silero VAD
3. ACCUMULATION: Speech segments accumulate until 3 seconds of pure speech
4. TRIGGER: When buffer reaches 3 seconds → invoke callback
5. RESET: Clear buffer and start accumulating again

Key Features:
- Real-time silence filtering (discards silence immediately)
- 3-second accumulation window (only counts speech)
- Thread-safe operation for concurrent inference
- Low-latency visualization feedback
"""

import pyaudio
import numpy as np
import threading
import queue
from typing import Optional, Callable
import time
from ..models.beast_models import get_beast_engine


class BeastAudioStreamer:
    """
    VAD-gated audio streaming with 3-second accumulation buffer.
    
    Architecture:
    - Capture: 16kHz mono audio from microphone
    - Filter: VAD removes silence, keeps only speech
    - Accumulate: 3 seconds of pure speech (not wall-clock time)
    - Trigger: Invoke callback when buffer reaches 3 seconds
    """
    
    SAMPLE_RATE = 16000  # 16kHz (required by VAD and all models)
    CHANNELS = 1  # Mono
    CHUNK_SIZE = 1024  # Frames per capture iteration
    ACCUMULATION_DURATION = 3  # seconds of PURE SPEECH (not wall-clock)
    FORMAT = pyaudio.paInt16  # 16-bit audio
    
    def __init__(
        self,
        callback: Optional[Callable] = None,
        visualization_callback: Optional[Callable] = None,
        vad_threshold: float = 0.5
    ):
        """
        Initialize Beast Audio Streamer.
        
        Args:
            callback: Called when 3-second buffer is ready with audio array
            visualization_callback: Called on each chunk for real-time visualization
            vad_threshold: Speech probability threshold (0.0-1.0), default 0.5
        """
        self.callback = callback
        self.visualization_callback = visualization_callback
        self.vad_threshold = vad_threshold
        
        # Get Beast Mode Engine for VAD
        self.engine = get_beast_engine()
        
        # Streaming state
        self.is_streaming = False
        self.audio = pyaudio.PyAudio()
        self.stream = None
        
        # Calculate target samples for 3 seconds
        self.accumulation_samples = self.SAMPLE_RATE * self.ACCUMULATION_DURATION
        
        # Accumulation buffer (only keeps speech)
        self.speech_buffer = []  # List of float32 arrays
        self.speech_samples_count = 0  # Running count of accumulated speech samples
        self.buffer_lock = threading.Lock()
        
        # Statistics
        self.total_samples_captured = 0
        self.total_speech_samples = 0
        self.silence_segments = 0
        
        # Processing queue
        self.process_queue = queue.Queue()
        self.processing_thread = None
        
    
    def _audio_callback(self, in_data, frame_count, time_info, status):
        """
        PyAudio callback for each audio chunk.
        
        Process:
        1. Convert bytes to float32
        2. Apply VAD filtering
        3. Accumulate if speech detected
        4. Trigger callback at 3 seconds
        5. Visualize in real-time
        """
        if status:
            print(f"Audio stream status: {status}")
        
        # Convert bytes → float32 [-1, 1]
        audio_chunk = np.frombuffer(in_data, dtype=np.int16).astype(np.float32) / 32768.0
        
        self.total_samples_captured += len(audio_chunk)
        
        # Run visualization callback (high-frequency, on raw audio)
        if self.visualization_callback:
            try:
                self.visualization_callback(audio_chunk)
            except Exception:
                pass  # Ignore viz errors to avoid stopping stream
        
        # Apply VAD filtering
        try:
            filtered_audio, has_speech = self.engine.apply_vad(audio_chunk, self.SAMPLE_RATE)
        except Exception as e:
            print(f"VAD error: {e}")
            has_speech = True  # Default to keeping audio if VAD fails
            filtered_audio = audio_chunk
        
        # Process based on VAD result
        with self.buffer_lock:
            if has_speech:
                # Add to accumulation buffer
                self.speech_buffer.append(filtered_audio.copy())
                self.speech_samples_count += len(filtered_audio)
                self.total_speech_samples += len(filtered_audio)
                
                # Check if we've accumulated 3 seconds
                if self.speech_samples_count >= self.accumulation_samples:
                    # Extract exactly 3 seconds
                    buffer_data = self._extract_3sec_buffer()
                    
                    # Queue for inference
                    if self.callback:
                        self.process_queue.put(buffer_data)
                    
                    # Reset accumulation
                    self.speech_buffer = []
                    self.speech_samples_count = 0
            else:
                # Silence detected - discard
                self.silence_segments += 1
        
        return (None, pyaudio.paContinue)
    
    def _extract_3sec_buffer(self) -> np.ndarray:
        """
        Extract exactly 3 seconds from accumulated speech buffer.
        
        Returns:
            numpy array of exactly 3 seconds at 16kHz (48000 samples)
        """
        # Concatenate all speech chunks
        full_buffer = np.concatenate(self.speech_buffer)
        
        # Extract exactly 3 seconds
        buffer_3sec = full_buffer[:self.accumulation_samples].copy()
        
        # Keep overflow for next buffer
        if len(full_buffer) > self.accumulation_samples:
            overflow = full_buffer[self.accumulation_samples:]
            self.speech_buffer = [overflow]
            self.speech_samples_count = len(overflow)
        
        return buffer_3sec
    
    def _process_worker(self):
        """
        Background worker that invokes callbacks for ready buffers.
        
        Runs in separate thread to avoid blocking audio capture.
        """
        while self.is_streaming:
            try:
                buffer = self.process_queue.get(timeout=0.1)
                
                if self.callback:
                    try:
                        self.callback(buffer)
                    except Exception as e:
                        print(f"Callback error: {e}")
                
                self.process_queue.task_done()
            except queue.Empty:
                continue
    
    
    def start_streaming(self, device_index: Optional[int] = None):
        """
        Start audio streaming from microphone.
        
        Args:
            device_index: PyAudio device index (None for default)
        """
        if self.is_streaming:
            print("Already streaming")
            return
        
        self.is_streaming = True
        self.speech_buffer = []
        self.speech_samples_count = 0
        self.total_samples_captured = 0
        self.total_speech_samples = 0
        self.silence_segments = 0
        
        # Start background processing thread
        self.processing_thread = threading.Thread(
            target=self._process_worker,
            daemon=True
        )
        self.processing_thread.start()
        
        # Open PyAudio stream
        try:
            self.stream = self.audio.open(
                format=self.FORMAT,
                channels=self.CHANNELS,
                rate=self.SAMPLE_RATE,
                input=True,
                input_device_index=device_index,
                frames_per_buffer=self.CHUNK_SIZE,
                stream_callback=self._audio_callback
            )
            
            self.stream.start_stream()
            print(f"🎤 Started Beast streaming: {self.SAMPLE_RATE}Hz, VAD enabled, 3-sec accumulation")
        except Exception as e:
            print(f"Error starting stream: {e}")
            self.is_streaming = False
            raise
    
    
    def stop_streaming(self):
        """Stop audio streaming and cleanup."""
        if not self.is_streaming:
            return
        
        self.is_streaming = False
        
        if self.stream:
            self.stream.stop_stream()
            self.stream.close()
            self.stream = None
        
        if self.processing_thread:
            self.processing_thread.join(timeout=2.0)
        
        print("⏹️  Stopped Beast streaming")
    
    def get_statistics(self) -> dict:
        """Get streaming statistics."""
        total_time = self.total_samples_captured / self.SAMPLE_RATE
        speech_time = self.total_speech_samples / self.SAMPLE_RATE
        
        return {
            'total_captured_seconds': total_time,
            'total_speech_seconds': speech_time,
            'speech_ratio': (speech_time / total_time * 100) if total_time > 0 else 0,
            'silence_segments': self.silence_segments,
        }
    
    
    @staticmethod
    def list_input_devices():
        """List available audio input devices."""
        p = pyaudio.PyAudio()
        info = p.get_host_api_info_by_index(0)
        numdevices = info.get('deviceCount')
        
        devices = []
        for i in range(numdevices):
            dev_info = p.get_device_info_by_host_api_device_index(0, i)
            if dev_info.get('maxInputChannels') > 0:
                devices.append((i, dev_info.get('name')))
        
        p.terminate()
        return devices
    
    def __del__(self):
        """Cleanup on deletion."""
        self.stop_streaming()
        if hasattr(self, 'audio'):
            self.audio.terminate()


# Example usage
if __name__ == "__main__":
    import matplotlib.pyplot as plt
    
    def on_buffer_ready(buffer):
        """Callback when 3-second buffer is ready."""
        print(f"\n✓ 3-sec buffer ready: {len(buffer)} samples, {len(buffer)/16000:.1f}s")
        print(f"  Min: {buffer.min():.3f}, Max: {buffer.max():.3f}, Mean: {buffer.mean():.3f}")
    
    print("Starting Beast Audio Streamer test...")
    print("Recording for 30 seconds (will trigger ~10 buffers)...\n")
    
    streamer = BeastAudioStreamer(callback=on_buffer_ready)
    streamer.start_streaming()
    
    try:
        time.sleep(30)
    except KeyboardInterrupt:
        print("\nInterrupted by user")
    finally:
        streamer.stop_streaming()
        
        stats = streamer.get_statistics()
        print("\n" + "="*50)
        print("STATISTICS:")
        for key, value in stats.items():
            if isinstance(value, float):
                print(f"  {key}: {value:.1f}")
            else:
                print(f"  {key}: {value}")
