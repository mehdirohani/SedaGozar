"""
Beast Mode Speaker Identification - Gradio UI

Modern interface with Beast Mode backend:
- VAD-filtered audio streaming
- 3-second accumulation buffer
- 3 parallel SOTA models (ECAPA-TDNN, TitaNet-Large, WavLM-Large)
- Real-time visualization and results
"""

import gradio as gr
import numpy as np
import threading
from typing import Optional, Tuple
import traceback
import librosa
import pandas as pd

# Import Beast Mode components
from ..audio.stream import BeastAudioStreamer
from ..audio.recorder import AudioRecorder
from ..features.spectrogram import generate_and_visualize, generate_equalizer_image
from ..database.manager import DatabaseManager
from ..models.beast_models import get_beast_engine


class BeastSpeakerIdentificationApp:
    """
    Beast Mode Speaker Identification UI using Gradio.
    
    Two main tabs:
    1. Live Identification: Real-time speaker ID with 3 models
    2. Speaker Registration: Register new speakers with all 3 embeddings
    """
    
    def __init__(self):
        """Initialize Beast Mode app."""
        print("\n" + "="*60)
        print("🔥 BEAST MODE SPEAKER IDENTIFICATION SYSTEM")
        print("="*60 + "\n")
        
        # Initialize database
        self.database = DatabaseManager()
        
        # Initialize Beast Mode Engine lazily (with timeout)
        self.engine = None
        self._init_engine_async()
        
        # Initialize UI components
        self.recorder = AudioRecorder()
        self.streamer = None
        
        # State management
        self.is_streaming = False
        self.current_results = "📊 Waiting for audio..."
        self.current_spectrogram = None
        self.current_eq = np.zeros((100, 400, 3), dtype=np.uint8)
        
        print("✅ Beast Mode App initialized\n")
    
    def _init_engine_async(self):
        """Initialize Beast Mode Engine in background thread."""
        def load_engine():
            try:
                self.engine = get_beast_engine()
            except Exception as e:
                print(f"⚠️ Warning: Could not fully initialize engine: {e}")
                self.engine = None
        
        # Start loading in background
        thread = threading.Thread(target=load_engine, daemon=True)
        thread.start()
    
    def _normalize_audio(self, buffer: np.ndarray) -> np.ndarray:
        """
        Normalize audio buffer to ensure consistent volume.
        
        This ensures the audio level matches the registered samples.
        """
        # Remove DC offset
        buffer = buffer - np.mean(buffer)
        
        # Normalize to [-1, 1] range
        max_val = np.max(np.abs(buffer))
        if max_val > 0:
            buffer = buffer / max_val
        
        return buffer.astype(np.float32)
    
    # ==================== TAB 1: LIVE IDENTIFICATION ====================
    
    def _on_buffer_ready(self, buffer: np.ndarray):
        """
        Process 3-second buffer ready from streamer.
        
        Called every 3 seconds of accumulated speech.
        Normalizes audio and runs weighted fusion inference.
        """
        try:
            # Wait for engine if still loading
            if self.engine is None:
                self.current_results = "⏳ Engine loading..."
                return
            
            # Normalize audio buffer for consistent volume
            buffer = self._normalize_audio(buffer)
            
            # Generate spectrogram
            self.current_spectrogram = generate_and_visualize(buffer, sample_rate=16000)
            
            # Extract embeddings from all 3 models
            embeddings = self.engine.extract_all_embeddings(buffer)
            
            # Get speakers from database
            speakers = self.database.list_speakers()
            
            if len(speakers) < 2:
                self.current_results = "❌ Need ≥2 registered speakers"
                return
            
            # Use weighted fusion for improved accuracy
            self.current_results = self._identify_with_weighted_fusion(
                embeddings,
                speakers
            )
            
        except Exception as e:
            print(f"Buffer processing error: {e}")
            traceback.print_exc()
            self.current_results = f"❌ Error: {str(e)[:60]}"
    
    def _identify_with_weighted_fusion(self, embeddings: dict, speakers: list) -> str:
        """
        Identify speaker using weighted fusion of all 3 models.
        
        Weighted Scoring:
        - ECAPA-TDNN: 30% weight
        - TitaNet-Large: 50% weight (most reliable)
        - WavLM: 20% weight
        
        If final score < 75%, reports 'Unknown'
        
        Returns:
            Formatted result string with unified score
        """
        try:
            # Weights for each model (TitaNet is more reliable)
            weights = {
                'ecapa': 0.3,
                'titanet': 0.5,
                'wavlm': 0.2
            }
            
            best_speaker = None
            best_weighted_score = -1
            
            # Compare with each registered speaker
            for speaker in speakers:
                speaker_id = speaker['id']
                weighted_score = 0.0
                valid_scores = 0
                
                # Get scores from all 3 models
                for model_name, weight in weights.items():
                    embedding = embeddings.get(model_name)
                    if embedding is None:
                        continue
                    
                    # Load registered speaker embedding
                    speaker_embedding = self.database.load_embedding(speaker_id, model_name)
                    if speaker_embedding is None:
                        continue
                    
                    # Compute similarity
                    similarity = self._cosine_similarity(embedding, speaker_embedding)
                    weighted_score += weight * similarity
                    valid_scores += weight
                
                # Normalize weighted score
                if valid_scores > 0:
                    weighted_score = weighted_score / valid_scores
                
                if weighted_score > best_weighted_score:
                    best_weighted_score = weighted_score
                    best_speaker = speaker
            
            # Check confidence threshold (75%)
            if best_speaker is None or best_weighted_score < 0.75:
                return f"❓ **Unknown Speaker**\n\n🎯 Confidence: {best_weighted_score:.1%}\n\n*Below 75% threshold*"
            
            return f"✅ **{best_speaker['name']}**\n\n🎯 Confidence: {best_weighted_score:.1%}\n\n*Weighted Fusion (ECAPA 30% | TitaNet 50% | WavLM 20%)*"
            
        except Exception as e:
            print(f"Weighted fusion error: {e}")
            traceback.print_exc()
            return f"❌ Error: {str(e)[:50]}"
    
    def _identify_with_model(self, embedding: Optional[np.ndarray], model_name: str, display_name: str) -> str:
        """
        Identify speaker using embeddings from a specific model.
        
        Args:
            embedding: Embedding vector from model (or None if failed)
            model_name: Key name for database (ecapa/titanet/wavlm)
            display_name: Display name (ECAPA-TDNN / TitaNet-Large / WavLM)
            
        Returns:
            Formatted result string for UI
        """
        if embedding is None:
            return f"❌ {display_name}\n\nModel failed to extract embedding"
        
        try:
            # Get all speakers
            speakers = self.database.list_speakers()
            
            best_speaker = None
            best_score = -1
            
            # Compare with each registered speaker
            for speaker in speakers:
                speaker_id = speaker['id']
                
                # Load speaker embedding
                speaker_embedding = self.database.load_embedding(speaker_id, model_name)
                
                if speaker_embedding is None:
                    continue
                
                # Compute cosine similarity
                similarity = self._cosine_similarity(embedding, speaker_embedding)
                
                if similarity > best_score:
                    best_score = similarity
                    best_speaker = speaker
            
            if best_speaker is None or best_score < 0.5:
                return f"❓ {display_name}\n\nUnknown speaker\nScore: {best_score:.1%}"
            
            return f"✅ {display_name}\n\n👤 **{best_speaker['name']}**\nConfidence: **{best_score:.1%}**"
            
        except Exception as e:
            print(f"Identification error: {e}")
            return f"❌ {display_name}\n\nError: {str(e)[:50]}"
    
    @staticmethod
    def _cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
        """Compute cosine similarity between two vectors.
        
        Handles dimension mismatches by flattening vectors.
        Returns similarity in range [0, 1].
        
        Fixes ECAPA dimension error: shapes (1,192) and (1,192) not aligned.
        """
        try:
            # Flatten vectors to handle shape mismatches (1,192) -> (192,)
            vec1 = np.asarray(vec1).flatten()
            vec2 = np.asarray(vec2).flatten()
            
            if len(vec1) == 0 or len(vec2) == 0:
                return 0.0
            
            # Compute normalized dot product
            norm1 = np.linalg.norm(vec1)
            norm2 = np.linalg.norm(vec2)
            
            if norm1 == 0 or norm2 == 0:
                return 0.0
            
            # Cosine similarity: (dot_product / (norm1 * norm2))
            # Returns value in [-1, 1], convert to [0, 1]
            similarity = np.dot(vec1, vec2) / (norm1 * norm2)
            return (similarity + 1) / 2
            
        except Exception as e:
            print(f"Cosine similarity error: {e}")
            return 0.0
    
    def _visualization_callback(self, chunk: np.ndarray):
        """Real-time visualization callback for each audio chunk."""
        try:
            img = generate_equalizer_image(chunk, width=400, height=100)
            self.current_eq = img
        except Exception:
            pass
    
    def streaming_generator_browser(self, audio_input):
        """
        Generator for browser-based audio streaming.
        
        Takes audio chunk from gr.Audio and processes it via Beast Mode.
        Uses weighted fusion for improved identification accuracy.
        
        Args:
            audio_input: Tuple of (sample_rate, audio_data) from gr.Audio
            
        Yields:
            Tuple of (status, eq_image, spectrogram, unified_result)
        """
        speakers = self.database.list_speakers()
        if len(speakers) < 2:
            yield (
                f"❌ Need ≥2 registered speakers (Current: {len(speakers)})",
                None, None,
                "❌ Not enough speakers"
            )
            return
        
        if audio_input is None:
            yield (
                "⏳ Waiting for audio input...",
                self.current_eq,
                self.current_spectrogram,
                self.current_results
            )
            return
        
        try:
            # Parse audio input
            sample_rate, audio_data = audio_input
            audio = audio_data.astype(np.float32) / 32768.0
            
            # Resample to 16kHz if needed
            if sample_rate != 16000:
                audio = librosa.resample(audio, orig_sr=sample_rate, target_sr=16000)
            
            # Apply VAD to filter silence
            if len(audio) > 16000:  # If > 1 second
                ints = librosa.effects.split(audio, top_db=20)
                if len(ints) > 0:
                    audio = np.concatenate([audio[int(s):int(e)] for s, e in ints])
            
            # Need minimum 2 seconds of audio
            if len(audio) < 2 * 16000:
                yield (
                    f"⏳ Audio too short ({len(audio)/16000:.1f}s), need at least 2 seconds",
                    self.current_eq,
                    self.current_spectrogram,
                    self.current_results
                )
                return
            
            # Split audio into chunks and stream
            chunk_size = int(0.1 * 16000)  # 100ms chunks
            num_chunks = len(audio) // chunk_size
            
            for i in range(num_chunks):
                if not self.is_streaming:
                    break
                
                # Process chunk with visualization callback
                chunk = audio[i*chunk_size:(i+1)*chunk_size]
                self._visualization_callback(chunk)
                
                yield (
                    f"🎤 Streaming... Chunk {i+1}/{num_chunks}",
                    self.current_eq,
                    self.current_spectrogram,
                    self.current_results
                )
                
                import time
                time.sleep(0.05)  # Small delay to simulate streaming
            
            # Final processing with full audio
            self._on_buffer_ready(audio)
            
            yield (
                "✅ Processing complete",
                self.current_eq,
                self.current_spectrogram,
                self.current_results
            )
            
        except Exception as e:
            print(f"Streaming error: {e}")
            traceback.print_exc()
            yield (
                f"❌ Error: {str(e)[:60]}",
                None, None,
                f"Error: {str(e)[:40]}"
            )
    
    def start_browser_streaming(self):
        """Mark streaming as started."""
        self.is_streaming = True
        return "🎤 Starting stream..."
    
    def stop_streaming(self):
        """Stop streaming."""
        self.is_streaming = False
        return "⏹️ Streaming stopped"
    
    # ==================== TAB 2: SPEAKER REGISTRATION ====================
    
    def register_speaker(self, name: str, recorded_audio: Optional[Tuple], uploaded_audio: Optional[Tuple]) -> str:
        """
        Register a new speaker.
        
        Extracts embeddings from all 3 models and saves to database.
        """
        if not name or name.strip() == "":
            return "❌ Please enter a speaker name"
        
        try:
            # Choose audio source
            if recorded_audio is not None:
                sample_rate, audio_data = recorded_audio
                audio = audio_data.astype(np.float32) / 32768.0
            elif uploaded_audio is not None:
                sample_rate, audio_data = uploaded_audio
                audio = audio_data.astype(np.float32) / 32768.0
            else:
                return "❌ Please record or upload audio"
            
            # Resample to 16kHz if needed
            if sample_rate != 16000:
                audio = librosa.resample(audio, orig_sr=sample_rate, target_sr=16000)
                sample_rate = 16000
            
            # Remove silence
            if len(audio) > 16000:  # If > 1 second
                ints = librosa.effects.split(audio, top_db=20)
                audio = np.concatenate([audio[int(s):int(e)] for s, e in ints])
            
            # Duration check
            duration = len(audio) / sample_rate
            if duration < 2:
                return f"❌ Audio too short ({duration:.1f}s), need at least 2 seconds"
            if duration < 5:
                # Repeat audio to reach 5 seconds
                repeats = int(5 / duration) + 1
                audio = np.tile(audio, repeats)[:int(5 * 16000)]
            elif duration > 15:
                audio = audio[:int(15 * 16000)]
            
            # Register in database (saves audio file)
            speaker_id = self.database.register_speaker(name.strip(), audio, sample_rate)
            
            # Wait for engine if still loading
            if self.engine is None:
                return f"✅ Speaker registered!\n\n👤 **{name}** (ID: {speaker_id})\n\n⚠️ Engine still loading, embeddings will be extracted soon."
            
            # Extract embeddings from all 3 models
            print(f"Extracting embeddings for speaker '{name}'...")
            embeddings = self.engine.extract_all_embeddings(audio)
            
            # Save each embedding
            for model_name, embedding in embeddings.items():
                if embedding is not None:
                    self.database.save_embedding(speaker_id, embedding, model_name)
                    print(f"  ✓ {model_name}: {embedding.shape}")
                else:
                    print(f"  ✗ {model_name}: Failed")
            
            # Update database metadata with embedding paths
            self.database.metadata['speakers'][speaker_id]['embeddings'] = {
                'ecapa': f"embeddings/{speaker_id}_ecapa.npy",
                'titanet': f"embeddings/{speaker_id}_titanet.npy",
                'wavlm': f"embeddings/{speaker_id}_wavlm.npy"
            }
            self.database._save_metadata()
            
            return f"✅ Speaker registered!\n\n👤 **{name}** (ID: {speaker_id})\n\nEmbeddings extracted and saved."
            
        except Exception as e:
            print(f"Registration error: {e}")
            traceback.print_exc()
            return f"❌ Error during registration: {str(e)}"
    
    def get_speaker_list(self) -> str:
        """Get formatted list of registered speakers."""
        speakers = self.database.list_speakers()
        if len(speakers) == 0:
            return "No speakers registered yet."
        
        lines = [f"**Registered Speakers ({len(speakers)}):**\n"]
        for i, speaker in enumerate(speakers, 1):
            lines.append(f"{i}. {speaker['name']} (ID: {speaker['id']})")
        return "\n".join(lines)
    
    def get_speaker_dataframe(self):
        """Get speaker list as pandas DataFrame for interactive display."""
        speakers = self.database.list_speakers()
        if len(speakers) == 0:
            # Return empty dataframe with correct columns
            return pd.DataFrame({"Speaker ID": [], "Name": []})
        
        df = pd.DataFrame({
            "Speaker ID": [s['id'] for s in speakers],
            "Name": [s['name'] for s in speakers]
        })
        return df
    
    def delete_speaker_callback(self, selected_data):
        """Handle delete button click - show confirmation."""
        if selected_data is None or (isinstance(selected_data, pd.DataFrame) and len(selected_data) == 0):
            return "", gr.update(visible=False)
        
        try:
            # Handle pandas DataFrame selection
            if isinstance(selected_data, pd.DataFrame):
                if len(selected_data) == 0:
                    return "❌ Please select a speaker to delete", gr.update(visible=False)
                # Get first selected row
                selected_row = selected_data.iloc[0]
                speaker_name = selected_row.get("Name", "")
            else:
                # Handle dict format
                speaker_name = selected_data.get("Name", "")
            
            if not speaker_name:
                return "❌ Invalid speaker selection", gr.update(visible=False)
            
            # Show confirmation with speaker name
            confirm_message = f"⚠️ Are you sure you want to delete speaker **{speaker_name}**?\n\nThis will move their data to the trash folder."
            # Store speaker name in a hidden state for confirm button
            return confirm_message, gr.update(visible=True)
            
        except Exception as e:
            print(f"Delete callback error: {e}")
            return f"❌ Error: {str(e)}", gr.update(visible=False)
    
    def confirm_delete_callback(self, selected_data):
        """Handle confirmed delete action."""
        if selected_data is None or (isinstance(selected_data, pd.DataFrame) and len(selected_data) == 0):
            return "❌ No speaker selected", gr.update(), gr.update(visible=False)
        
        try:
            # Handle pandas DataFrame selection
            if isinstance(selected_data, pd.DataFrame):
                if len(selected_data) == 0:
                    return "❌ No speaker selected", gr.update(), gr.update(visible=False)
                selected_row = selected_data.iloc[0]
                speaker_name = selected_row.get("Name", "")
            else:
                speaker_name = selected_data.get("Name", "")
            
            if not speaker_name:
                return "❌ Invalid speaker selection", gr.update(), gr.update(visible=False)
            
            # Execute soft delete
            success, message = self.database.soft_delete_speaker(speaker_name)
            
            # Refresh dataframe
            updated_data = self.get_speaker_dataframe()
            
            return message, gr.update(value=updated_data), gr.update(visible=False)
            
        except Exception as e:
            print(f"Confirm delete error: {e}")
            return f"❌ Error: {str(e)}", gr.update(), gr.update(visible=False)
    
    def cancel_delete_callback(self):
        """Handle delete cancellation."""
        return "", gr.update(visible=False)
    
    # ==================== GRADIO INTERFACE ====================
    
    def create_ui(self) -> gr.Blocks:
        """Create Gradio UI with Beast Mode."""
        
        with gr.Blocks(title="SedaGozar - AI Voice Identification", theme=gr.themes.Soft()) as demo:
            
            gr.Markdown("# 🎤 SedaGozar - AI voice identify")
            gr.Markdown("**State-of-the-Art Multi-Model Engine**")
            
            # Status row - show engine status or loading message
            with gr.Row():
                if self.engine is not None:
                    status_text = (f"📊 Device: {self.engine.get_device().upper()}\n"
                                  f"Models: {sum(self.engine.get_model_status().values())}/4 loaded")
                else:
                    status_text = "⏳ Engine loading in background..."
                
                status = gr.Textbox(
                    value=status_text,
                    label="System Status",
                    interactive=False
                )
            
            # Main tabs
            with gr.Tabs():
                
                # ==================== TAB 1: LIVE IDENTIFICATION ====================
                with gr.Tab("🎤 Live Identification"):
                    
                    # Status display at top
                    with gr.Row():
                        status = gr.Markdown(value="⏳ Ready to stream...", label="Status")
                    
                    with gr.Row():
                        gr.Markdown("**Select microphone from browser and stream audio for identification**")
                    
                    # Browser audio input - user selects microphone
                    with gr.Row():
                        browser_audio = gr.Audio(
                            label="Select Microphone or Record",
                            type="numpy",
                            sources=["microphone"],
                            interactive=True
                        )
                    
                    # Start/Stop buttons
                    with gr.Row():
                        start_btn = gr.Button("▶️ Start Streaming", variant="primary")
                        stop_btn = gr.Button("⏹️ Stop")
                    
                    # Real-time visualizations
                    with gr.Row():
                        with gr.Column(scale=1):
                            eq_display = gr.Image(
                                label="Live Equalizer",
                                type="numpy",
                                interactive=False
                            )
                        with gr.Column(scale=1):
                            spec_display = gr.Image(
                                label="Spectrogram",
                                type="numpy",
                                interactive=False
                            )
                    
                    # Unified speaker identification result (Weighted Fusion)
                    with gr.Row():
                        with gr.Column():
                            result_unified = gr.Markdown(
                                value="📊 Waiting for audio...",
                                label="🎯 Identification Result (Weighted Fusion)"
                            )
                    
                    # Start streaming callback
                    start_btn.click(
                        self.start_browser_streaming,
                        outputs=status
                    ).then(
                        self.streaming_generator_browser,
                        inputs=[browser_audio],
                        outputs=[
                            status,
                            eq_display,
                            spec_display,
                            result_unified
                        ]
                    )
                    
                    # Stop streaming callback
                    stop_btn.click(
                        self.stop_streaming,
                        outputs=status
                    )
                
                # ==================== TAB 2: SPEAKER REGISTRATION ====================
                with gr.Tab("➕ Register Speaker"):
                    
                    with gr.Row():
                        speaker_name = gr.Textbox(
                            label="Speaker Name",
                            placeholder="Enter speaker name"
                        )
                    
                    with gr.Row():
                        gr.Markdown("**Audio Input Options:**")
                    
                    with gr.Row():
                        with gr.Column():
                            recorded_audio = gr.Audio(
                                label="Record Audio (2-15 seconds)",
                                type="numpy",
                                sources=["microphone"]
                            )
                        with gr.Column():
                            uploaded_audio = gr.Audio(
                                label="Or Upload Audio",
                                type="numpy",
                                sources=["upload"]
                            )
                    
                    with gr.Row():
                        register_btn = gr.Button("💾 Save to Database", variant="primary")
                    
                    with gr.Row():
                        reg_output = gr.Markdown(label="Registration Result")
                    
                    # Speaker list with delete functionality
                    with gr.Row():
                        gr.Markdown("**Registered Speakers:**")
                    
                    # Speaker dataframe (interactive)
                    with gr.Row():
                        speaker_dataframe = gr.Dataframe(
                            value=self.get_speaker_dataframe(),
                            interactive=False,
                            label="Speakers",
                            type="pandas",
                            wrap=True
                        )
                    
                    # Delete button row
                    with gr.Row():
                        delete_btn = gr.Button("🗑️ Delete Selected Speaker", variant="stop", scale=2)
                        refresh_btn = gr.Button("🔄 Refresh List", scale=1)
                    
                    # Delete confirmation section (initially hidden)
                    with gr.Group(visible=False) as delete_confirmation:
                        confirm_text = gr.Markdown("")
                        with gr.Row():
                            confirm_yes_btn = gr.Button("✅ Yes, Delete", variant="stop", scale=1)
                            confirm_cancel_btn = gr.Button("❌ Cancel", scale=1)
                    
                    # Delete result message
                    with gr.Row():
                        delete_result = gr.Markdown(label="Delete Result")
                    
                    # Callbacks
                    register_btn.click(
                        self.register_speaker,
                        inputs=[speaker_name, recorded_audio, uploaded_audio],
                        outputs=reg_output
                    )
                    
                    register_btn.click(
                        self.get_speaker_dataframe,
                        outputs=speaker_dataframe
                    )
                    
                    # Delete button triggered - show confirmation
                    delete_btn.click(
                        self.delete_speaker_callback,
                        inputs=[speaker_dataframe],
                        outputs=[confirm_text, delete_confirmation]
                    )
                    
                    # Confirm delete
                    confirm_yes_btn.click(
                        self.confirm_delete_callback,
                        inputs=[speaker_dataframe],
                        outputs=[delete_result, speaker_dataframe, delete_confirmation]
                    )
                    
                    # Cancel delete
                    confirm_cancel_btn.click(
                        self.cancel_delete_callback,
                        outputs=[delete_result, delete_confirmation]
                    )
                    
                    # Refresh list
                    refresh_btn.click(
                        self.get_speaker_dataframe,
                        outputs=speaker_dataframe
                    )
                    
                    # Initial load on page load
                    demo.load(
                        fn=self.get_speaker_dataframe,
                        outputs=speaker_dataframe
                    )
                    
                    # Extra: also load when tab is visible
                    # This ensures refresh on tab change
                    def refresh_on_load():
                        import time
                        time.sleep(0.3)
                        return self.get_speaker_dataframe()
                    
                    # Run refresh on demo load
                    demo.load(
                        fn=refresh_on_load,
                        outputs=speaker_dataframe
                    )
        
        return demo


def create_app() -> gr.Blocks:
    """Create and return Gradio app."""
    app = BeastSpeakerIdentificationApp()
    return app.create_ui()


if __name__ == "__main__":
    demo = create_app()
    try:
        # Try with newer Gradio API
        demo.queue(max_size=20).launch(
            server_name="127.0.0.1",
            server_port=7860,
            share=False
        )
    except TypeError:
        # Fallback for older Gradio versions
        demo.queue().launch(
            server_name="127.0.0.1",
            server_port=7860,
            share=False
        )
