"""
Speaker database management using SQLite.

Provides unified database storage with speaker metadata,
audio file references, and embedding paths.
"""

import sqlite3
import numpy as np
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import soundfile as sf
from datetime import datetime


class DatabaseManager:
    """
    Manages speaker database using SQLite with file storage.
    
    Database Schema:
    - speakers table: id, name, speaker_id, audio_path, register_date
    - embeddings table: speaker_id, model_name, embedding_path
    """
    
    def __init__(self, data_dir: str = "data"):
        """
        Initialize SQLite database manager.
        
        Args:
            data_dir: Root directory for all speaker data
        """
        self.data_dir = Path(data_dir)
        self.audio_dir = self.data_dir / "audio"
        self.embeddings_dir = self.data_dir / "embeddings"
        self.db_path = self.data_dir / "speakers.db"
        
        # Create directories
        self.audio_dir.mkdir(parents=True, exist_ok=True)
        self.embeddings_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize database
        self._init_database()
        
        print(f"✅ Database initialized at {self.db_path}")
    
    def _init_database(self):
        """Initialize SQLite database with tables."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Speakers table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS speakers (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                speaker_id TEXT UNIQUE NOT NULL,
                name TEXT NOT NULL,
                audio_path TEXT NOT NULL,
                register_date TEXT NOT NULL,
                deleted BOOLEAN DEFAULT 0
            )
        ''')
        
        # Embeddings table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS embeddings (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                speaker_id TEXT NOT NULL,
                model_name TEXT NOT NULL,
                embedding_path TEXT NOT NULL,
                FOREIGN KEY (speaker_id) REFERENCES speakers(speaker_id)
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def _get_next_speaker_id(self) -> str:
        """Get next available speaker ID."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('SELECT MAX(CAST(SUBSTR(speaker_id, 9) AS INTEGER)) FROM speakers WHERE deleted=0')
        result = cursor.fetchone()[0]
        conn.close()
        
        next_id = (result or 0) + 1
        return f"speaker_{next_id:03d}"
    
    def register_speaker(self, name: str, audio: np.ndarray, 
                        sample_rate: int = 16000) -> str:
        """
        Register a new speaker in database.
        
        Args:
            name: Speaker name
            audio: Audio array (16kHz, mono)
            sample_rate: Sample rate (default 16000)
            
        Returns:
            Speaker ID
        """
        speaker_id = self._get_next_speaker_id()
        audio_path = self.audio_dir / f"{speaker_id}.wav"
        
        # Save audio file
        sf.write(audio_path, audio, sample_rate)
        
        # Insert into database
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO speakers (speaker_id, name, audio_path, register_date)
            VALUES (?, ?, ?, ?)
        ''', (speaker_id, name, str(audio_path), datetime.now().isoformat()))
        
        conn.commit()
        conn.close()
        
        print(f"✅ Registered speaker '{name}' with ID {speaker_id}")
        return speaker_id
    
    def save_embedding(self, speaker_id: str, embedding: np.ndarray, model_name: str):
        """
        Save embedding for a speaker.
        
        Args:
            speaker_id: Speaker ID
            embedding: Embedding array
            model_name: Model name (ecapa, titanet, wavlm)
        """
        embedding_path = self.embeddings_dir / f"{speaker_id}_{model_name}.npy"
        
        # Save embedding file
        np.save(embedding_path, embedding)
        
        # Insert into database (or update if exists)
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Delete existing embedding first
        cursor.execute('''
            DELETE FROM embeddings 
            WHERE speaker_id=? AND model_name=?
        ''', (speaker_id, model_name))
        
        # Insert new embedding
        cursor.execute('''
            INSERT INTO embeddings (speaker_id, model_name, embedding_path)
            VALUES (?, ?, ?)
        ''', (speaker_id, model_name, str(embedding_path)))
        
        conn.commit()
        conn.close()
    
    def load_embedding(self, speaker_id: str, model_name: str) -> Optional[np.ndarray]:
        """
        Load embedding for a speaker.
        
        Args:
            speaker_id: Speaker ID
            model_name: Model name (ecapa, titanet, wavlm)
            
        Returns:
            Embedding array or None if not found
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT embedding_path FROM embeddings 
            WHERE speaker_id=? AND model_name=?
        ''', (speaker_id, model_name))
        
        result = cursor.fetchone()
        conn.close()
        
        if result:
            embedding_path = Path(result[0])
            if embedding_path.exists():
                return np.load(embedding_path)
        
        return None
    
    def get_speaker_audio(self, speaker_id: str) -> Tuple[np.ndarray, int]:
        """
        Load audio for a speaker.
        
        Args:
            speaker_id: Speaker ID
            
        Returns:
            Tuple of (audio_data, sample_rate)
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('SELECT audio_path FROM speakers WHERE speaker_id=? AND deleted=0', (speaker_id,))
        result = cursor.fetchone()
        conn.close()
        
        if result:
            audio_path = Path(result[0])
            if audio_path.exists():
                audio, sr = sf.read(audio_path)
                return audio, sr
        
        return None, None
    
    def list_speakers(self) -> List[Dict]:
        """
        List all active speakers.
        
        Returns:
            List of speaker dictionaries
        """
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT speaker_id, name, register_date 
            FROM speakers 
            WHERE deleted=0 
            ORDER BY register_date DESC
        ''')
        
        speakers = [dict(row) for row in cursor.fetchall()]
        conn.close()
        
        return speakers
    
    def soft_delete_speaker(self, speaker_name: str) -> Tuple[bool, str]:
        """
        Soft-delete a speaker (mark as deleted in DB).
        
        Args:
            speaker_name: Speaker name to delete
            
        Returns:
            Tuple of (success, message)
        """
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Find speaker by name
            cursor.execute('SELECT speaker_id FROM speakers WHERE name=? AND deleted=0', (speaker_name,))
            result = cursor.fetchone()
            
            if not result:
                conn.close()
                return False, f"❌ Speaker '{speaker_name}' not found"
            
            speaker_id = result[0]
            
            # Mark as deleted instead of removing
            cursor.execute('''
                UPDATE speakers SET deleted=1 WHERE speaker_id=?
            ''', (speaker_id,))
            
            conn.commit()
            conn.close()
            
            return True, f"✅ Speaker '{speaker_name}' deleted successfully"
            
        except Exception as e:
            return False, f"❌ Error deleting speaker: {str(e)}"
    
    def delete_speaker(self, speaker_id: str):
        """
        Permanently delete a speaker (mark as deleted).
        
        Args:
            speaker_id: Speaker ID to delete
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('UPDATE speakers SET deleted=1 WHERE speaker_id=?', (speaker_id,))
        
        conn.commit()
        conn.close()
    
    def get_stats(self) -> Dict:
        """Get database statistics."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('SELECT COUNT(*) FROM speakers WHERE deleted=0')
        total_speakers = cursor.fetchone()[0]
        
        cursor.execute('SELECT COUNT(*) FROM embeddings')
        total_embeddings = cursor.fetchone()[0]
        
        conn.close()
        
        return {
            'total_speakers': total_speakers,
            'total_embeddings': total_embeddings
        }
    
    # Compatibility properties for old code
    @property
    def metadata(self):
        """Return metadata for backward compatibility."""
        speakers_list = self.list_speakers()
        speakers_dict = {}
        for speaker in speakers_list:
            speakers_dict[speaker['speaker_id']] = {
                'name': speaker['name'],
                'id': speaker['speaker_id']
            }
        return {'speakers': speakers_dict}
    
    def _save_metadata(self):
        """No-op for backward compatibility."""
        pass
