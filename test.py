import os
import torch
import torchaudio
import logging
import gc

# قطع دسترسی‌های مخفی به اینترنت توسط کتابخانه‌ها
os.environ['TRANSFORMERS_OFFLINE'] = '1'
os.environ['HF_DATASETS_OFFLINE'] = '1'

# رفع مشکل تورچ‌اودیو برای SpeechBrain
if not hasattr(torchaudio, 'list_audio_backends'):
    torchaudio.list_audio_backends = lambda: ['ffmpeg', 'sox', 'soundfile']

import speechbrain.inference.speaker as sb_speaker
from transformers import WavLMModel, Wav2Vec2FeatureExtractor
import nemo.collections.asr as nemo_asr

logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(message)s')
logger = logging.getLogger("BeastOffline")

class OfflineBeastEngine:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.models = {}
        self.load_all_offline()

    def load_all_offline(self):
        logger.info(f"--- Starting Offline Mode on {self.device} ---")
        
        try:
            # ۱. Silero VAD - لود مستقیم از پوشه پروژه
            logger.info("Loading Layer 1: VAD (Local Source)")
            self.models['vad'], _ = torch.hub.load(
                repo_or_dir='./models/silero_vad_local', 
                model='silero_vad', 
                source='local', 
                trust_repo=True
            )
            self.models['vad'] = self.models['vad'].to(self.device)

            # ۲. ECAPA-TDNN - لود از پوشه پروژه (بدون چک کردن HF)
            logger.info("Loading Layer 2: ECAPA (Local Source)")
            self.models['ecapa'] = sb_speaker.EncoderClassifier.from_hparams(
                source="./models/ecapa_voxceleb",
                run_opts={"device": str(self.device)},
                savedir="./models/ecapa_voxceleb"
            )

            # ۳. TitaNet - لود از فایل .nemo
            logger.info("Loading Layer 3: TitaNet (Local File)")
            self.models['titanet'] = nemo_asr.models.EncDecSpeakerLabelModel.restore_from(
                "./nemo_models/speakerverification_en_titanet_large.nemo"
            ).to(self.device)

            # ۴. WavLM - لود با فلگ اجباری Offline
            logger.info("Loading Layer 4: WavLM (Local Directory)")
            self.models['wavlm_feat'] = Wav2Vec2FeatureExtractor.from_pretrained(
                "./wavlm_model", local_files_only=True
            )
            self.models['wavlm_model'] = WavLMModel.from_pretrained(
                "./wavlm_model", local_files_only=True
            ).to(self.device)

            self.models['ecapa'].eval()
            self.models['titanet'].eval()
            self.models['wavlm_model'].eval()

            logger.info("🔥🔥 BEAST MODE READY: 100% OFFLINE & LOADED 🔥🔥")
            
            if torch.cuda.is_available():
                used_mem = torch.cuda.memory_reserved() / 1024**3
                logger.info(f"VRAM Reserved: {used_mem:.2f} GB")

        except Exception as e:
            logger.error(f"❌ Critical Error in Offline Load: {str(e)}")

if __name__ == "__main__":
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        
    engine = OfflineBeastEngine()