# Speaker Identification System
# تشخیص هویت گوینده از روی صدا

**A comprehensive speaker identification system with three parallel model approaches**

## 📋 Overview

این پروژه یک سیستم کامل تشخیص هویت گوینده با استفاده از سه روش مختلف است:

1. **Classic Model**: MFCC features + SVM classifier
2. **Semi-Professional Model**: ECAPA-TDNN deep learning embeddings  
3. **Deep Dual Model**: ECAPA-TDNN + X-Vector (parallel deep models)

## 🎯 Features

- ✅ Live audio streaming from microphone (16kHz, mono)
- ✅ 5-second windowed processing
- ✅ Real-time speaker identification
- ✅ Mel spectrogram visualization
- ✅ Speaker registration interface
- ✅ GPU acceleration (with CPU fallback)
- ✅ Three parallel identification approaches
- ✅ Confidence scoring for all models

## 🏗️ Project Structure

```
SedaGozar/
├── src/
│   ├── audio/              # Audio capture and recording
│   │   ├── stream.py       # Live streaming with 5-sec buffers
│   │   └── recorder.py     # Recording for registration
│   ├── features/           # Feature extraction
│   │   ├── mfcc.py         # MFCC extraction (40-dim)
│   │   └── spectrogram.py  # Mel spectrogram visualization
│   ├── models/             # Speaker identification models
│   │   ├── classic.py      # MFCC + SVM
│   │   ├── semipro.py      # ECAPA-TDNN embeddings
│   │   └── deep.py         # Dual deep models
│   ├── database/           # Speaker data management
│   │   └── manager.py      # Database operations
│   └── ui/                 # User interface
│       └── gradio_app.py   # Gradio web interface
├── data/                   # Speaker database
│   ├── audio/              # Speaker audio samples
│   ├── features/           # MFCC features
│   └── embeddings/         # Deep learning embeddings
├── main.py                 # Application entry point
├── requirements.txt        # Python dependencies
└── README.md               # This file
```

## 🚀 Installation

### Prerequisites

- Python 3.8 or higher
- Microphone access
- (Optional) NVIDIA GPU with CUDA for faster processing

### Step 1: Clone/Download the Project

```bash
cd c:\Users\Mehdi\PycharmProjects\SedaGozar
```

### Step 2: Create Virtual Environment (Recommended)

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# Linux/Mac
source venv/bin/activate
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

**Note for Windows users**: If PyAudio installation fails, download the wheel from:
https://www.lfd.uci.edu/~gohlke/pythonlibs/#pyaudio

Then install with:
```bash
pip install PyAudio‑0.2.XX‑cpXX‑cpXX‑win_amd64.whl
```

## 💻 Usage

### Running the Application

```bash
python main.py
```

The application will start and open at: **http://localhost:7860**

### Tab 1: Live Speaker Identification

1. Click **"Start Recording"**
2. Speak into the microphone
3. Every 5 seconds:
   - System processes audio
   - Updates spectrogram visualization
   - Shows predictions from all three models
4. Click **"Get Latest Results"** to see current predictions
5. Click **"Stop Recording"** when done

### Tab 2: Speaker Registration

1. Set recording duration (3-10 seconds, recommended: 5 seconds)
2. Click **"Record Sample"**
3. Speak naturally (different sentences, not the same phrase)
4. Enter the speaker's name
5. Click **"Save Speaker"**
6. System automatically trains all models with new speaker

## 📊 Model Comparison

### Classic Model (MFCC + SVM)

**Approach**: Extracts 40 MFCC coefficients, computes mean and variance (80 features), trains SVM classifier

**Strengths**:
- Fast (no GPU needed)
- Works with small datasets (2-5 speakers)
- Interpretable features
- Low memory footprint

**Weaknesses**:
- Sensitive to background noise
- Microphone variation affects accuracy
- Limited robustness

**Confidence Score**: 0-100% probability from SVM

### Semi-Professional Model (ECAPA-TDNN)

**Approach**: Uses pretrained ECAPA-TDNN model to extract 192-dim embeddings, identifies via cosine similarity

**Strengths**:
- Much more robust to noise
- Pretrained on 1000s of hours (VoxCeleb)
- Better generalization
- No feature engineering needed

**Weaknesses**:
- Requires GPU for real-time performance (works on CPU)
- Less interpretable
- Domain mismatch possible

**Similarity Score**: 0-100% (cosine similarity normalized)

### Deep Dual Model (ECAPA + X-Vector)

**Approach**: Runs TWO deep models in parallel (ECAPA-TDNN + X-Vector), shows independent predictions

**Strengths**:
- Highest accuracy
- Maximum robustness
- Redundancy (if one fails, other may succeed)
- Can detect uncertain predictions (disagreement)

**Weaknesses**:
- 2x computational cost
- 2x memory requirement
- Slower inference
- Overkill for small speaker sets

**Scores**: Two independent scores (0-100%) from each model

## 🎯 Confidence Score Interpretation

| Range | Meaning | Action |
|-------|---------|--------|
| >85% | Very high confidence | Strongly believe identification is correct ✅ |
| 70-85% | High confidence | Likely correct ✅ |
| 60-70% | Moderate confidence | Uncertain ⚠️ |
| <60% | Low confidence | Unreliable, possibly unknown speaker ❌ |

## ⚠️ Limitations

### 1. **Noise Sensitivity**
   - Classic model: Most sensitive
   - Semi-pro model: More robust
   - Deep models: Most robust
   - Recommendation: Record in quiet environment

### 2. **Microphone Variation**
   - Different microphones between registration and identification reduce accuracy
   - Recommendation: Use same microphone for consistency

### 3. **Speaker Health/Emotion**
   - Voice changes due to illness, stress, or emotion affect performance
   - Recommendation: Re-register if voice characteristics change significantly

### 4. **Short Utterances**
   - System needs ~2-3 seconds of actual speech in 5-second buffer
   - Silence or very short speech produces unreliable results
   - Recommendation: Speak continuously during recording

### 5. **Number of Speakers**
   - Minimum: 2 speakers
   - Optimal: 3-20 speakers
   - More speakers increase confusion between similar voices

## 🔧 Troubleshooting

### PyAudio Installation Issues

**Windows**: Download wheel from https://www.lfd.uci.edu/~gohlke/pythonlibs/#pyaudio

**Linux**: 
```bash
sudo apt-get install portaudio19-dev
pip install pyaudio
```

**Mac**:
```bash
brew install portaudio
pip install pyaudio
```

### GPU Not Detected

Check CUDA installation:
```python
import torch
print(torch.cuda.is_available())
```

If False but GPU exists, reinstall PyTorch with CUDA support:
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### Models Not Loading

Ensure SpeechBrain is installed:
```bash
pip install speechbrain
```

First run will download pretrained models (~500MB), this is normal.

## 📖 Technical Details

### Feature Extraction

**MFCC (Mel-Frequency Cepstral Coefficients)**:
- 40 coefficients extracted using librosa
- Mean and variance computed over time → 80-dimensional vector
- Captures spectral envelope (vocal tract characteristics)

**Mel Spectrogram**:
- 128 Mel bands
- Time-frequency representation
- Visualizes audio content

### Models Architecture

**ECAPA-TDNN**:
- Emphasized Channel Attention
- Res2Net backbone
- Temporal pooling
- 192-dim embeddings

**X-Vector**:
- Frame-level TDNN
- Statistics pooling
- Speaker embeddings
- 512-dim (in SpeechBrain)

### Similarity Metric

Cosine similarity between normalized embeddings:
```
similarity = (emb1 · emb2) / (||emb1|| * ||emb2||)
```

Range: [-1, 1], converted to [0, 100]%

## 📚 Scientific Background

This project implements multiple speaker recognition paradigms:

1. **Traditional ML**: Feature engineering (MFCCs) + discriminative classifier (SVM)
2. **Deep Learning**: End-to-end learned embeddings (ECAPA-TDNN)
3. **Ensemble**: Multiple model consensus for robustness

**Key Papers**:
- ECAPA-TDNN: "ECAPA-TDNN: Emphasized Channel Attention..." (Interspeech 2020)
- X-Vector: "X-Vectors: Robust DNN Embeddings..." (ICASSP 2018)
- MFCC: Classic speech processing (Davis & Mermelstein, 1980)

## 🎓 Academic Use

This project is suitable for university presentations/reports. Key academic aspects:

- ✅ Multiple approaches comparison (classic vs deep learning)
- ✅ Detailed scientific rationale in code comments
- ✅ Explainable confidence scores
- ✅ Documented limitations and failure cases
- ✅ Real-world applicability demonstration

## 🤝 Contributing

این پروژه برای اهداف آموزشی و تحقیقاتی طراحی شده است.

## 📄 License

Educational/Research use only.

## 👨‍💻 Development

### Adding New Models

1. Create new file in `src/models/`
2. Implement `predict()` method
3. Add to `gradio_app.py` interface
4. Update README

### Adding New Features

- Feature extractors go in `src/features/`
- Follow existing pattern with caching
- Document scientific rationale

## 🔗 Resources

- SpeechBrain: https://speechbrain.github.io/
- VoxCeleb Dataset: https://www.robots.ox.ac.uk/~vgg/data/voxceleb/
- Librosa Documentation: https://librosa.org/

---

**Made with ❤️ for Speaker Recognition Research**
