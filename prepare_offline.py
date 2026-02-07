import os
import shutil
from pathlib import Path

def setup_offline_folders():
    # ایجاد ساختار پوشه‌بندی لوکال
    paths = [
        "models/ecapa_voxceleb",
        "models/silero_vad_local",
        "nemo_models",
        "wavlm_model"
    ]
    for p in paths:
        os.makedirs(p, exist_ok=True)
        print(f"✅ Folder ready: {p}")

    # ۱. کپی کردن مدل ECAPA از کش هگینگ‌فیس به پروژه
    hf_cache = Path(os.path.expanduser("~/.cache/huggingface/hub"))
    ecapa_cache = list(hf_cache.glob("models--speechbrain--spkrec-ecapa-voxceleb/snapshots/*"))
    
    if ecapa_cache:
        latest_snapshot = ecapa_cache[-1]
        for item in os.listdir(latest_snapshot):
            s = os.path.join(latest_snapshot, item)
            d = os.path.join("models/ecapa_voxceleb", item)
            if os.path.isfile(s):
                shutil.copy2(s, d)
        print("✅ ECAPA models copied to local project.")
    else:
        print("❌ ECAPA cache not found. Please run the online test once first.")

    # ۲. کپی کردن VAD (فرض بر این است که کش شده)
    vad_cache = Path(os.path.expanduser("~/.cache/torch/hub/snakers4_silero-vad_master"))
    if vad_cache.exists():
        if os.path.exists("models/silero_vad_local"):
            shutil.rmtree("models/silero_vad_local")
        shutil.copytree(vad_cache, "models/silero_vad_local")
        print("✅ Silero VAD source copied to local project.")

if __name__ == "__main__":
    setup_offline_folders()