import os
import sys
import argparse
import torch
import subprocess

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
WAV2LIP_DIR = os.path.join(BASE_DIR, "Wav2Lip")
if WAV2LIP_DIR not in sys.path:
    sys.path.insert(0, WAV2LIP_DIR)

# --- LƯU LẠI THAM SỐ THẬT (Bạn bị thiếu dòng này ban nãy) ---
old_argv = sys.argv

# Tham số giả lừa Wav2Lip
sys.argv = [
    'inference.py', 
    '--checkpoint_path', 'dummy.pth', 
    '--face', 'dummy.jpg', 
    '--audio', 'dummy.wav'
]

import inference as w2l_infer

# Trả lại tham số thật
sys.argv = old_argv

from core_config import WAV2LIP_ARGS

def run_wav2lip_native(face_path, audio_path, out_path):
    face_path = os.path.abspath(face_path)
    audio_path = os.path.abspath(audio_path)
    out_path = os.path.abspath(out_path)

    # --- VIÊN ĐẠN BẠC: Tắt cuDNN để né lỗi phân mảnh VRAM do AIModels gây ra ---
    import torch
    torch.backends.cudnn.enabled = False
    # -------------------------------------------------------------------------

    wav2lip_dir = os.path.join(BASE_DIR, "Wav2Lip")
    if wav2lip_dir not in sys.path: sys.path.insert(0, wav2lip_dir)
    original_cwd = os.getcwd()
    os.chdir(wav2lip_dir)
    
    args_dict = WAV2LIP_ARGS.copy()
    # Nhồi đầy đủ tham số an toàn nhất
    args_dict.update({
        "face": face_path, 
        "audio": audio_path, 
        "outfile": out_path, 
        "resize_factor": 2, 
        "nosmooth": True,
        "face_det_batch_size": 1
    })
    w2l_infer.args = argparse.Namespace(**args_dict)
    
    try: 
        w2l_infer.main()
    except Exception as e: 
        print(f"Lỗi Native: {e}")
        
    os.chdir(original_cwd)
    
    # Bật lại cuDNN cho các tiến trình khác và dọn rác
    torch.backends.cudnn.enabled = True
    torch.cuda.empty_cache()

def run_wav2lip_subprocess(face_path, audio_path, out_path):
    wav2lip_dir = os.path.join(BASE_DIR, "Wav2Lip")
    cmd = [
        "python", "inference.py", 
        "--checkpoint_path", "checkpoints/wav2lip_gan.pth",
        "--face", os.path.abspath(face_path),
        "--audio", os.path.abspath(audio_path),
        "--outfile", os.path.abspath(out_path),
        "--nosmooth", "--wav2lip_batch_size", "64",
        "--face_det_batch_size", "1", "--resize_factor", "2"
    ]
    subprocess.run(cmd, cwd=wav2lip_dir, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

def merge_audio(video_path, audio_path):
    if not os.path.exists(video_path): return
    temp = video_path.replace(".mp4", "_silent.mp4")
    os.rename(video_path, temp)
    subprocess.run(["ffmpeg", "-y", "-i", temp, "-i", audio_path, "-map", "0:v:0", "-map", "1:a:0", 
                    "-c:v", "copy", "-c:a", "aac", video_path, "-loglevel", "quiet"])
    if os.path.exists(temp): os.remove(temp)