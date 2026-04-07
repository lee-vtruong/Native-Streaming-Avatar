import os
import torch
import gc
from transformers import AutoModelForCausalLM, AutoTokenizer

class AIModels:
    def __init__(self):
        import whisper
        torch.backends.cudnn.enabled = False
        self.stt = whisper.load_model("large-v3-turbo").to("cuda:0")
        torch.backends.cudnn.enabled = True
        
        self.llm_id = "Qwen/Qwen2.5-7B-Instruct"
        self.tokenizer = AutoTokenizer.from_pretrained(self.llm_id)
        self.llm = AutoModelForCausalLM.from_pretrained(self.llm_id, torch_dtype=torch.float16, device_map="cuda:0")

    def transcribe(self, audio_path):
        torch.backends.cudnn.enabled = False
        res = self.stt.transcribe(audio_path, language="vi")
        torch.backends.cudnn.enabled = True
        return res["text"].strip()

    def cleanup(self):
        gc.collect()
        torch.cuda.empty_cache()

def generate_tts(text, out_path):
    import subprocess
    temp_mp3 = out_path.replace(".wav", ".mp3")

    subprocess.run(["edge-tts", "--voice", "vi-VN-HoaiMyNeural", "--text", text, "--write-media", temp_mp3], 
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    subprocess.run(["ffmpeg", "-y", "-i", temp_mp3, "-ar", "16000", out_path], 
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    if os.path.exists(temp_mp3): os.remove(temp_mp3)