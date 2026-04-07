import os
import time
import glob
import shutil
import threading
import queue
import subprocess
from transformers import TextIteratorStreamer
from core_config import BASE_DIR, OPTIMIZED_FACE, OUTPUT_DIR
from core_models import generate_tts
from core_wav2lip import run_wav2lip_native, run_wav2lip_subprocess, merge_audio

# --- V1: SADTALKER (Sequential Baseline 2) ---
def run_sadtalker(ai_models, audio_path, out_video_path):
    t0 = time.time()
    
    # 1. STT (Nghe)
    text = ai_models.transcribe(audio_path)
    
    # 2. LLM (Nghĩ - Cùng 1 prompt và tham số với Wav2Lip Base)
    messages = [
        {"role": "system", "content": "Bạn là AI trợ lý giao tiếp bằng giọng nói. Hãy trả lời câu hỏi của người dùng một cách rõ ràng, tự nhiên bằng tiếng Việt. Trả lời đầy đủ ý nhưng súc tích (từ 2 đến 3 câu). Không được để câu bị cụt hoặc dang dở."},
        {"role": "user", "content": text}
    ]
    prompt = ai_models.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = ai_models.tokenizer(prompt, return_tensors="pt").to("cuda:0")
    
    outputs = ai_models.llm.generate(**inputs, max_new_tokens=250, temperature=0.7, top_p=0.9)
    response = ai_models.tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True).strip()
    
    # 3. TTS (Nói)
    temp_wav = os.path.join(OUTPUT_DIR, "temp_sadtalker.wav")
    generate_tts(response, temp_wav)
    
    # 4. SadTalker Render (Nhép môi bằng audio của AI)
    sadtalker_dir = os.path.join(BASE_DIR, "SadTalker")
    temp_dir = os.path.join(OUTPUT_DIR, "sadtalker_temp")
    os.makedirs(temp_dir, exist_ok=True)
    
    cmd = ["python", "inference.py", "--driven_audio", os.path.abspath(temp_wav),
           "--source_image", os.path.abspath(OPTIMIZED_FACE), "--result_dir", os.path.abspath(temp_dir),
           "--still", "--preprocess", "crop"]
    # subprocess.run(cmd, cwd=sadtalker_dir, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    # Thay vì giấu lỗi, hãy để nó hiện ra
    subprocess.run(cmd, cwd=sadtalker_dir)
    
    gen_vids = glob.glob(f"{temp_dir}/**/*.mp4", recursive=True)
    if gen_vids: 
        shutil.move(gen_vids[0], out_video_path)
        
    # Xóa thư mục rác của SadTalker
    shutil.rmtree(temp_dir, ignore_errors=True)
    if os.path.exists(temp_wav): os.remove(temp_wav)
    
    total_t = time.time() - t0
    return total_t, total_t # Đối với hệ thống Tuần tự: TTFF = Total Time

# --- V2: WAV2LIP BASELINE (Sequential) ---
def run_wav2lip_base(ai_models, audio_path, out_video_path):
    t0 = time.time()
    text = ai_models.transcribe(audio_path)
    
    # ----------------------------------------------------
    # SYSTEM PROMPT CHO BASELINE (Sequential)
    # ----------------------------------------------------
    messages = [
        {"role": "system", "content": "Bạn là AI trợ lý giao tiếp bằng giọng nói. Hãy trả lời câu hỏi của người dùng một cách rõ ràng, tự nhiên bằng tiếng Việt. Trả lời đầy đủ ý nhưng súc tích (từ 2 đến 3 câu). Không được để câu bị cụt hoặc dang dở."},
        {"role": "user", "content": text}
    ]
    prompt = ai_models.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = ai_models.tokenizer(prompt, return_tensors="pt").to("cuda:0")
    
    # Cho phép dài tối đa 250 tokens để không bị cắt ngang
    outputs = ai_models.llm.generate(**inputs, max_new_tokens=250, temperature=0.7, top_p=0.9)
    response = ai_models.tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True).strip()
    
    temp_wav = os.path.join(OUTPUT_DIR, "temp_base.wav")
    generate_tts(response, temp_wav)
    
    run_wav2lip_subprocess(OPTIMIZED_FACE, temp_wav, out_video_path)
    merge_audio(out_video_path, temp_wav)
    
    total_t = time.time() - t0
    return total_t, total_t # TTFF = Total Time

# --- V3 & V4: ASYNC STREAMING WORKERS ---
def _stream_pipeline(ai_models, audio_path, out_prefix, mode="native"):
    t0 = time.time()
    text = ai_models.transcribe(audio_path)
    
    text_q, audio_q = queue.Queue(), queue.Queue()
    ttff_data = {"ttff": 0.0}

    def tts_worker():
        idx = 0
        while True:
            t = text_q.get()
            if t is None:
                audio_q.put(None); text_q.task_done(); break
            chunk_wav = f"{out_prefix}_audio_{idx}.wav"
            generate_tts(t, chunk_wav)
            if os.path.exists(chunk_wav):
                audio_q.put(chunk_wav)
                idx += 1
            text_q.task_done()

    def w2l_worker():
        idx = 0
        while True:
            a = audio_q.get()
            if a is None:
                audio_q.task_done(); break
            v = f"{out_prefix}_chunk_{idx}.mp4"
            
            if mode == "native": run_wav2lip_native(OPTIMIZED_FACE, a, v)
            else: run_wav2lip_subprocess(OPTIMIZED_FACE, a, v)
            
            merge_audio(v, a)
            
            if idx == 0: ttff_data["ttff"] = time.time() - t0
            idx += 1
            audio_q.task_done()

    threading.Thread(target=tts_worker, daemon=True).start()
    threading.Thread(target=w2l_worker, daemon=True).start()

    # ----------------------------------------------------
    # SYSTEM PROMPT CHO STREAMING (V3 & V4)
    # ----------------------------------------------------
    messages = [
        {"role": "system", "content": "Bạn là trợ lý AI tên là Hoài My, thông minh và thân thiện. Hãy trả lời ngắn gọn, lưu loát bằng tiếng Việt. Độ dài khoảng 2 đến 3 câu. Hoàn thành câu trọn vẹn, tuyệt đối không được nói cụt lủn."},
        {"role": "user", "content": text}
    ]
    prompt = ai_models.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = ai_models.tokenizer(prompt, return_tensors="pt").to("cuda:0")

    streamer = TextIteratorStreamer(ai_models.tokenizer, skip_prompt=True, skip_special_tokens=True)
    
    # Bơm thông số vào quá trình Generate
    generation_kwargs = dict(
        **inputs, 
        streamer=streamer, 
        max_new_tokens=250,      # Chống nói nửa chừng
        temperature=0.7,         # Độ tự nhiên
        top_p=0.9,               # Lọc từ vô nghĩa
        repetition_penalty=1.1   # Chống lặp từ
    )
    
    threading.Thread(target=ai_models.llm.generate, kwargs=generation_kwargs).start()

    curr = ""
    for chunk in streamer:
        curr += chunk
        # Chỉ tách chunk khi gặp dấu ngắt câu để ngữ điệu đọc của TTS mượt mà nhất
        if any(p in chunk for p in ['.', ',', '?', '!', '\n']):
            cl = curr.strip()
            if len(cl) > 3: text_q.put(cl)
            curr = ""
            
    if curr.strip(): text_q.put(curr.strip())
    text_q.put(None)
    
    text_q.join()
    audio_q.join()
    
    # Nối các chunks lại thành 1 video final để benchmark FPS và SyncNet
    final_video = f"{out_prefix}_final.mp4"
    list_txt = f"{out_prefix}_list.txt"
    chunks = sorted(glob.glob(f"{out_prefix}_chunk_*.mp4"), key=lambda x: int(x.split('_')[-1].split('.')[0]))
    
    if chunks:
        with open(list_txt, "w") as f:
            for c in chunks: f.write(f"file '{os.path.abspath(c)}'\n")
        subprocess.run(["ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", list_txt, "-c", "copy", final_video, "-loglevel", "quiet"])
        os.remove(list_txt)

    total_t = time.time() - t0
    return ttff_data["ttff"], total_t, final_video

def run_stream_v1(ai_models, audio_path, out_prefix):
    return _stream_pipeline(ai_models, audio_path, out_prefix, mode="subprocess")

def run_stream_v2(ai_models, audio_path, out_prefix):
    return _stream_pipeline(ai_models, audio_path, out_prefix, mode="native")