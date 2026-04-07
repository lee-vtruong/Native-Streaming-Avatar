import os
import glob
import csv
import cv2
from core_config import DATASET_DIR, OUTPUT_DIR
from core_models import AIModels
from pipelines import run_sadtalker, run_wav2lip_base, run_stream_v1, run_stream_v2

def measure_fps(video_path, render_time):
    if not os.path.exists(video_path) or render_time <= 0: return 0
    cap = cv2.VideoCapture(video_path)
    frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return round(frames / render_time, 2)

def main():
    audios = glob.glob(f"{DATASET_DIR}/*.wav")
    if not audios:
        print("❌ Không có file wav nào trong assets/dataset/")
        return

    ai_models = AIModels()
    csv_path = "benchmark_ttff_fps_FINAL.csv"
    
    with open(csv_path, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["Audio", "Method", "TTFF (s)", "Total Time (s)", "FPS"])
        
        for idx, audio in enumerate(audios):
            name = os.path.basename(audio).replace(".wav", "")
            print(f"\n[{idx+1}/{len(audios)}] Đang xử lý: {name}")

            # 1. SadTalker
            out_sad = f"{OUTPUT_DIR}/{name}_sadtalker.mp4"
            ttff, total = run_sadtalker(ai_models, audio, out_sad)
            writer.writerow([name, "SadTalker", round(ttff,2), round(total,2), measure_fps(out_sad, total)])

            # 2. Wav2Lip Base
            out_base = f"{OUTPUT_DIR}/{name}_base.mp4"
            ttff, total = run_wav2lip_base(ai_models, audio, out_base)
            writer.writerow([name, "Wav2Lip_Base", round(ttff,2), round(total,2), measure_fps(out_base, total)])

            # 3. Stream V1 (Subprocess)
            out_v1_pref = f"{OUTPUT_DIR}/{name}_v1"
            ttff, total, fin_v1 = run_stream_v1(ai_models, audio, out_v1_pref)
            writer.writerow([name, "Stream_V1", round(ttff,2), round(total,2), measure_fps(fin_v1, total)])

            # 4. Stream V2 (Native)
            out_v2_pref = f"{OUTPUT_DIR}/{name}_v2"
            ttff, total, fin_v2 = run_stream_v2(ai_models, audio, out_v2_pref)
            writer.writerow([name, "Stream_V2", round(ttff,2), round(total,2), measure_fps(fin_v2, total)])
            
            f.flush()

    print(f"\n✅ Benchmark hoàn tất! Đã lưu kết quả tại: {csv_path}")

if __name__ == "__main__":
    main()