import os
import glob
import subprocess
import csv
from core_config import OUTPUT_DIR, BASE_DIR

SYNCNET_DIR = os.path.join(BASE_DIR, "syncnet_python")
CSV_PATH = "benchmark_syncnet_lse.csv"

def main():
    # Quét các video final của 4 version
    videos = glob.glob(f"{OUTPUT_DIR}/*_sadtalker.mp4") + \
             glob.glob(f"{OUTPUT_DIR}/*_base.mp4") + \
             glob.glob(f"{OUTPUT_DIR}/*_v1_final.mp4") + \
             glob.glob(f"{OUTPUT_DIR}/*_v2_final.mp4")
             
    if not videos:
        print("❌ Không có video nào để chấm điểm!")
        return

    with open(CSV_PATH, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["Video File", "LSE-D (Distance)", "LSE-C (Confidence)"])
        
        for idx, vid in enumerate(videos):
            name = os.path.basename(vid).split('.')[0]
            print(f"[{idx+1}/{len(videos)}] Chấm điểm: {name}")
            
            subprocess.run(["python", "run_pipeline.py", "--videofile", vid, "--reference", name, "--data_dir", "data/work"],
                           cwd=SYNCNET_DIR, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            
            res = subprocess.run(["python", "run_syncnet.py", "--videofile", vid, "--reference", name, "--data_dir", "data/work"],
                                 cwd=SYNCNET_DIR, capture_output=True, text=True)
            
            lse_d, lse_c = "N/A", "N/A"
            for line in res.stdout.split('\n'):
                if "Min dist" in line: lse_d = line.split(":")[-1].strip()
                if "Confidence" in line: lse_c = line.split(":")[-1].strip()
                
            writer.writerow([name, lse_d, lse_c])
            f.flush()

    print(f"\n✅ Đánh giá SyncNet hoàn tất! Lưu tại: {CSV_PATH}")

if __name__ == "__main__":
    main()