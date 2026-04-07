import csv
import statistics
from collections import defaultdict

# --- ĐIỀN TÊN 2 FILE FINAL CỦA BẠN VÀO ĐÂY ---
ttff_file = "benchmark_ttff_fps_FINAL.csv"
sync_file = "benchmark_syncnet_lse.csv" # Hoặc đổi thành tên file LSE cuối cùng của bạn
output_file = "benchmark_summary_report.csv" # File CSV đầu ra

# Dictionary để gom dữ liệu
data = defaultdict(lambda: {'ttff': [], 'fps': [], 'lsed': [], 'lsec': []})

# Hàm ép kiểu an toàn (Lọc bỏ N/A)
def safe_float(value):
    try:
        return float(value)
    except ValueError:
        return None # Nếu là 'N/A' hoặc lỗi thì trả về None

def get_stats(values):
    # Lọc bỏ các giá trị None (do N/A tạo ra)
    clean_values = [v for v in values if v is not None]
    if not clean_values:
        return "N/A"
    if len(clean_values) == 1:
        return f"{clean_values[0]:.2f} ± 0.00"
    
    mean = statistics.mean(clean_values)
    stdev = statistics.stdev(clean_values)
    return f"{mean:.2f} ± {stdev:.2f}"

try:
    # 1. ĐỌC DỮ LIỆU TTFF & FPS
    with open(ttff_file, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            method = row['Method'].lower()
            data[method]['ttff'].append(safe_float(row['TTFF (s)']))
            data[method]['fps'].append(safe_float(row['FPS']))

    # 2. ĐỌC DỮ LIỆU SYNCNET LSE
    with open(sync_file, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            video_name = row['Video File'].lower()
            method = ""
            if "sadtalker" in video_name: method = "sadtalker"
            elif "base" in video_name: method = "wav2lip_base"
            elif "v1" in video_name: method = "stream_v1"
            elif "v2" in video_name: method = "stream_v2"
            
            if method:
                data[method]['lsed'].append(safe_float(row['LSE-D (Distance)']))
                data[method]['lsec'].append(safe_float(row['LSE-C (Confidence)']))

    # Danh sách thứ tự các model muốn in ra
    methods_order = [
        ('SadTalker', 'sadtalker'),
        ('Wav2Lip Base', 'wav2lip_base'),
        ('Stream V1', 'stream_v1'),
        ('Stream V2', 'stream_v2')
    ]

    # 3. GHI RA FILE CSV MỚI
    with open(output_file, 'w', newline='', encoding='utf-8') as f_out:
        writer = csv.writer(f_out)
        # Ghi dòng tiêu đề
        writer.writerow(["Method", "TTFF (s) Mean ± SD", "FPS Mean ± SD", "LSE-D Mean ± SD", "LSE-C Mean ± SD"])
        
        # Ghi dữ liệu từng model
        for display_name, key in methods_order:
            metrics = data[key]
            ttff_str = get_stats(metrics['ttff'])
            fps_str = get_stats(metrics['fps'])
            lsed_str = get_stats(metrics['lsed'])
            lsec_str = get_stats(metrics['lsec'])
            
            writer.writerow([display_name, ttff_str, fps_str, lsed_str, lsec_str])

    print(f"\n✅ Đã tính toán xong! Dữ liệu tổng hợp (Mean ± SD) đã được lưu vào file: {output_file}")
    print("Mở file này bằng Excel để copy vào báo cáo hoặc vẽ biểu đồ nhé!\n")

except Exception as e:
    print(f"❌ Lỗi: {e}")