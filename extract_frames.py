import cv2
import os

# ==========================================
# 1. CẤU HÌNH ĐƯỜNG DẪN
# ==========================================
video_files = {
    "SadTalker": "outputs/record7_sadtalker.mp4",
    "Wav2Lip_Base": "outputs/record7_base.mp4",
    "Stream_V1": "outputs/record7_v1_final.mp4",
    "Stream_V2": "outputs/record7_v2_final.mp4"
}

# Thư mục gốc chứa các ảnh
BASE_OUTPUT_DIR = "figures"

# ==========================================
# 2. XỬ LÝ TRÍCH XUẤT TOÀN BỘ FRAME
# ==========================================
print("🎬 Bắt đầu trích xuất TOÀN BỘ frame từ các video...\n")

for model_name, video_path in video_files.items():
    if not os.path.exists(video_path):
        print(f"❌ Không tìm thấy video: {video_path}")
        continue

    # Tạo folder riêng cho từng model (vd: figures/SadTalker)
    model_output_dir = os.path.join(BASE_OUTPUT_DIR, model_name)
    os.makedirs(model_output_dir, exist_ok=True)
    
    # Mở video để đọc
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"⏳ Đang xử lý [{model_name}] - Tổng dự kiến: {total_frames} frames...")
    
    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break  # Thoát vòng lặp khi video kết thúc
            
        # Đặt tên file có số thứ tự dạng 0000, 0001, 0002... để máy tính sort cho chuẩn
        filename = f"frame_{frame_count:04d}.png"
        out_path = os.path.join(model_output_dir, filename)
        
        # Lưu TRỰC TIẾP toàn bộ frame gốc, nén 0 để ảnh nguyên bản nhất
        cv2.imwrite(out_path, frame, [cv2.IMWRITE_PNG_COMPRESSION, 0])
        frame_count += 1
        
        # In tiến độ mỗi 50 frames để bạn biết script vẫn đang chạy bình thường
        if frame_count % 50 == 0:
            print(f"   -> Đã lưu {frame_count}/{total_frames} frames...")

    cap.release()
    print(f"✅ Hoàn tất! Đã lưu {frame_count} frames của [{model_name}] vào thư mục: {model_output_dir}\n")

print("🎉 Hoàn tất trích xuất toàn bộ video! Hãy kiểm tra thư mục figures.")