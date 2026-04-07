import os
import cv2

os.environ["CUDA_VISIBLE_DEVICES"] = "6"
os.environ["TRANSFORMERS_VERBOSITY"] = "error"

BASE_DIR = os.path.abspath(".")
ASSETS_DIR = os.path.join(BASE_DIR, "assets")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")
DATASET_DIR = os.path.join(ASSETS_DIR, "dataset")
os.makedirs(OUTPUT_DIR, exist_ok=True)

def get_optimized_face():
    face_path = os.path.join(ASSETS_DIR, "face_default.jpg")
    opt_path = os.path.join(ASSETS_DIR, "face_optimized.jpg")
    if os.path.exists(face_path):
        img = cv2.imread(face_path)
        h, w = img.shape[:2]
        if max(h, w) > 480:
            scale = 480 / max(h, w)
            img = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_LANCZOS4)
        cv2.imwrite(opt_path, img) # Luôn luôn ghi đè ảnh mới chuẩn 480px
    return opt_path

OPTIMIZED_FACE = get_optimized_face()

WAV2LIP_ARGS = {
    "checkpoint_path": "checkpoints/wav2lip_gan.pth",
    "static": False, "fps": 25.0, "pads": [0, 10, 0, 0],
    "face_det_batch_size": 1, "wav2lip_batch_size": 64,
    "resize_factor": 2, "crop": [0, -1, 0, -1],
    "box": [-1, -1, -1, -1], "rotate": False,
    "nosmooth": True, "img_size": 96
}