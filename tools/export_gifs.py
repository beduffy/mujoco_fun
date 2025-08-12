import os
from glob import glob
import imageio

OUTPUT_DIR = "outputs"
GIF_DIR = os.path.join(OUTPUT_DIR, "gifs")
PNG_DIR = os.path.join(OUTPUT_DIR, "screens")

os.makedirs(GIF_DIR, exist_ok=True)
os.makedirs(PNG_DIR, exist_ok=True)

mp4_paths = sorted(glob(os.path.join(OUTPUT_DIR, "*.mp4")))

for mp4_path in mp4_paths:
    base = os.path.splitext(os.path.basename(mp4_path))[0]
    gif_path = os.path.join(GIF_DIR, f"{base}.gif")
    png_path = os.path.join(PNG_DIR, f"{base}.png")

    reader = imageio.get_reader(mp4_path)
    meta = reader.get_meta_data()
    fps = max(1, int(meta.get("fps", 15)))

    # Save first frame as PNG screenshot
    try:
        first_frame = reader.get_data(0)
        imageio.imwrite(png_path, first_frame)
    except Exception:
        pass

    # Write GIF with limited frames for size
    writer = imageio.get_writer(gif_path, fps=min(fps, 20))
    try:
        raw_nframes = meta.get("nframes")
        try:
            num_frames = int(raw_nframes) if raw_nframes is not None else 0
        except Exception:
            num_frames = 0
        if num_frames <= 0:
            # Fallback: iterate until StopIteration
            idx = 0
            for frame in reader:
                if idx % 2 == 0:
                    writer.append_data(frame)
                idx += 1
        else:
            step = max(1, num_frames // 150)
            for i in range(0, int(num_frames), int(step)):
                frame = reader.get_data(i)
                writer.append_data(frame)
    finally:
        writer.close()
        reader.close()