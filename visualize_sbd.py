"""Visualize SBD segments: extract thumbnails + cut video clips + generate HTML gallery."""

import os
import json
import cv2
import subprocess
import argparse
from pathlib import Path


def extract_thumbnail(video_path, frame_idx, output_path):
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ret, frame = cap.read()
    cap.release()
    if ret:
        cv2.imwrite(output_path, frame)
        return True
    return False


def cut_clip(video_path, start_frame, end_frame, fps, output_path):
    start_time = start_frame / fps
    duration = (end_frame - start_frame) / fps
    cmd = [
        "ffmpeg", "-y", "-ss", f"{start_time:.3f}",
        "-i", video_path, "-t", f"{duration:.3f}",
        "-c:v", "libx264", "-preset", "fast", "-crf", "23",
        "-an", output_path
    ]
    subprocess.run(cmd, capture_output=True)


def get_fps(video_path):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    cap.release()
    return fps


def generate_html(segments, out_dir, video_name):
    html = f"""<!DOCTYPE html>
<html><head><meta charset="utf-8">
<title>SBD Segments - {video_name}</title>
<style>
body {{ font-family: -apple-system, sans-serif; max-width: 1400px; margin: 0 auto; padding: 20px; background: #1a1a2e; color: #eee; }}
h1 {{ color: #e94560; }}
.segment {{ background: #16213e; border-radius: 12px; padding: 16px; margin: 16px 0; display: flex; gap: 16px; align-items: flex-start; }}
.segment:hover {{ background: #1a2744; }}
.seg-info {{ min-width: 140px; }}
.seg-info h3 {{ color: #e94560; margin: 0 0 8px; }}
.seg-info p {{ margin: 4px 0; font-size: 13px; color: #aaa; }}
.thumbs {{ display: flex; gap: 8px; flex-wrap: wrap; }}
.thumbs img {{ height: 120px; border-radius: 6px; border: 2px solid #333; }}
.thumbs img:hover {{ border-color: #e94560; }}
video {{ height: 200px; border-radius: 6px; }}
.stats {{ background: #0f3460; padding: 12px 20px; border-radius: 8px; margin-bottom: 20px; display: flex; gap: 30px; }}
.stats span {{ font-size: 14px; }}
.stats b {{ color: #e94560; }}
</style></head><body>
<h1>SBD Skill Segments - {video_name}</h1>
<div class="stats">
  <span>Total segments: <b>{len(segments)}</b></span>
  <span>Total frames: <b>{segments[-1]['end_frame']}</b></span>
  <span>Avg segment length: <b>{sum(s['end_frame']-s['start_frame'] for s in segments)//len(segments)} frames</b></span>
</div>
"""
    for i, seg in enumerate(segments):
        sf, ef = seg["start_frame"], seg["end_frame"]
        length = ef - sf
        clip_file = f"clip_{i:02d}.mp4"
        html += f"""<div class="segment">
  <div class="seg-info">
    <h3>Segment {i}</h3>
    <p>Frames: {sf} - {ef}</p>
    <p>Length: {length} frames</p>
    <p>~{length/30:.1f}s</p>
  </div>
  <div class="thumbs">
    <img src="seg_{i:02d}_first.jpg" title="First frame ({sf})">
    <img src="seg_{i:02d}_mid.jpg" title="Mid frame ({(sf+ef)//2})">
    <img src="seg_{i:02d}_last.jpg" title="Last frame ({ef})">
  </div>
  <video src="{clip_file}" controls preload="metadata"></video>
</div>
"""
    html += "</body></html>"
    with open(os.path.join(out_dir, "gallery.html"), "w") as f:
        f.write(html)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sbd-json", default="eval_gallery/sbd_results/sbd_segments.json")
    parser.add_argument("--output-dir", default="eval_gallery/sbd_visual")
    args = parser.parse_args()

    with open(args.sbd_json) as f:
        segments = json.load(f)

    out_dir = args.output_dir
    os.makedirs(out_dir, exist_ok=True)

    videos = {}
    for seg in segments:
        vp = seg["video_path"]
        if vp not in videos:
            videos[vp] = get_fps(vp)

    print(f"Processing {len(segments)} segments from {len(videos)} video(s)...")

    for i, seg in enumerate(segments):
        vp = seg["video_path"]
        sf, ef = seg["start_frame"], seg["end_frame"]
        mid = (sf + ef) // 2
        fps = videos[vp]

        print(f"  [{i+1}/{len(segments)}] Segment {i}: frames {sf}-{ef} (len={ef-sf})")

        extract_thumbnail(vp, sf, os.path.join(out_dir, f"seg_{i:02d}_first.jpg"))
        extract_thumbnail(vp, mid, os.path.join(out_dir, f"seg_{i:02d}_mid.jpg"))
        extract_thumbnail(vp, max(ef - 1, sf), os.path.join(out_dir, f"seg_{i:02d}_last.jpg"))

        clip_path = os.path.join(out_dir, f"clip_{i:02d}.mp4")
        cut_clip(vp, sf, ef, fps, clip_path)

    video_name = Path(segments[0]["video_path"]).stem
    generate_html(segments, out_dir, video_name)

    print(f"\nDone! Open: {out_dir}/gallery.html")


if __name__ == "__main__":
    main()
