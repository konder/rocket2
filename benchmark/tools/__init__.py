"""
Benchmark helper tools (not used by benchmark_eval.py).

- capture_first_frames: capture first frame per task for detector comparison.
- compare_detectors: run multiple detection backends on first frames, output detection_results.json.
- judge_detections: VLM judge on detection_results.json to pick best candidate per task.

Official evaluation uses GroundingDino via benchmark_eval.py; these tools support
comparison and ablation of detection backends (Molmo+SAM2, DINO+SAM2, Sa2VA, Qwen-VL, etc.).
"""
