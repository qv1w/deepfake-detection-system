# ============================================================
# video_analyzer_v2.py - محلل الفيديو
# ============================================================
# هذا الملف يحلل الفيديوهات عن طريق استخراج فريمات (صور) منها
# وتحليل كل فريم لحاله بموديل الديبفيك
#
# الطريقه:
#   1. يفتح الفيديو بـ OpenCV
#   2. يستخرج 20 فريم موزعين بالتساوي (يتجاهل اول واخر 5%)
#   3. كل فريم يمر بنفس خطوات تحليل الصوره (كشف وجه + موديل)
#   4. اذا 40% او اكثر من الفريمات طلعت مزيفه = الفيديو مزيف
#
# الفورمات المدعومه: MP4, AVI, MOV, MKV, WebM
# الحد الاقصى: 100 ميقا
# ============================================================

import cv2
import torch
from PIL import Image
import numpy as np
import os
import time


class VideoAnalyzerV2:
    def __init__(self, detector):
        """
        نجهز محلل الفيديو
        detector: كائن DeepfakeDetector الي فيه الموديل وكاشف الوجوه
        """
        self.detector = detector
        self.max_frames = 20          # اقصى عدد فريمات نحللها
        self.min_frames = 5           # اقل عدد فريمات مقبول
        self.deepfake_threshold = 0.4  # اذا 40% مزيفه = الفيديو مزيف

        print("   Video Analyzer initialized")
        print(f"      Max frames: {self.max_frames} | Threshold: {self.deepfake_threshold*100:.0f}%")

    def extract_frames(self, video_path, num_frames=None):
        """
        يستخرج فريمات من الفيديو موزعه بالتساوي
        يتجاهل اول واخر 5% عشان عادة تكون مقدمه وخاتمه مالها دخل
        يرجع: (قائمة الفريمات كصور PIL، معلومات الفيديو)
        """
        if num_frames is None:
            num_frames = self.max_frames

        # نفتح الفيديو بـ OpenCV
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")

        # ناخذ معلومات الفيديو
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        duration = total_frames / fps if fps > 0 else 0

        metadata = {
            "total_frames": total_frames,
            "fps": round(fps, 2),
            "width": width,
            "height": height,
            "duration_seconds": round(duration, 2),
            "duration": round(duration, 2),
            "resolution": f"{width}x{height}"
        }

        minutes = int(duration // 60)
        seconds = int(duration % 60)
        print(f"\n   Video: {width}x{height} | {fps:.1f}fps | {minutes}:{seconds:02d}")

        # نتاكد ان الفيديو فيه فريمات كافيه
        if total_frames < self.min_frames:
            raise ValueError(f"Video too short: {total_frames} frames")

        actual_frames = min(num_frames, total_frames)

        # نحدد الفريمات الي نبي نستخرجها
        if total_frames <= num_frames:
            # الفيديو قصير - ناخذ كل الفريمات
            frame_indices = list(range(total_frames))
        else:
            # نوزعها بالتساوي مع تجاهل اول واخر 5%
            start = int(total_frames * 0.05)
            end = int(total_frames * 0.95)
            frame_indices = np.linspace(start, end, actual_frames, dtype=int).tolist()

        # نستخرج الفريمات
        frames = []
        frame_positions = []

        for idx in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if ret:
                # OpenCV يرجع BGR فنحوله لـ RGB عشان PIL
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                pil_image = Image.fromarray(frame_rgb)
                frames.append(pil_image)
                # نحسب وقت الفريم بالثواني
                frame_positions.append(round(idx / fps, 1) if fps > 0 else 0)

        cap.release()
        print(f"   Extracted {len(frames)} frames")

        metadata["extracted_frames"] = len(frames)
        metadata["frame_positions"] = frame_positions

        return frames, metadata

    def analyze_video(self, video_path):
        """
        التحليل الكامل للفيديو:
        يستخرج الفريمات ويحلل كل واحد ويجمع النتائج
        ويقرر اذا الفيديو حقيقي او مزيف بناء على النسبه
        """
        start_time = time.time()

        print(f"\n{'='*50}")
        print(f" VIDEO ANALYSIS")
        print(f"{'='*50}")
        print(f"   File: {os.path.basename(video_path)}")

        # نستخرج الفريمات
        frames, metadata = self.extract_frames(video_path)

        if len(frames) < self.min_frames:
            return {"success": False, "error": f"Not enough frames: {len(frames)}"}

        print(f"\n   Analyzing {len(frames)} frames...")

        # نحلل كل فريم
        frame_results = []
        deepfake_count = 0
        real_count = 0
        total_deepfake_prob = 0
        total_real_prob = 0

        for i, frame in enumerate(frames):
            # كل فريم يمر بكشف الوجه والموديل
            predicted_class, confidence, probs = self.detector.analyze_frame(frame)

            frame_result = {
                "frame_index": i + 1,
                "timestamp": metadata["frame_positions"][i] if i < len(metadata["frame_positions"]) else 0,
                "predicted_class": predicted_class,
                "confidence": confidence,
                "probabilities": probs
            }
            frame_results.append(frame_result)

            if predicted_class == "deepfake":
                deepfake_count += 1
            else:
                real_count += 1

            total_deepfake_prob += probs.get("deepfake", 0)
            total_real_prob += probs.get("real", 0)

            # نطبع كل 5 فريمات عشان نعرف وين وصلنا
            if (i + 1) % 5 == 0 or i == 0 or i == len(frames) - 1:
                status = "FAKE" if predicted_class == "deepfake" else "REAL"
                print(f"      Frame {i+1}/{len(frames)}: {status} ({confidence}%)")

        # --- نجمع النتائج ---
        total_analyzed = len(frame_results)
        deepfake_ratio = deepfake_count / total_analyzed
        avg_deepfake_prob = total_deepfake_prob / total_analyzed
        avg_real_prob = total_real_prob / total_analyzed

        # القرار النهائي: اذا 40% او اكثر مزيفه = مزيف
        if deepfake_ratio >= self.deepfake_threshold:
            final_class = "deepfake"
            final_confidence = round(avg_deepfake_prob, 2)
            risk_level = "CRITICAL"
            description = (f"Deepfake video - {deepfake_count}/{total_analyzed} "
                         f"frames flagged ({deepfake_ratio*100:.0f}%)")
        else:
            final_class = "real"
            final_confidence = round(avg_real_prob, 2)
            risk_level = "SAFE"
            description = (f"Authentic video - {real_count}/{total_analyzed} "
                         f"frames verified ({(1-deepfake_ratio)*100:.0f}%)")

        analysis_time = time.time() - start_time

        # نجهز النتيجه النهائيه
        result = {
            "success": True,
            "media_type": "video",
            "predicted_class": final_class,
            "confidence": final_confidence,
            "risk_level": risk_level,
            "description": description,
            "probabilities": {
                "real": round(avg_real_prob, 2),
                "deepfake": round(avg_deepfake_prob, 2)
            },
            "video_metadata": metadata,
            "analysis_details": {
                "total_frames_analyzed": total_analyzed,
                "deepfake_frames": deepfake_count,
                "real_frames": real_count,
                "deepfake_ratio": round(deepfake_ratio * 100, 2),
                "threshold_used": self.deepfake_threshold * 100,
                "analysis_time_seconds": round(analysis_time, 2)
            },
            "frame_results": frame_results
        }

        # نطبع الملخص
        verdict = "FAKE" if final_class == "deepfake" else "REAL"
        print(f"\n{'='*50}")
        print(f"   Verdict: {verdict} ({final_confidence}%)")
        print(f"   Frames: {deepfake_count} deepfake / {real_count} real")
        print(f"   Time: {analysis_time:.1f}s")
        print(f"{'='*50}\n")

        return result