# ============================================================
# detection_service.py - محرك كشف التزييف
# ============================================================
# هذا الملف هو قلب النظام - فيه كل شي يخص الذكاء الاصطناعي:
#   1. تحميل الموديلات (ديبفيك + فلتر)
#   2. كشف الوجه بـ MTCNN وقصه من الصوره
#   3. تحليل الصوره وتصنيفها (حقيقيه / مزيفه)
#   4. كشف نوع الفلتر (7 انواع)
#   5. حفظ النتائج بقاعدة البيانات SQLite
#
# الموديلين مبنيين على EfficientNet-B4 (نوع من الشبكات العصبيه)
# استخدمنا Transfer Learning - يعني اخذنا موديل متدرب على ImageNet
# وعدلنا الطبقه الاخيره وعاد درّبناه على بياناتنا
#
# موديل الديبفيك: 99.36% دقه (حقيقي / مزيف)
# موديل الفلتر: 98.93% دقه (7 انواع فلاتر)
# ============================================================

import torch
import torch.nn as nn
from torchvision import transforms, models
import os
import sqlite3
from PIL import Image
import time
import numpy as np


# ============================================================
# قاعدة البيانات - SQLite
# ============================================================
# استخدمنا SQLite لانها ماتحتاج سيرفر منفصل زي MySQL
# الداتابيس كلها ملف واحد وسهل تنقله

DB_PATH = os.path.join("backend", "data", "detections.db")


def init_db():
    """ننشئ قاعدة البيانات والجداول اذا ماكانت موجوده"""
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()

    # جدول التحليلات - يحفظ كل نتيجه تحليل
    c.execute('''CREATE TABLE IF NOT EXISTS detections (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        filename TEXT NOT NULL,
        original_filename TEXT NOT NULL,
        result TEXT NOT NULL,
        confidence REAL NOT NULL,
        prob_real REAL DEFAULT 0,
        prob_deepfake REAL DEFAULT 0,
        prob_filtered REAL DEFAULT 0,
        risk_level TEXT DEFAULT 'UNKNOWN',
        media_type TEXT DEFAULT 'image',
        filter_result TEXT DEFAULT '',
        filter_confidence REAL DEFAULT 0,
        filter_type TEXT DEFAULT '',
        face_detected INTEGER DEFAULT 0,
        preprocessing TEXT DEFAULT '',
        video_frames_analyzed INTEGER DEFAULT 0,
        video_deepfake_ratio REAL DEFAULT 0,
        video_duration REAL DEFAULT 0,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )''')

    # جدول الاعدادات - يحفظ تفضيلات المستخدم
    c.execute('''CREATE TABLE IF NOT EXISTS settings (
        key TEXT PRIMARY KEY,
        value TEXT NOT NULL
    )''')

    # اعدادات افتراضيه
    defaults = [
        ('email_alerts', 'true'),
        ('auto_delete', 'false'),
        ('dark_mode', 'true'),
        ('gpu_acceleration', 'true'),
        ('save_history', 'true'),
        ('api_access', 'true'),
        ('username', 'Admin'),
        ('email', 'admin@deepfake.com'),
    ]
    for key, val in defaults:
        c.execute('INSERT OR IGNORE INTO settings (key, value) VALUES (?, ?)', (key, val))

    conn.commit()
    conn.close()
    print("  Database initialized")


# ننشئ القاعده اول مايشتغل الملف
init_db()


def get_db():
    """يفتح اتصال بقاعدة البيانات ويرجعه"""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row  # عشان نقدر نوصل للاعمده بالاسم
    return conn


# ============================================================
# تحميل الموديل
# ============================================================
# نبني EfficientNet-B4 بنفس الهيكل الي درّبنا عليه
# ونحمّل الاوزان المحفوظه من التدريب

def load_model(model_path, num_classes, device):
    """يحمّل موديل EfficientNet-B4 من ملف .pth"""
    model = models.efficientnet_b4(weights=None)

    # نغيّر الطبقه الاخيره عشان تطابق عدد الكلاسات حقتنا
    num_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.4),
        nn.Linear(num_features, 512),
        nn.ReLU(),
        nn.Dropout(p=0.3),
        nn.Linear(512, num_classes)
    )

    # نحمّل الاوزان من الملف
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()  # وضع التقييم مو التدريب
    return model, checkpoint


# ============================================================
# كاشف الوجوه - FaceExtractor
# ============================================================
# هذا الكلاس يكشف الوجه بالصوره ويقصه
# مهم جدا لان الصور الي تدرب عليها الموديل كلها وجوه مقصوصه
# فلازم الصور الجديده تمر بنفس المعالجه
#
# يستخدم MTCNN (الافضل) واذا ماكان متوفر يستخدم Haar Cascade
# واذا ماكشف وجه يستخدم الصوره كامله

class FaceExtractor:
    def __init__(self):
        self.face_cascade = None
        self.mtcnn = None
        self.method = None

        # نجرب MTCNN اول - هو الافضل والادق
        try:
            from facenet_pytorch import MTCNN
            self.mtcnn = MTCNN(
                keep_all=True,
                device='cuda' if torch.cuda.is_available() else 'cpu',
                min_face_size=40,
                thresholds=[0.6, 0.7, 0.7]
            )
            self.method = "MTCNN"
            print("   Face detector: MTCNN (ready)")
            return
        except ImportError:
            pass

        # اذا MTCNN مو موجود نجرب OpenCV Haar Cascade
        try:
            import cv2
            cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
            self.face_cascade = cv2.CascadeClassifier(cascade_path)
            if not self.face_cascade.empty():
                self.method = "Haar"
                print("   Face detector: Haar Cascade (ready)")
                return
        except:
            pass

        # مالقينا شي - نستخدم الصوره كامله
        self.method = None
        print("   Face detector: None (using full image)")

    def extract_face(self, image, padding=0.3):
        """
        يكشف الوجه بالصوره ويقصه مع padding حوله
        يرجع: (الصوره المقصوصه، هل لقى وجه، احداثيات الوجه)
        """
        if self.method is None:
            return image, False, None
        try:
            if self.method == "MTCNN":
                return self._extract_mtcnn(image, padding)
            elif self.method == "Haar":
                return self._extract_haar(image, padding)
        except:
            pass
        return image, False, None

    def _extract_mtcnn(self, image, padding):
        """يكشف الوجوه بـ MTCNN ويختار الاكبر"""
        boxes, probs = self.mtcnn.detect(image)
        if boxes is None or len(boxes) == 0:
            return image, False, None

        # نختار اكبر وجه (بالمساحه)
        areas = [(b[2] - b[0]) * (b[3] - b[1]) for b in boxes]
        best_idx = np.argmax(areas)
        box = boxes[best_idx]

        cropped = self._crop_with_padding(image, box, padding)
        box_info = [int(b) for b in box]
        return cropped, True, box_info

    def _extract_haar(self, image, padding):
        """يكشف الوجوه بـ Haar Cascade (الخطه البديله)"""
        import cv2
        img_array = np.array(image)
        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        faces = self.face_cascade.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=5, minSize=(40, 40)
        )
        if len(faces) == 0:
            return image, False, None

        # نختار اكبر وجه
        areas = [w * h for (x, y, w, h) in faces]
        best_idx = np.argmax(areas)
        x, y, w, h = faces[best_idx]
        box = [x, y, x + w, y + h]

        cropped = self._crop_with_padding(image, box, padding)
        return cropped, True, box

    def _crop_with_padding(self, image, box, padding):
        """
        يقص الوجه من الصوره مع padding حوله
        الـ padding مهم عشان يشمل الشعر والاذن والذقن
        """
        w_img, h_img = image.size
        x1, y1, x2, y2 = [int(b) for b in box]

        face_w = x2 - x1
        face_h = y2 - y1
        pad_w = int(face_w * padding)
        pad_h = int(face_h * padding)

        # نتاكد مانطلع برا حدود الصوره
        x1 = max(0, x1 - pad_w)
        y1 = max(0, y1 - pad_h)
        x2 = min(w_img, x2 + pad_w)
        y2 = min(h_img, y2 + pad_h)

        return image.crop((x1, y1, x2, y2))


# ============================================================
# محرك الكشف الرئيسي - DeepfakeDetector
# ============================================================
# هذا الكلاس يجمع كل شي مع بعض:
# - يحمّل الموديلين (ديبفيك + فلتر)
# - يجهز كاشف الوجوه
# - يجهز محلل الفيديو
# - يوفر دوال لتحليل الصور والفيديو

class DeepfakeDetector:
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        print("=" * 70)
        print(" Initializing Deepfake Detection System")
        print("=" * 70)
        print(f"  Device: {self.device}")

        if torch.cuda.is_available():
            print(f"  GPU: {torch.cuda.get_device_name(0)}")
            try:
                vram = torch.cuda.get_device_properties(0).total_memory / 1024**3
                print(f"  VRAM: {vram:.1f} GB")
            except:
                pass

        # معالجة الصور - نفس المعالجه الي استخدمناها بالتدريب
        # Resize -> CenterCrop -> ToTensor -> Normalize
        self.transform = transforms.Compose([
            transforms.Resize((380, 380)),
            transforms.CenterCrop(299),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

        # ---- تحميل موديل الديبفيك (الموديل الاول) ----
        print("\n  [1/2] Loading Deepfake Model...")
        model_path = None
        for name in ["deepfake_2class_v2.pth", "deepfake_2class_v1.pth"]:
            path = os.path.join("backend", "models", "saved_models", name)
            if os.path.exists(path):
                model_path = path
                break

        # اذا مالقينا بالاسم ندور اي ملف فيه 2class
        if not model_path:
            search_dir = os.path.join("backend", "models", "saved_models")
            if os.path.exists(search_dir):
                for f in os.listdir(search_dir):
                    if "2class" in f and f.endswith(".pth"):
                        model_path = os.path.join(search_dir, f)
                        break

        if not model_path:
            raise FileNotFoundError("Deepfake model not found!")

        self.deepfake_model, cp = load_model(model_path, 2, self.device)
        self.deepfake_classes = cp.get('class_names', ['deepfake', 'real'])
        self.deepfake_acc = cp.get('val_acc', 99.36)
        print(f"   Loaded: {os.path.basename(model_path)}")
        print(f"   Accuracy: {self.deepfake_acc:.2f}%")
        print(f"   Classes: {self.deepfake_classes}")

        # ---- تحميل موديل الفلتر (الموديل الثاني) ----
        print("\n  [2/2] Loading Filter Model...")
        filter_path = os.path.join("backend", "models", "saved_models", "filter_detector_v1.pth")
        self.filter_model = None
        self.filter_classes = None
        self.filter_acc = 0

        if os.path.exists(filter_path):
            cp2 = torch.load(filter_path, map_location=self.device, weights_only=False)
            num_filter_classes = cp2.get('num_classes', len(cp2.get('class_names', [])))

            # نبني الموديل بنفس الهيكل الي تدرب عليه
            filter_model = models.efficientnet_b4(weights=None)
            num_features = filter_model.classifier[1].in_features
            filter_model.classifier = nn.Sequential(
                nn.Dropout(p=0.4),
                nn.Linear(num_features, 512),
                nn.ReLU(),
                nn.Dropout(p=0.3),
                nn.Linear(512, num_filter_classes)
            )
            filter_model.load_state_dict(cp2['model_state_dict'])
            filter_model = filter_model.to(self.device)
            filter_model.eval()

            self.filter_model = filter_model
            self.filter_classes = cp2.get('class_names',
                ['clean', 'age', 'animal', 'artistic', 'color', 'distortion', 'mixed'])
            self.filter_acc = cp2.get('val_acc', 98.93)
            print(f"   Loaded: filter_detector_v1.pth")
            print(f"   Accuracy: {self.filter_acc:.2f}%")
            print(f"   Classes: {self.filter_classes}")
        else:
            print("   Filter model not found - disabled")

        # ---- كاشف الوجوه ----
        print("\n  Initializing Face Detector...")
        self.face_extractor = FaceExtractor()

        # ---- محلل الفيديو ----
        print("\n  Initializing Video Analyzer...")
        try:
            try:
                from services.video_analyzer_v2 import VideoAnalyzerV2
            except ImportError:
                from video_analyzer_v2 import VideoAnalyzerV2
            self.video_analyzer = VideoAnalyzerV2(self)
            self.video_supported = True
        except ImportError as e:
            print(f"   Video analyzer not available: {e}")
            self.video_supported = False
        except Exception as e:
            print(f"   Video init error: {e}")
            self.video_supported = False

        # ملخص الحاله
        print("\n" + "=" * 70)
        print(" Detection System Ready!")
        print(f"   Deepfake: {self.deepfake_acc:.2f}%")
        filt_status = f"{self.filter_acc:.2f}%" if self.filter_model else "Disabled"
        print(f"   Filter: {filt_status}")
        print(f"   Face Detector: {self.face_extractor.method or 'None'}")
        print(f"   Video: {'Yes' if self.video_supported else 'No'}")
        print("=" * 70)

    def _predict_deepfake(self, image):
        """
        يمرر الصوره على موديل الديبفيك ويرجع النتيجه
        يرجع: (الكلاس، نسبة الثقه، احتمالات كل كلاس)
        """
        image_tensor = self.transform(image).unsqueeze(0).to(self.device)
        with torch.no_grad():
            outputs = self.deepfake_model(image_tensor)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            confidence_score, predicted = torch.max(probabilities, 1)

        predicted_class = self.deepfake_classes[predicted.item()]
        confidence = round(confidence_score.item() * 100, 2)

        # نجمع احتمالات كل كلاس بقاموس
        probs_dict = {}
        for idx, class_name in enumerate(self.deepfake_classes):
            probs_dict[class_name] = round(probabilities[0][idx].item() * 100, 2)

        return predicted_class, confidence, probs_dict

    def _predict_filter(self, image):
        """
        يمرر الصوره على موديل الفلتر ويرجع النتيجه
        يرجع: (نوع الفلتر، نسبة الثقه، احتمالات كل نوع)
        """
        if self.filter_model is None:
            return None, 0, {}

        image_tensor = self.transform(image).unsqueeze(0).to(self.device)
        with torch.no_grad():
            outputs = self.filter_model(image_tensor)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            confidence_score, predicted = torch.max(probabilities, 1)

        predicted_class = self.filter_classes[predicted.item()]
        confidence = round(confidence_score.item() * 100, 2)

        probs_dict = {}
        for idx, class_name in enumerate(self.filter_classes):
            probs_dict[class_name] = round(probabilities[0][idx].item() * 100, 2)

        return predicted_class, confidence, probs_dict

    def analyze_frame(self, image):
        """يحلل فريم واحد من الفيديو (يكشف الوجه ويحلله)"""
        face_img, face_found, _ = self.face_extractor.extract_face(image)
        return self._predict_deepfake(face_img)

    def analyze_image(self, image_path, original_filename="unknown"):
        """
        التحليل الكامل لصوره:
        1. يفتح الصوره
        2. يكشف الوجه ويقصه
        3. يحلل بموديل الديبفيك
        4. يحلل بموديل الفلتر
        5. يحفظ النتيجه بقاعدة البيانات
        6. يرجع كل التفاصيل
        """
        try:
            start_time = time.time()

            image = Image.open(image_path).convert('RGB')
            original_size = image.size

            # --- الخطوه 1: كشف الوجه وقصه ---
            face_img, face_found, face_box = self.face_extractor.extract_face(image)
            face_size = face_img.size if face_found else None

            # نجمع معلومات المعالجه عشان نعرضها للمستخدم
            preprocessing_info = {
                "original_size": f"{original_size[0]}x{original_size[1]}",
                "face_detected": face_found,
                "face_detector": self.face_extractor.method or "none",
                "face_box": face_box,
                "face_crop_size": f"{face_size[0]}x{face_size[1]}" if face_size else None,
                "model_input_size": "299x299",
                "steps": []
            }

            if face_found:
                preprocessing_info["steps"].append(
                    f"Face detected using {self.face_extractor.method}")
                preprocessing_info["steps"].append(
                    f"Cropped face region: {face_size[0]}x{face_size[1]}px")
            else:
                preprocessing_info["steps"].append("No face detected - using full image")

            preprocessing_info["steps"].append("Resized to 380x380px")
            preprocessing_info["steps"].append("Center cropped to 299x299px")
            preprocessing_info["steps"].append("Normalized (ImageNet standards)")

            if face_found:
                print(f"   Face detected and cropped ({face_size[0]}x{face_size[1]})")
            else:
                print(f"   No face found - using full image")

            # --- الخطوه 2: تحليل الديبفيك ---
            predicted_class, confidence, probs_dict = self._predict_deepfake(face_img)

            # --- الخطوه 3: تحليل الفلتر ---
            filter_class, filter_conf, filter_probs = self._predict_filter(face_img)
            is_filtered = filter_class is not None and filter_class != 'clean'

            # نحدد مستوى الخطر
            if predicted_class == "deepfake":
                risk_level = "CRITICAL"
            elif is_filtered:
                risk_level = "WARNING"
            else:
                risk_level = "SAFE"

            descriptions = {
                'real': 'Natural unmanipulated image - no deepfake detected',
                'deepfake': 'Deepfake detected - face swap, AI-generated, or manipulated content'
            }

            # --- الخطوه 4: نحفظ بقاعدة البيانات ---
            stored_filename = os.path.basename(image_path)
            conn = get_db()
            c = conn.cursor()
            c.execute('''INSERT INTO detections
                (filename, original_filename, result, confidence, prob_real, prob_deepfake,
                 prob_filtered, risk_level, media_type, filter_result, filter_confidence,
                 filter_type, face_detected, preprocessing)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)''',
                (stored_filename, original_filename, predicted_class, confidence,
                 probs_dict.get('real', 0), probs_dict.get('deepfake', 0),
                 filter_conf if is_filtered else 0,
                 risk_level, 'image',
                 filter_class or '', filter_conf,
                 filter_class if is_filtered else '',
                 1 if face_found else 0,
                 self.face_extractor.method or 'none'))
            conn.commit()
            detection_id = c.lastrowid
            conn.close()

            analysis_time = time.time() - start_time

            # --- نجهز الرد بكل التفاصيل ---
            details = {
                "id": detection_id,
                "model": "EfficientNet-B4 V2",
                "architecture": "EfficientNet-B4",
                "device": str(self.device),
                "model_accuracy": f"{self.deepfake_acc:.2f}%",
                "predicted_class": predicted_class,
                "probabilities": probs_dict,
                "risk_level": risk_level,
                "description": descriptions.get(predicted_class, ''),
                "analysis_time": round(analysis_time, 2),
                "preprocessing": preprocessing_info,
                "filter": {
                    "enabled": self.filter_model is not None,
                    "result": filter_class,
                    "confidence": filter_conf,
                    "is_filtered": is_filtered,
                    "probabilities": filter_probs,
                    "model_accuracy": f"{self.filter_acc:.2f}%" if self.filter_model else None
                }
            }

            # نطبع النتيجه بالكونسول
            filter_str = f" | Filter: {filter_class}({filter_conf}%)" if filter_class else ""
            status = "FAKE" if predicted_class == "deepfake" else "REAL"
            print(f"  Result: {status} | {confidence}%{filter_str} | #{detection_id} | {analysis_time:.1f}s")

            return predicted_class == 'deepfake', confidence, details

        except Exception as e:
            print(f"  Error: {e}")
            import traceback
            traceback.print_exc()
            return False, 0.0, {"error": str(e)}

    def analyze_video(self, video_path, original_filename="unknown"):
        """
        تحليل فيديو:
        يرسل الفيديو لمحلل الفيديو الي يستخرج 20 فريم
        ويحلل كل واحد، واذا 40% او اكثر مزيفه يعتبر الفيديو مزيف
        """
        if not self.video_supported:
            return False, 0.0, {
                "error": "Video analysis not available",
                "predicted_class": "unknown"
            }

        try:
            result = self.video_analyzer.analyze_video(video_path)

            if not result.get("success"):
                return False, 0.0, result

            # نحفظ النتيجه بقاعدة البيانات
            stored_filename = os.path.basename(video_path)
            conn = get_db()
            c = conn.cursor()
            c.execute('''INSERT INTO detections
                (filename, original_filename, result, confidence, prob_real, prob_deepfake,
                 prob_filtered, risk_level, media_type, video_frames_analyzed,
                 video_deepfake_ratio, video_duration)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)''',
                (stored_filename, original_filename,
                 result["predicted_class"], result["confidence"],
                 result["probabilities"].get("real", 0),
                 result["probabilities"].get("deepfake", 0),
                 0, result["risk_level"], 'video',
                 result["analysis_details"]["total_frames_analyzed"],
                 result["analysis_details"]["deepfake_ratio"],
                 result.get("video_metadata", {}).get("duration_seconds", 0)))
            conn.commit()
            detection_id = c.lastrowid
            conn.close()

            result["id"] = detection_id

            status = "FAKE" if result["predicted_class"] == "deepfake" else "REAL"
            print(f"  Video: {status} | {result['confidence']}% | #{detection_id}")

            return result["predicted_class"] == "deepfake", result["confidence"], result

        except Exception as e:
            print(f"  Video error: {e}")
            import traceback
            traceback.print_exc()
            return False, 0.0, {"error": str(e), "predicted_class": "unknown"}


# ============================================================
# تشغيل محرك الكشف
# ============================================================
# اول مانستورد هذا الملف ينشئ الموديل ويحمّل كل شي
# اذا صار خطأ يخلي detector = None والسيرفر يقدر يشتغل بدونه

print("\n" + "=" * 70)
print(" Starting Detection Service...")
print("=" * 70)

try:
    detector = DeepfakeDetector()
    print("  Service Ready!\n")
except Exception as e:
    print(f"  Failed to start: {e}")
    import traceback
    traceback.print_exc()
    detector = None