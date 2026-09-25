# ============================================================
# app.py - السيرفر الرئيسي
# ============================================================
# هذا ملف السيرفر حق نظام كشف التزييف العميق
# يستقبل الصور والفيديوهات من المستخدم ويرسلها للموديل
# يحللها ويرجع النتيجه بصيغة JSON للفرونت اند
#
# استخدمنا Flask لانه خفيف وبسيط ومايحتاج اشياء زياده
# زي Django الي فيه اشياء كثيره ماتحتاجها
#
# الروابط:
#   POST /api/detect         - رفع وتحليل صوره او فيديو
#   GET  /api/stats          - احصائيات ومعلومات الموديل
#   GET  /api/history        - سجل التحليلات السابقه
#   DELETE /api/history/<id> - حذف تحليل معين
#   GET  /api/settings       - جلب الاعدادات
#   POST /api/settings       - تعديل الاعدادات
#   GET  /api/info           - حالة النظام
# ============================================================

from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import os
import torch
from datetime import datetime, timedelta

# نجيب محرك الكشف وقاعدة البيانات
from services.detection_service import detector, get_db

# نجهز Flask
app = Flask(__name__)
CORS(app)  # عشان الفرونت اند يقدر يتواصل مع السيرفر

# مكان حفظ الملفات المرفوعه
UPLOAD_FOLDER = os.path.join('backend', 'data', 'uploads')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# انواع الملفات المدعومه
ALLOWED_IMAGE_EXT = {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp', '.tiff'}
ALLOWED_VIDEO_EXT = {'.mp4', '.avi', '.mov', '.mkv', '.wmv', '.flv', '.webm', '.m4v', '.3gp'}
MAX_VIDEO_SIZE = 100 * 1024 * 1024  # اقصى حجم للفيديو 100 ميقا


# ============================================================
# الصفحه الرئيسيه
# ============================================================
@app.route('/')
def home():
    """يرجع صفحة index.html من مجلد الفرونت اند"""
    return send_from_directory('../frontend', 'index.html')


# ============================================================
# كشف التزييف - تحليل الصور والفيديو
# ============================================================
@app.route('/api/detect', methods=['POST'])
def detect():
    """
    هذا اهم endpoint عندنا
    يستقبل صوره او فيديو ويشغل عليها الموديل
    ويرجع النتيجه: حقيقيه او مزيفه + نوع الفلتر
    """
    # نتاكد ان الموديل شغال
    if detector is None:
        return jsonify({'success': False, 'error': 'Model not loaded!'}), 500

    # نستقبل الملف - يمكن يجي باسم image او file
    file = request.files.get('image') or request.files.get('file')

    if file is None:
        return jsonify({'success': False, 'error': 'No file uploaded'}), 400
    if file.filename == '':
        return jsonify({'success': False, 'error': 'Empty filename'}), 400

    original_filename = file.filename
    ext = os.path.splitext(original_filename.lower())[1]

    # نشوف اذا صوره او فيديو
    is_video = ext in ALLOWED_VIDEO_EXT
    is_image = ext in ALLOWED_IMAGE_EXT

    if not is_video and not is_image:
        return jsonify({
            'success': False,
            'error': f'Unsupported file type: {ext}',
            'supported_images': list(ALLOWED_IMAGE_EXT),
            'supported_videos': list(ALLOWED_VIDEO_EXT)
        }), 400

    # نشيك حجم الفيديو عشان مايكون كبير ويطول
    if is_video:
        file.seek(0, 2)          # نروح نهاية الملف
        file_size = file.tell()  # ناخذ الحجم
        file.seek(0)             # نرجع البدايه
        if file_size > MAX_VIDEO_SIZE:
            return jsonify({
                'success': False,
                'error': f'Video too large: {file_size / 1024 / 1024:.1f}MB (max {MAX_VIDEO_SIZE / 1024 / 1024:.0f}MB)'
            }), 400

    # نحفظ الملف باسم فيه وقت عشان مايتكرر
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    stored_filename = f"{timestamp}_{original_filename}"
    filepath = os.path.join(UPLOAD_FOLDER, stored_filename)
    file.save(filepath)

    try:
        if is_video:
            # --- تحليل الفيديو ---
            # يستخرج 20 فريم ويحلل كل واحد لحاله
            is_fake, confidence, details = detector.analyze_video(filepath, original_filename)
            return jsonify({
                'success': True,
                'media_type': 'video',
                'prediction': details.get('predicted_class', 'unknown'),
                'confidence': confidence,
                'risk_level': details.get('risk_level', 'UNKNOWN'),
                'description': details.get('description', ''),
                'probabilities': details.get('probabilities', {}),
                'video_metadata': details.get('video_metadata', {}),
                'analysis_details': details.get('analysis_details', {}),
                'frame_results': details.get('frame_results', []),
                'details': details
            })
        else:
            # --- تحليل الصوره ---
            # اول يكشف الوجه بعدين يحلل بالموديلين (ديبفيك + فلتر)
            is_fake, confidence, details = detector.analyze_image(filepath, original_filename)
            return jsonify({
                'success': True,
                'media_type': 'image',
                'prediction': details.get('predicted_class', 'unknown'),
                'confidence': confidence,
                'risk_level': details.get('risk_level', 'UNKNOWN'),
                'description': details.get('description', ''),
                'details': details,
                'filter': details.get('filter', {}),
                'preprocessing': details.get('preprocessing', {})
            })

    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


# ============================================================
# الاحصائيات - بيانات الداشبورد
# ============================================================
@app.route('/api/stats')
def get_stats():
    """
    يرجع كل البيانات الي يحتاجها الداشبورد:
    عدد التحليلات، التوزيع، النشاط الاسبوعي، الخ
    """
    conn = get_db()
    c = conn.cursor()

    # عدد كل نوع (حقيقي، مزيف)
    c.execute('SELECT result, COUNT(*) as count FROM detections GROUP BY result')
    class_counts = {row['result']: row['count'] for row in c.fetchall()}
    total = sum(class_counts.values())

    # عدد حسب نوع الملف (صوره او فيديو)
    c.execute('SELECT media_type, COUNT(*) as count FROM detections GROUP BY media_type')
    media_counts = {row['media_type']: row['count'] for row in c.fetchall()}

    # النشاط الاسبوعي - اخر 7 ايام
    weekly = []
    for i in range(6, -1, -1):
        day = datetime.now() - timedelta(days=i)
        day_str = day.strftime('%Y-%m-%d')
        day_name = day.strftime('%a')
        c.execute('SELECT COUNT(*) as count FROM detections WHERE DATE(created_at) = ?', (day_str,))
        count = c.fetchone()['count']
        weekly.append({'day': day_name, 'count': count})

    # اخر 5 تحليلات
    c.execute('SELECT * FROM detections ORDER BY created_at DESC LIMIT 5')
    recent = [dict(row) for row in c.fetchall()]

    # احصائيات الفيديو
    c.execute('''SELECT COUNT(*) as cnt, AVG(video_deepfake_ratio) as avg_ratio,
                 AVG(video_frames_analyzed) as avg_frames
                 FROM detections WHERE media_type = 'video' ''')
    vrow = c.fetchone()
    video_stats = {
        'total_videos': vrow['cnt'] or 0,
        'avg_deepfake_ratio': round(vrow['avg_ratio'] or 0, 2),
        'avg_frames_analyzed': round(vrow['avg_frames'] or 0, 1)
    }

    conn.close()

    # معلومات الجهاز
    gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'

    # قدرات النظام
    capabilities = ['Image Detection']
    if detector and detector.video_supported:
        capabilities.append('Video Detection')
    if detector and detector.filter_model:
        capabilities.append(f'Filter Detection ({len(detector.filter_classes)} types)')

    return jsonify({
        'success': True,
        'total': total,
        'class_counts': {
            'real': class_counts.get('real', 0),
            'deepfake': class_counts.get('deepfake', 0),
            'filtered': class_counts.get('filtered', 0)
        },
        'media_counts': media_counts,
        'video_stats': video_stats,
        'weekly': weekly,
        'recent': recent,
        'model': {
            'architecture': 'EfficientNet-B4',
            'accuracy': f"{detector.deepfake_acc:.2f}%" if detector else '0%',
            'training_data': '243,968 images',
            'classes': 2,
            'gpu': gpu_name,
            'capabilities': capabilities,
            'filter_model': bool(detector and detector.filter_model),
            'filter_accuracy': f"{detector.filter_acc:.2f}%" if detector and detector.filter_model else None,
            'filter_classes': detector.filter_classes if detector and detector.filter_model else None,
            'face_detector': detector.face_extractor.method if detector else None
        }
    })


# ============================================================
# السجل - التحليلات السابقه
# ============================================================
@app.route('/api/history')
def get_history():
    """
    يرجع قائمة التحليلات السابقه مع صفحات
    يقدر المستخدم يفلتر بالنوع او يبحث باسم الملف
    """
    conn = get_db()
    c = conn.cursor()

    # باراميترات البحث والصفحات
    page = request.args.get('page', 1, type=int)
    per_page = request.args.get('per_page', 10, type=int)
    filter_class = request.args.get('filter', 'all')
    media_filter = request.args.get('media_type', 'all')
    search = request.args.get('search', '')

    # نبني شرط WHERE حسب الفلاتر المختاره
    where_clauses = []
    params = []

    if filter_class != 'all':
        where_clauses.append('result = ?')
        params.append(filter_class)

    if media_filter != 'all':
        where_clauses.append('media_type = ?')
        params.append(media_filter)

    if search:
        where_clauses.append('original_filename LIKE ?')
        params.append(f'%{search}%')

    where_sql = ''
    if where_clauses:
        where_sql = 'WHERE ' + ' AND '.join(where_clauses)

    # العدد الكلي عشان الصفحات
    c.execute(f'SELECT COUNT(*) as count FROM detections {where_sql}', params)
    total = c.fetchone()['count']

    # نجيب الصفحه المطلوبه
    offset = (page - 1) * per_page
    c.execute(
        f'SELECT * FROM detections {where_sql} ORDER BY created_at DESC LIMIT ? OFFSET ?',
        params + [per_page, offset]
    )
    rows = [dict(row) for row in c.fetchall()]
    conn.close()

    total_pages = max(1, (total + per_page - 1) // per_page)

    return jsonify({
        'success': True,
        'detections': rows,
        'page': page,
        'per_page': per_page,
        'total': total,
        'total_pages': total_pages
    })


# ============================================================
# حذف تحليل
# ============================================================
@app.route('/api/history/<int:detection_id>', methods=['DELETE'])
def delete_detection(detection_id):
    """يحذف تحليل معين مع ملفه من الجهاز"""
    conn = get_db()
    c = conn.cursor()

    # نلقى السجل عشان ناخذ اسم الملف
    c.execute('SELECT filename FROM detections WHERE id = ?', (detection_id,))
    row = c.fetchone()

    if row:
        # نحذف الملف من الجهاز
        filepath = os.path.join(UPLOAD_FOLDER, row['filename'])
        if os.path.exists(filepath):
            os.remove(filepath)

        # نحذف السجل من قاعدة البيانات
        c.execute('DELETE FROM detections WHERE id = ?', (detection_id,))
        conn.commit()
        conn.close()
        return jsonify({'success': True, 'message': 'Deleted'})

    conn.close()
    return jsonify({'success': False, 'error': 'Not found'}), 404


# ============================================================
# الاعدادات
# ============================================================
@app.route('/api/settings', methods=['GET'])
def get_settings():
    """يرجع الاعدادات المحفوظه (الوضع الليلي، التنبيهات، الخ)"""
    conn = get_db()
    c = conn.cursor()
    c.execute('SELECT key, value FROM settings')
    settings = {row['key']: row['value'] for row in c.fetchall()}
    conn.close()
    return jsonify({'success': True, 'settings': settings})


@app.route('/api/settings', methods=['POST'])
def save_settings():
    """يحفظ او يحدث الاعدادات"""
    data = request.json
    conn = get_db()
    c = conn.cursor()
    for key, value in data.items():
        c.execute('INSERT OR REPLACE INTO settings (key, value) VALUES (?, ?)', (key, str(value)))
    conn.commit()
    conn.close()
    return jsonify({'success': True, 'message': 'Settings saved'})


# ============================================================
# معلومات النظام
# ============================================================
@app.route('/api/info')
def info():
    """يرجع معلومات اساسيه عن النظام - مفيد عشان نشيك اذا شغال"""
    return jsonify({
        'status': 'running',
        'model': 'EfficientNet-B4',
        'classes': ['Real', 'Deepfake'],
        'accuracy': f"{detector.deepfake_acc:.2f}%" if detector else 'N/A',
        'capabilities': {
            'image_detection': True,
            'video_detection': detector.video_supported if detector else False,
            'filter_detection': detector.filter_model is not None if detector else False
        },
        'supported_formats': {
            'images': list(ALLOWED_IMAGE_EXT),
            'videos': list(ALLOWED_VIDEO_EXT)
        }
    })


# ============================================================
# تشغيل السيرفر
# ============================================================
if __name__ == '__main__':
    print("\n" + "=" * 60)
    print("  Deepfake Detection System")
    print("=" * 60)

    if detector:
        print(f"  Model: EfficientNet-B4")
        print(f"  Deepfake Accuracy: {detector.deepfake_acc:.2f}%")
        if detector.filter_model:
            print(f"  Filter Detection: Enabled ({len(detector.filter_classes)} types, {detector.filter_acc:.2f}%)")
        else:
            print(f"  Filter Detection: Disabled")
        print(f"  Face Detector: {detector.face_extractor.method or 'None'}")
        print(f"  Video Support: {'Yes' if detector.video_supported else 'No'}")
        print(f"  GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
    else:
        print("  WARNING: Detection model failed to load!")

    print(f"  Server: http://localhost:5000")
    print("=" * 60 + "\n")

    app.run(debug=True, port=5000, use_reloader=False)