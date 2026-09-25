# ============================================================
# train_v2.py - سكربت تدريب موديل كشف الديبفيك
# ============================================================
# هذا السكربت يدرّب موديل EfficientNet-B4 على بيانات الديبفيك
# الموديل يصنف الصور: حقيقيه (real) او مزيفه (deepfake)
#
# البيانات:
#   - real: صور حقيقيه من FFHQ و CelebA (Kaggle)
#   - deepfake: صور مزيفه من FaceForensics++ و DFDC (Kaggle)
#   - ai_generated: صور مولده بالذكاء الاصطناعي من StyleGAN (Kaggle)
#     * دمجناها مع الديبفيك عشان التصنيف الثنائي اسهل وادق
#
# التقنيات المستخدمه:
#   - Transfer Learning: ناخذ موديل متدرب على ImageNet ونعدل الطبقه الاخيره
#   - Mixed Precision (FP16): يسرّع التدريب ويقلل استخدام الذاكره
#   - Cosine Annealing LR: يقلل معدل التعلم تدريجيا
#   - Data Augmentation: تدوير وقلب وتغيير الوان عشان الموديل يتعلم افضل
#   - Label Smoothing: يمنع الموديل يكون واثق زياده
#   - Early Stopping: يوقف التدريب اذا ماتحسن 3 مرات متتاليه
#
# النتيجه: 99.36% دقه على بيانات التحقق
# ============================================================

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
import os
import time
import random
import numpy as np
from PIL import Image
from datetime import datetime


# ============================================================
# الاعدادات
# ============================================================
# غيّر dataset_path لمسار الداتاسيت عندك
CONFIG = {
    'dataset_path': r'C:\Users\qlxk1\deepfake-ultimate-dataset',
    'batch_size': 24,
    'epochs': 15,
    'learning_rate': 1e-4,
    'weight_decay': 1e-4,
    'image_size': 380,       # نكبّر الصوره لهذا الحجم
    'crop_size': 299,        # بعدين ناخذ المنتصف بهذا الحجم
    'num_workers': 4,        # عدد العمليات لتحميل البيانات
    'max_images_per_class': 150000,  # اقصى عدد صور لكل كلاس
    'val_split': 0.1,        # 10% للتحقق
    'patience': 3,           # نوقف اذا ماتحسن 3 مرات
    'model_name': 'deepfake_2class_v2.pth',
    'save_dir': r'backend\models\saved_models',
}


# ============================================================
# كلاس الداتاسيت
# ============================================================
# يحمّل الصور من المسارات ويطبق عليها المعالجه

class DeepfakeDataset(Dataset):
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        try:
            image = Image.open(self.image_paths[idx]).convert('RGB')
            if self.transform:
                image = self.transform(image)
            return image, self.labels[idx]
        except:
            # اذا صار خطأ بالصوره نجيب وحده ثانيه عشوائيه
            new_idx = random.randint(0, len(self) - 1)
            return self.__getitem__(new_idx)


def collect_images(folder_path, max_count=None):
    """يجمع مسارات الصور من مجلد (يدخل المجلدات الفرعيه بعد)"""
    valid_ext = {'.jpg', '.jpeg', '.png', '.webp', '.bmp', '.tiff', '.gif'}
    images = []

    if not os.path.exists(folder_path):
        print(f"   Not found: {folder_path}")
        return images

    for root, dirs, files in os.walk(folder_path):
        for f in files:
            if os.path.splitext(f)[1].lower() in valid_ext:
                images.append(os.path.join(root, f))

    if max_count and len(images) > max_count:
        random.shuffle(images)
        images = images[:max_count]

    return images


# ============================================================
# دالة التدريب
# ============================================================
def train():
    print("=" * 60)
    print(" DEEPFAKE DETECTION MODEL TRAINING")
    print("=" * 60)
    print(f"  Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"  Device: {device}")
    if torch.cuda.is_available():
        print(f"  GPU: {torch.cuda.get_device_name(0)}")

    # ---- نجمع البيانات ----
    print("\n  Loading dataset...")
    dataset_path = CONFIG['dataset_path']
    max_per_class = CONFIG['max_images_per_class']

    # كلاس 0: مزيف (deepfake + ai_generated مدموجين مع بعض)
    deepfake_images = collect_images(os.path.join(dataset_path, 'deepfake'), max_per_class)
    ai_images = collect_images(os.path.join(dataset_path, 'ai_generated'), max_per_class)
    fake_images = deepfake_images + ai_images
    print(f"   [0] deepfake: {len(deepfake_images)} images")
    print(f"   [0] ai_generated: {len(ai_images)} images")
    print(f"   [0] total fake: {len(fake_images)} images")

    # كلاس 1: حقيقي
    real_images = collect_images(os.path.join(dataset_path, 'real'), max_per_class)
    print(f"   [1] real: {len(real_images)} images")

    if len(fake_images) == 0 or len(real_images) == 0:
        print("  Not enough data!")
        return

    # نوازن الكلاسات عشان ماتكون وحده اكثر من الثانيه
    min_count = min(len(fake_images), len(real_images))
    if len(fake_images) > min_count:
        random.shuffle(fake_images)
        fake_images = fake_images[:min_count]
    if len(real_images) > min_count:
        random.shuffle(real_images)
        real_images = real_images[:min_count]

    print(f"\n   Balanced: {min_count} per class")
    print(f"   Total: {min_count * 2} images")

    class_names = ['deepfake', 'real']

    # نجمعها ونخلطها
    all_paths = fake_images + real_images
    all_labels = [0] * len(fake_images) + [1] * len(real_images)

    combined = list(zip(all_paths, all_labels))
    random.shuffle(combined)
    all_paths, all_labels = zip(*combined)
    all_paths, all_labels = list(all_paths), list(all_labels)

    # نقسمها: 90% تدريب + 10% تحقق
    val_size = int(len(all_paths) * CONFIG['val_split'])
    train_paths, val_paths = all_paths[val_size:], all_paths[:val_size]
    train_labels, val_labels = all_labels[val_size:], all_labels[:val_size]

    print(f"   Train: {len(train_paths)} | Val: {len(val_paths)}")

    # ---- معالجة الصور ----
    # التدريب: نضيف تنويع عشان الموديل يتعلم افضل
    train_transform = transforms.Compose([
        transforms.Resize((CONFIG['image_size'], CONFIG['image_size'])),
        transforms.CenterCrop(CONFIG['crop_size']),
        transforms.RandomHorizontalFlip(p=0.5),       # قلب افقي عشوائي
        transforms.RandomRotation(10),                  # تدوير خفيف
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        transforms.RandomErasing(p=0.1)                # مسح جزء عشوائي
    ])

    # التحقق: بدون تنويع - نبي نشوف الاداء الحقيقي
    val_transform = transforms.Compose([
        transforms.Resize((CONFIG['image_size'], CONFIG['image_size'])),
        transforms.CenterCrop(CONFIG['crop_size']),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    train_dataset = DeepfakeDataset(train_paths, train_labels, train_transform)
    val_dataset = DeepfakeDataset(val_paths, val_labels, val_transform)

    train_loader = DataLoader(train_dataset, batch_size=CONFIG['batch_size'],
                              shuffle=True, num_workers=CONFIG['num_workers'],
                              pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=CONFIG['batch_size'],
                            shuffle=False, num_workers=CONFIG['num_workers'],
                            pin_memory=True)

    # ---- بناء الموديل ----
    print("\n  Building model...")

    # نجيب EfficientNet-B4 المتدرب على ImageNet
    model = models.efficientnet_b4(weights='IMAGENET1K_V1')

    # نجمد الطبقات الاولى عشان نحتفظ بالمميزات العامه
    for name, param in model.features[:5].named_parameters():
        param.requires_grad = False

    # نغير الطبقه الاخيره من 1000 كلاس الى 2 كلاس
    num_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.4),
        nn.Linear(num_features, 512),
        nn.ReLU(),
        nn.Dropout(p=0.3),
        nn.Linear(512, 2)
    )
    model = model.to(device)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"   Parameters: {trainable:,} trainable / {total_params:,} total")

    # ---- اعدادات التدريب ----
    criterion = nn.CrossEntropyLoss(label_smoothing=0.05)
    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=CONFIG['learning_rate'], weight_decay=CONFIG['weight_decay']
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=CONFIG['epochs'], eta_min=1e-6
    )
    scaler = torch.cuda.amp.GradScaler()

    # ---- حلقة التدريب ----
    best_score = 0
    patience_counter = 0
    os.makedirs(CONFIG['save_dir'], exist_ok=True)
    save_path = os.path.join(CONFIG['save_dir'], CONFIG['model_name'])

    print(f"\n{'='*60}")
    print(f"  Starting training: {CONFIG['epochs']} epochs")
    print(f"{'='*60}")

    for epoch in range(CONFIG['epochs']):
        epoch_start = time.time()

        # --- التدريب ---
        model.train()
        train_loss = 0
        train_correct = 0
        train_total = 0

        for batch_idx, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            with torch.cuda.amp.autocast():
                outputs = model(images)
                loss = criterion(outputs, labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            train_loss += loss.item()
            _, predicted = outputs.max(1)
            train_total += labels.size(0)
            train_correct += predicted.eq(labels).sum().item()

            if (batch_idx + 1) % 100 == 0:
                print(f"   Batch {batch_idx+1}/{len(train_loader)} | "
                      f"Loss: {train_loss/(batch_idx+1):.4f} | "
                      f"Acc: {100.*train_correct/train_total:.1f}%")

        train_acc = 100. * train_correct / train_total
        train_loss = train_loss / len(train_loader)

        # --- التحقق ---
        model.eval()
        val_loss = 0
        val_correct = 0
        val_total = 0
        class_correct = [0, 0]
        class_total = [0, 0]

        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                with torch.cuda.amp.autocast():
                    outputs = model(images)
                    loss = criterion(outputs, labels)

                val_loss += loss.item()
                _, predicted = outputs.max(1)
                val_total += labels.size(0)
                val_correct += predicted.eq(labels).sum().item()

                for i in range(len(labels)):
                    label = labels[i].item()
                    class_total[label] += 1
                    if predicted[i].item() == label:
                        class_correct[label] += 1

        val_acc = 100. * val_correct / val_total
        val_loss = val_loss / len(val_loader)

        class_accs = []
        for i in range(2):
            acc = 100. * class_correct[i] / class_total[i] if class_total[i] > 0 else 0
            class_accs.append(acc)
        min_class_acc = min(class_accs)

        scheduler.step()
        epoch_time = time.time() - epoch_start

        score = val_acc * 0.6 + min_class_acc * 0.4

        print(f"\n{'='*60}")
        print(f"  Epoch {epoch+1}/{CONFIG['epochs']}")
        print(f"   Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
        print(f"   Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%")
        print(f"   Deepfake Acc: {class_accs[0]:.2f}% | Real Acc: {class_accs[1]:.2f}%")
        print(f"   Time: {epoch_time/60:.1f} min")

        if score > best_score:
            best_score = score
            patience_counter = 0
            torch.save({
                'model_state_dict': model.state_dict(),
                'class_names': class_names,
                'val_acc': val_acc,
                'min_class_acc': min_class_acc,
                'class_accuracies': {class_names[i]: class_accs[i] for i in range(2)},
                'epoch': epoch + 1,
                'config': CONFIG
            }, save_path)
            print(f"   SAVED! Best acc: {val_acc:.2f}%")
        else:
            patience_counter += 1
            print(f"   No improvement ({patience_counter}/{CONFIG['patience']})")
            if patience_counter >= CONFIG['patience']:
                print(f"\n  Early stopping at epoch {epoch+1}")
                break

    print(f"\n{'='*60}")
    print(f"  TRAINING COMPLETE!")
    print(f"{'='*60}")
    print(f"   Best Accuracy: {val_acc:.2f}%")
    print(f"   Model saved: {save_path}")
    print(f"   Finished: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # اختبار سريع
    print("\n  Quick test...")
    model.eval()
    correct = 0
    total = 100
    test_indices = random.sample(range(len(val_dataset)), min(total, len(val_dataset)))
    for idx in test_indices:
        img, label = val_dataset[idx]
        img = img.unsqueeze(0).to(device)
        with torch.no_grad():
            output = model(img)
            pred = output.argmax(1).item()
        if pred == label:
            correct += 1
    print(f"   Result: {correct}/{total} correct ({100*correct/total:.1f}%)")


if __name__ == '__main__':
    # نثبت البذره عشان النتائج تكون قابله للتكرار
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    train()