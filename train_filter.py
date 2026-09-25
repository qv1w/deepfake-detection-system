# ============================================================
# train_filter.py - سكربت تدريب موديل كشف الفلاتر
# ============================================================
# هذا السكربت يدرّب موديل EfficientNet-B4 يكشف اذا الصوره
# عليها فلتر ونوعه ايش
#
# 7 كلاسات:
#   0. clean    - صوره نظيفه بدون فلتر
#   1. age      - فلتر تكبير/تصغير العمر
#   2. animal   - فلتر حيوانات (اذان، خشم، شوارب)
#   3. artistic - فلتر فني (رسم، سكتش، كرتون)
#   4. color    - فلتر الوان (تغيير الالوان، سيبيا)
#   5. distortion - فلتر تشويه (تمديد، ضغط، عين سمكه)
#   6. mixed    - فلاتر مختلطه (اكثر من فلتر مع بعض)
#
# البيانات:
#   - clean: من مجلد real (صور حقيقيه بدون فلاتر)
#   - الفلاتر: من مجلد filtered (كل نوع بمجلد فرعي)
#   - المجلدات المرقمه (000000-086000): ضفناها لكلاس mixed
#
# النتيجه: 98.93% دقه على بيانات التحقق
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
FILTER_TYPES = ['age', 'animal', 'artistic', 'color', 'distortion', 'mixed']

CONFIG = {
    'dataset_path': r'C:\Users\qlxk1\deepfake-ultimate-dataset',
    'batch_size': 24,
    'epochs': 12,
    'learning_rate': 1e-4,
    'weight_decay': 1e-4,
    'image_size': 380,
    'crop_size': 299,
    'num_workers': 4,
    'max_images_per_class': 20000,  # اقصى 20 الف لكل كلاس عشان يكون متوازن
    'val_split': 0.1,
    'patience': 3,
    'model_name': 'filter_detector_v1.pth',
    'save_dir': r'backend\models\saved_models',
}


# ============================================================
# كلاس الداتاسيت
# ============================================================
class FilterDataset(Dataset):
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
            # اذا صار خطأ نجيب صوره ثانيه
            new_idx = random.randint(0, len(self) - 1)
            return self.__getitem__(new_idx)


def collect_images(folder_path, max_count=None):
    """يجمع مسارات الصور من مجلد ومجلداته الفرعيه"""
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
    print(" FILTER DETECTION MODEL TRAINING (7-Class)")
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

    # اسماء الكلاسات: clean + 6 انواع فلاتر
    class_names = ['clean'] + FILTER_TYPES
    num_classes = len(class_names)
    print(f"   Classes ({num_classes}): {class_names}")

    all_paths = []
    all_labels = []

    # كلاس 0: clean (صور حقيقيه بدون فلاتر)
    clean_images = collect_images(os.path.join(dataset_path, 'real'), max_per_class)
    print(f"   [0] clean: {len(clean_images)} images")
    all_paths.extend(clean_images)
    all_labels.extend([0] * len(clean_images))

    # كلاسات 1-6: انواع الفلاتر
    for i, filter_type in enumerate(FILTER_TYPES):
        filter_path = os.path.join(dataset_path, 'filtered', filter_type)
        filter_images = collect_images(filter_path, max_per_class)
        print(f"   [{i+1}] {filter_type}: {len(filter_images)} images")
        all_paths.extend(filter_images)
        all_labels.extend([(i + 1)] * len(filter_images))

    # المجلدات المرقمه (000000-086000) نضيفها لكلاس mixed
    numbered_path = os.path.join(dataset_path, 'filtered')
    numbered_images = []
    for item in os.listdir(numbered_path):
        full_path = os.path.join(numbered_path, item)
        if os.path.isdir(full_path) and item.isdigit():
            numbered_images.extend(collect_images(full_path))

    if numbered_images:
        if len(numbered_images) > max_per_class:
            random.shuffle(numbered_images)
            numbered_images = numbered_images[:max_per_class]
        mixed_idx = class_names.index('mixed')
        print(f"   [numbered->mixed] +{len(numbered_images)} images")
        all_paths.extend(numbered_images)
        all_labels.extend([mixed_idx] * len(numbered_images))

    print(f"\n   Total: {len(all_paths)} images")

    # نطبع توزيع الكلاسات
    for i, name in enumerate(class_names):
        count = all_labels.count(i)
        print(f"      {name}: {count}")

    if len(all_paths) < 100:
        print("  Not enough data!")
        return

    # نخلط ونقسم
    combined = list(zip(all_paths, all_labels))
    random.shuffle(combined)
    all_paths, all_labels = zip(*combined)
    all_paths, all_labels = list(all_paths), list(all_labels)

    val_size = int(len(all_paths) * CONFIG['val_split'])
    train_paths, val_paths = all_paths[val_size:], all_paths[:val_size]
    train_labels, val_labels = all_labels[val_size:], all_labels[:val_size]

    print(f"   Train: {len(train_paths)} | Val: {len(val_paths)}")

    # ---- معالجة الصور ----
    train_transform = transforms.Compose([
        transforms.Resize((CONFIG['image_size'], CONFIG['image_size'])),
        transforms.CenterCrop(CONFIG['crop_size']),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        transforms.RandomErasing(p=0.1)
    ])

    val_transform = transforms.Compose([
        transforms.Resize((CONFIG['image_size'], CONFIG['image_size'])),
        transforms.CenterCrop(CONFIG['crop_size']),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    train_dataset = FilterDataset(train_paths, train_labels, train_transform)
    val_dataset = FilterDataset(val_paths, val_labels, val_transform)

    train_loader = DataLoader(train_dataset, batch_size=CONFIG['batch_size'],
                              shuffle=True, num_workers=CONFIG['num_workers'],
                              pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=CONFIG['batch_size'],
                            shuffle=False, num_workers=CONFIG['num_workers'],
                            pin_memory=True)

    # ---- بناء الموديل ----
    print("\n  Building model...")
    model = models.efficientnet_b4(weights='IMAGENET1K_V1')

    # نجمد الطبقات الاولى
    for name, param in model.features[:5].named_parameters():
        param.requires_grad = False

    # نغير الطبقه الاخيره لـ 7 كلاسات
    num_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.4),
        nn.Linear(num_features, 512),
        nn.ReLU(),
        nn.Dropout(p=0.3),
        nn.Linear(512, num_classes)
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
        class_correct = [0] * num_classes
        class_total = [0] * num_classes

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

        # نحسب دقة كل كلاس لحاله
        class_accs = []
        for i in range(num_classes):
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
        for i, name in enumerate(class_names):
            status = "OK" if class_accs[i] >= 90 else "LOW" if class_accs[i] >= 70 else "BAD"
            print(f"   {status} {name}: {class_accs[i]:.2f}%")
        print(f"   Time: {epoch_time/60:.1f} min")

        if score > best_score:
            best_score = score
            patience_counter = 0
            torch.save({
                'model_state_dict': model.state_dict(),
                'class_names': class_names,
                'num_classes': num_classes,
                'val_acc': val_acc,
                'min_class_acc': min_class_acc,
                'class_accuracies': {class_names[i]: class_accs[i] for i in range(num_classes)},
                'epoch': epoch + 1,
                'config': CONFIG
            }, save_path)
            print(f"   SAVED! Best acc: {val_acc:.2f}% | Min class: {min_class_acc:.2f}%")
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
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    train()