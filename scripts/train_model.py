#!/usr/bin/env python3
"""เทรนโมเดลจำแนกภาพโรคแตงโมของตัวเอง แล้วส่งออกเป็น ONNX ให้ระบบใช้งาน

โมเดลที่ได้จะถูกนำไปผสมกับผลวิเคราะห์จาก Claude (ensemble) เพื่อเพิ่มความมั่นใจ
หรือใช้เป็นตัววิเคราะห์หลักเมื่อไม่มีอินเทอร์เน็ต

โครงสร้างโฟลเดอร์ข้อมูลที่ต้องเตรียม (ชื่อโฟลเดอร์ต้องตรงกับรหัสโรคในคลังความรู้):
    datasets/watermelon/train/anthracnose/*.jpg
    datasets/watermelon/train/downy_mildew/*.jpg
    datasets/watermelon/train/healthy/*.jpg
    datasets/watermelon/val/anthracnose/*.jpg
    ...

ตัวอย่างการใช้งาน:
    python3 scripts/train_model.py --list-classes
    python3 scripts/train_model.py --check-data datasets/watermelon
    python3 scripts/train_model.py --data datasets/watermelon --epochs 15 --batch-size 32
    # จากนั้นตั้งค่าใน .env:
    #   MELON_ONNX_MODEL=models/watermelon_cls.onnx
    #   MELON_ONNX_LABELS=models/watermelon_cls.labels.json

ต้องติดตั้งเพิ่ม:  pip install torch torchvision onnx onnxruntime
(สำหรับ CPU ใช้:  pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu)
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
MODEL_DIR = ROOT / "models"
IMAGE_EXT = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}

TORCH_HINT = (
    "ไม่พบ PyTorch กรุณาติดตั้งก่อน:\n"
    "    pip install torch torchvision\n"
    "สำหรับเครื่องที่ไม่มี GPU:\n"
    "    pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu"
)


def known_class_ids() -> list[str]:
    """รหัสโรคทั้งหมดจากคลังความรู้ บวกคลาส healthy สำหรับต้นปกติ"""
    ids: list[str] = ["healthy"]
    diseases_dir = DATA_DIR / "diseases"
    for path in sorted(diseases_dir.glob("*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        for item in payload.get("items", []):
            ids.append(item["id"])
    return ids


def cmd_list_classes() -> int:
    ids = known_class_ids()
    print(f"รหัสคลาสที่ใช้ตั้งชื่อโฟลเดอร์ได้ทั้งหมด {len(ids)} รายการ:\n")
    for class_id in ids:
        print(f"  {class_id}")
    print(
        "\nคำแนะนำ: ไม่จำเป็นต้องมีครบทุกคลาส ให้เริ่มจากโรคที่พบบ่อยในแปลงของคุณ 5-10 คลาส "
        "และควรมีภาพอย่างน้อย 100-200 ภาพต่อคลาสเพื่อให้โมเดลใช้งานได้จริง"
    )
    return 0


def cmd_check_data(data_root: Path) -> int:
    """ตรวจโครงสร้างโฟลเดอร์และจำนวนภาพ พร้อมเทียบชื่อคลาสกับคลังความรู้"""
    known = set(known_class_ids())
    problems = 0
    print(f"=== ตรวจชุดข้อมูลที่ {data_root} ===\n")
    for split in ("train", "val"):
        split_dir = data_root / split
        if not split_dir.is_dir():
            print(f"  [ผิดพลาด] ไม่พบโฟลเดอร์ {split_dir}")
            problems += 1
            continue
        counts: Counter[str] = Counter()
        for class_dir in sorted(p for p in split_dir.iterdir() if p.is_dir()):
            images = [p for p in class_dir.rglob("*") if p.suffix.lower() in IMAGE_EXT]
            counts[class_dir.name] = len(images)
        print(f"  {split}: {len(counts)} คลาส รวม {sum(counts.values())} ภาพ")
        for name, count in counts.most_common():
            flags = []
            if name not in known:
                flags.append("ชื่อคลาสไม่ตรงกับรหัสในคลังความรู้")
                problems += 1
            if split == "train" and count < 50:
                flags.append("ภาพน้อยกว่า 50 ภาพ ความแม่นยำจะต่ำมาก")
            note = ("  <-- " + ", ".join(flags)) if flags else ""
            print(f"      {name:34s} {count:5d} ภาพ{note}")
        print()
    if problems:
        print(
            f"พบปัญหา {problems} จุด ชื่อโฟลเดอร์ต้องตรงกับรหัสโรค "
            "(ดูรายการด้วย --list-classes) ระบบจึงจะเชื่อมผลทำนายกับข้อมูลยาและปุ๋ยได้"
        )
    else:
        print("โครงสร้างข้อมูลถูกต้อง พร้อมเทรน")
    return 1 if problems else 0


def train(args: argparse.Namespace) -> int:
    try:
        import torch
        from torch import nn
        from torch.utils.data import DataLoader, WeightedRandomSampler
        from torchvision import datasets, models, transforms
    except ImportError:
        print(TORCH_HINT)
        return 1

    data_root = Path(args.data)
    train_dir, val_dir = data_root / "train", data_root / "val"
    if not train_dir.is_dir() or not val_dir.is_dir():
        print(f"ไม่พบโฟลเดอร์ {train_dir} หรือ {val_dir} (ตรวจด้วย --check-data ก่อน)")
        return 1

    size = args.image_size
    train_tf = transforms.Compose(
        [
            transforms.RandomResizedCrop(size, scale=(0.6, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(25),
            transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.04),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            transforms.RandomErasing(p=0.2, scale=(0.02, 0.1)),
        ]
    )
    eval_tf = transforms.Compose(
        [
            transforms.Resize(int(size * 1.15)),
            transforms.CenterCrop(size),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )

    train_ds = datasets.ImageFolder(train_dir, train_tf)
    val_ds = datasets.ImageFolder(val_dir, eval_tf)
    classes = train_ds.classes
    if val_ds.classes != classes:
        print("คลาสใน train และ val ไม่ตรงกัน กรุณาจัดโฟลเดอร์ให้เหมือนกันทั้งสองชุด")
        return 1
    print(f"คลาสทั้งหมด {len(classes)} คลาส: {', '.join(classes)}")
    print(f"ภาพสำหรับเทรน {len(train_ds)} ภาพ, ตรวจสอบ {len(val_ds)} ภาพ")

    # ถ่วงน้ำหนักการสุ่มตัวอย่างเพื่อชดเชยจำนวนภาพที่ไม่เท่ากันระหว่างคลาส
    label_counts = Counter(label for _, label in train_ds.samples)
    weights = [1.0 / label_counts[label] for _, label in train_ds.samples]
    sampler = WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)

    train_dl = DataLoader(
        train_ds, batch_size=args.batch_size, sampler=sampler, num_workers=args.workers, pin_memory=True
    )
    val_dl = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"อุปกรณ์ที่ใช้เทรน: {device}")

    if args.arch == "mobilenet_v3_large":
        model = models.mobilenet_v3_large(weights=models.MobileNet_V3_Large_Weights.DEFAULT)
        model.classifier[3] = nn.Linear(model.classifier[3].in_features, len(classes))
    elif args.arch == "efficientnet_b0":
        model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, len(classes))
    else:
        model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        model.fc = nn.Linear(model.fc.in_features, len(classes))
    model.to(device)

    criterion = nn.CrossEntropyLoss(label_smoothing=0.05)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_acc, best_state, patience = 0.0, None, 0
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        model.train()
        running, seen, correct = 0.0, 0, 0
        for images, labels in train_dl:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running += loss.item() * labels.size(0)
            seen += labels.size(0)
            correct += (outputs.argmax(1) == labels).sum().item()
        scheduler.step()
        train_loss, train_acc = running / max(seen, 1), correct / max(seen, 1)

        model.eval()
        v_correct, v_seen, v_loss = 0, 0, 0.0
        confusion = [[0] * len(classes) for _ in classes]
        with torch.no_grad():
            for images, labels in val_dl:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                v_loss += criterion(outputs, labels).item() * labels.size(0)
                preds = outputs.argmax(1)
                v_correct += (preds == labels).sum().item()
                v_seen += labels.size(0)
                for truth, pred in zip(labels.tolist(), preds.tolist()):
                    confusion[truth][pred] += 1
        val_acc = v_correct / max(v_seen, 1)
        print(
            f"epoch {epoch:3d}/{args.epochs} | train loss {train_loss:.4f} acc {train_acc:.3f}"
            f" | val loss {v_loss/max(v_seen,1):.4f} acc {val_acc:.3f}"
        )

        if val_acc > best_acc:
            best_acc, patience = val_acc, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            best_confusion = [row[:] for row in confusion]
        else:
            patience += 1
            if patience >= args.patience:
                print(f"ไม่ดีขึ้นต่อเนื่อง {patience} รอบ จึงหยุดการเทรน (early stopping)")
                break

    if best_state is None:
        print("ไม่สามารถเทรนได้สำเร็จ")
        return 1
    model.load_state_dict(best_state)
    print(f"\nความแม่นยำที่ดีที่สุดบนชุดตรวจสอบ: {best_acc:.3f}")
    report_metrics(classes, best_confusion)

    out_onnx = MODEL_DIR / f"{args.name}.onnx"
    out_labels = MODEL_DIR / f"{args.name}.labels.json"
    model.eval().to("cpu")
    dummy = torch.randn(1, 3, size, size)
    torch.onnx.export(
        model,
        dummy,
        str(out_onnx),
        input_names=["input"],
        output_names=["logits"],
        dynamic_axes={"input": {0: "batch"}, "logits": {0: "batch"}},
        opset_version=17,
    )
    out_labels.write_text(
        json.dumps(
            {
                "labels": classes,
                "input_size": size,
                "arch": args.arch,
                "val_accuracy": round(best_acc, 4),
                "train_images": len(train_ds),
                "note": "ชื่อคลาสต้องตรงกับรหัสโรคใน data/diseases/*.json จึงจะเชื่อมกับข้อมูลยาและปุ๋ยได้",
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"\nบันทึกโมเดลแล้ว: {out_onnx}")
    print(f"บันทึกรายชื่อคลาส: {out_labels}")
    print("\nเปิดใช้งานโดยเพิ่มใน .env:")
    print(f"    MELON_ONNX_MODEL={out_onnx.relative_to(ROOT)}")
    print(f"    MELON_ONNX_LABELS={out_labels.relative_to(ROOT)}")
    return 0


def report_metrics(classes: list[str], confusion: list[list[int]]) -> None:
    """รายงาน precision/recall/F1 รายคลาส โดยไม่ต้องใช้ scikit-learn"""
    print("\nผลรายคลาสบนชุดตรวจสอบ:")
    print(f"  {'คลาส':34s} {'precision':>10s} {'recall':>8s} {'f1':>7s} {'จำนวน':>7s}")
    for i, name in enumerate(classes):
        tp = confusion[i][i]
        fn = sum(confusion[i]) - tp
        fp = sum(confusion[r][i] for r in range(len(classes))) - tp
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
        print(f"  {name:34s} {precision:10.3f} {recall:8.3f} {f1:7.3f} {sum(confusion[i]):7d}")
    weakest = [
        classes[i]
        for i in range(len(classes))
        if sum(confusion[i]) and confusion[i][i] / sum(confusion[i]) < 0.6
    ]
    if weakest:
        print(
            "\nคลาสที่โมเดลยังแยกได้ไม่ดี (recall ต่ำกว่า 0.6): "
            + ", ".join(weakest)
            + "\nแนะนำให้เพิ่มภาพของคลาสเหล่านี้ โดยเฉพาะภาพที่ถ่ายในสภาพแสงและมุมที่หลากหลาย"
        )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="เทรนโมเดลจำแนกภาพโรคแตงโมและส่งออกเป็น ONNX",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--list-classes", action="store_true", help="แสดงรหัสคลาสที่ใช้ได้")
    parser.add_argument("--check-data", metavar="DIR", help="ตรวจโครงสร้างชุดข้อมูลก่อนเทรน")
    parser.add_argument("--data", default="datasets/watermelon", help="โฟลเดอร์ข้อมูล")
    parser.add_argument("--name", default="watermelon_cls", help="ชื่อไฟล์โมเดลที่จะบันทึก")
    parser.add_argument(
        "--arch",
        default="mobilenet_v3_large",
        choices=["mobilenet_v3_large", "efficientnet_b0", "resnet50"],
        help="สถาปัตยกรรมโมเดล (mobilenet เล็กและเร็วที่สุด เหมาะกับการรันบน CPU)",
    )
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--patience", type=int, default=5, help="หยุดเทรนเมื่อไม่ดีขึ้นกี่รอบ")
    args = parser.parse_args()

    if args.list_classes:
        return cmd_list_classes()
    if args.check_data:
        return cmd_check_data(Path(args.check_data))
    return train(args)


if __name__ == "__main__":
    try:
        sys.exit(main())
    except BrokenPipeError:
        # เกิดเมื่อส่งผลลัพธ์ต่อให้คำสั่งอย่าง head ที่ปิดท่อก่อน ไม่ใช่ข้อผิดพลาดจริง
        sys.stderr.close()
        sys.exit(0)
    except KeyboardInterrupt:
        print("\nยกเลิกการทำงาน")
        sys.exit(130)
