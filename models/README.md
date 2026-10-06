# โฟลเดอร์โมเดล

โฟลเดอร์นี้ใช้เก็บโมเดลจำแนกภาพที่เทรนเอง ซึ่งเป็นส่วนเสริม ไม่จำเป็นต่อการใช้งานระบบ

หลังรัน `scripts/train_model.py` จะได้ไฟล์

- `watermelon_cls.onnx` — โมเดลสำหรับ onnxruntime
- `watermelon_cls.labels.json` — รายชื่อคลาส ขนาดภาพเข้า และความแม่นยำบนชุดตรวจสอบ

จากนั้นเปิดใช้งานโดยเพิ่มใน `.env`

```ini
MELON_ONNX_MODEL=models/watermelon_cls.onnx
MELON_ONNX_LABELS=models/watermelon_cls.labels.json
```

ไฟล์โมเดล (`*.onnx`, `*.pt`, `*.pth`, `*.tflite`, `*.keras`) ถูกกำหนดให้ git ไม่ติดตาม
เพราะมีขนาดใหญ่ ให้เก็บไว้นอก repository หรือใช้ Git LFS หากจำเป็นต้องแชร์
