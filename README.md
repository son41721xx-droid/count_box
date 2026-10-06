# YOLO box crossing counter

โปรแกรม Windows สำหรับนับกล่องจากกล้อง RTSP ด้วยโมเดล Ultralytics YOLO ที่ train แล้ว (`best.pt`)

- กล่องผ่านเส้นจากบนลงล่าง: `+1`
- กล่องผ่านเส้นจากล่างขึ้นบน: `-1`
- ลากเส้นสีแดงขึ้นหรือลงด้วยเมาส์เพื่อปรับตำแหน่ง
- กด `R` เพื่อล้างตัวนับ และกด `Q` หรือ `Esc` เพื่อออก

## ติดตั้งและรันใน VS Code บน Windows

เปิดโฟลเดอร์นี้ใน VS Code แล้ววาง `best.pt` ไว้ข้างไฟล์ `box_counter.py` จากนั้นเปิด Terminal แบบ PowerShell:

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
python box_counter.py
```

ค่าเริ่มต้นเชื่อมต่อ `rtsp://10.3.7.129:554/Streaming/Channels/102` หากกล้องต้องใช้ชื่อผู้ใช้ ให้ระบุ username แล้วโปรแกรมจะถาม password แบบซ่อนข้อความ:

```powershell
python box_counter.py --username admin
```

หรือเก็บข้อมูลไว้ใน environment variable สำหรับ Terminal ปัจจุบัน:

```powershell
$env:BOX_COUNTER_USERNAME = "admin"
$env:BOX_COUNTER_PASSWORD = "YOUR_PASSWORD"
python box_counter.py
```

โปรแกรมจะแปลงอักขระพิเศษใน username/password เป็น URL encoding ให้อัตโนมัติ

กล้องแต่ละยี่ห้ออาจต้องมี stream path เช่น `/Streaming/Channels/101` หรือ `/cam/realmonitor?...` ให้ใช้ RTSP URL ที่เปิดได้ใน VLC

ตัวอย่างการปรับค่า:

```powershell
python box_counter.py --conf 0.45 --line-ratio 0.80 --imgsz 640
```

ถ้าโมเดลมีหลายคลาสและต้องการนับเฉพาะคลาสกล่อง สมมติกล่องเป็น class ID 0:

```powershell
python box_counter.py --class-id 0
```

`--hysteresis 12` คือระยะกันสั่นเหนือและใต้เส้น หน่วยเป็นพิกเซล หากวัตถุสั่นใกล้เส้นแล้วนับผิดให้เพิ่มเป็น `20` หรือ `30`

โปรแกรมจะข้ามเฟรม H.264 ที่เสียเป็นครั้งคราว และจะ reconnect เมื่ออ่านพลาดต่อเนื่อง 10 เฟรม ปรับได้ด้วย `--reconnect-after 20`

โปรแกรมอ่าน RTSP ใน thread แยกและเก็บเฉพาะเฟรมล่าสุด จึงไม่สะสมภาพเก่าจนเกิด delay หาก YOLO ช้ากว่ากล้อง เฟรมบางส่วนจะถูกข้ามโดยตั้งใจเพื่อรักษาภาพแบบ real-time

หากมี NVIDIA GPU ให้รันด้วย CUDA และ FP16:

```powershell
python box_counter.py --username admin --device 0 --half
```

หากใช้ CPU และภาพยังช้า ให้ลดขนาด inference:

```powershell
python box_counter.py --username admin --device cpu --imgsz 416
```

## กล่องหายชั่วคราวก่อนข้ามเส้น

โปรแกรมใช้ `box_bytetrack.yaml` เป็นค่าเริ่มต้นและจำ tracking ID ที่หายไปได้นาน 90 เฟรมประมวลผล เมื่อกล่องกลับมาปรากฏอีกด้านของเส้นด้วย ID เดิม โปรแกรมจะนับการข้ามตามปกติ โดยไม่ต้องหน่วงวิดีโอ

หาก confidence ของกล่องตกชั่วคราวใกล้เส้น ให้ลองลด threshold:

```powershell
python box_counter.py --username admin --conf 0.20
```

หากต้องการจำวัตถุนานขึ้น แก้ค่า `track_buffer` ใน `box_bytetrack.yaml` เช่น `150` แต่ค่าที่สูงเกินไปอาจจับ ID ผิดเมื่อมีกล่องหลายใบอยู่ใกล้กัน
