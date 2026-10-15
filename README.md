# YOLO box area counter

โปรแกรม Windows สำหรับนับกล่องจากกล้อง RTSP ด้วยโมเดล Ultralytics YOLO ที่ train แล้ว (`best.pt`)

- จุดกึ่งกลางกล่องเคลื่อนจากนอกเข้า Area: `+1`
- จุดกึ่งกลางกล่องเคลื่อนจากในออก Area: `-1`
- ลากจุดสีเหลือง `P1-P4` เพื่อปรับ Area สี่เหลี่ยม
- โปรแกรมบันทึกตำแหน่ง `P1-P4` ลง `area_config.json` หลังปล่อยเมาส์ และโหลดกลับให้อัตโนมัติเมื่อเปิดครั้งถัดไป
- กด `R` เพื่อล้างตัวนับ, `A` คืน Area เริ่มต้น และ `Q` หรือ `Esc` เพื่อออก

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
python box_counter.py --conf 0.45 --imgsz 640
```

หากต้องการเก็บตำแหน่ง Area ไว้ในไฟล์อื่น ให้ระบุเส้นทางด้วย `--area-config`:

```powershell
python box_counter.py --area-config camera01_area.json
```

ปุ่ม `A` จะคืน Area เริ่มต้นและบันทึกค่าเริ่มต้นนั้นลงไฟล์ด้วย

## แสดงเลข Order จาก Dock API

โปรแกรมเรียก `https://pwdapi.fec-corp.com/api/asrs/dock-front?dock=DOCK03`
ทุก 10 วินาที ดึง `order_no` ที่ไม่เป็นค่าว่างจาก `lanes` ตัดเลขซ้ำ และแสดงบนวิดีโอ
ตัวนับและเลข Order จะแสดงในแถบสถานะเหนือภาพกล้อง จึงไม่บังวัตถุในวิดีโอ
ให้กำหนด Bearer Token ใน environment variable ก่อนรัน เพื่อไม่เก็บ Token ลงในโค้ด:

```powershell
$env:DOCK_API_TOKEN = "YOUR_BEARER_TOKEN"
.\.venv\Scripts\python.exe box_counter.py
```

เปลี่ยนช่วงเวลาเรียก API หรือ URL ได้ เช่น:

```powershell
.\.venv\Scripts\python.exe box_counter.py --api-refresh 30 --api-url "https://pwdapi.fec-corp.com/api/asrs/dock-front?dock=DOCK03"
```

ถ้าไม่กำหนด `DOCK_API_TOKEN` วิดีโอยังทำงานตามปกติ แต่จะแสดงสถานะ `NO TOKEN`

เมื่ออ่าน Order ได้ โปรแกรมจะยึด `order_no` ตัวแรกไว้ตลอดการรัน และเมื่อ `TOTAL`
เปลี่ยนจะส่งค่าล่าสุดไปที่ `POST /api/asrs/dock-front/box-counts` ในรูปแบบ:

```json
{
  "order_no": "6100235059",
  "boxes": "9999"
}
```

การ POST ทำงานใน thread แยก หากส่งไม่สำเร็จจะลองใหม่ทุก 5 วินาทีและเก็บเฉพาะ
ค่า `TOTAL` ล่าสุด สถานะ `POST:OK`, `POST:HTTP 401` หรือ `POST:ERROR` จะแสดงบนแถบด้านบน
สามารถเปลี่ยนปลายทางและเวลาลองใหม่ด้วย `--box-count-url` และ `--post-retry`

หาก API ตอบสำเร็จและ `order_no` ของทุก lane เป็น `null` ต่อเนื่องอย่างน้อย 10 วินาที
โปรแกรมจะล้าง `ENTER`, `EXIT`, `TOTAL` และปลด Order เดิม จากนั้นรอรับ Order ใหม่
กรณีเครือข่ายขัดข้อง, HTTP error หรือ Token ผิดจะไม่ทำให้ยอดถูก reset ปรับเวลาได้ด้วย:

```powershell
.\.venv\Scripts\python.exe box_counter.py --null-reset-after 10
```

ถ้าโมเดลมีหลายคลาสและต้องการนับเฉพาะคลาสกล่อง สมมติกล่องเป็น class ID 0:

```powershell
python box_counter.py --class-id 0
```

`--hysteresis 12` คือระยะกันสั่นรอบขอบ Area หน่วยเป็นพิกเซล หากจุดกึ่งกลางกล่องสั่นอยู่ใกล้ขอบแล้วนับซ้ำ ให้เพิ่มเป็น `20` หรือ `30`

โปรแกรมจะข้ามเฟรม H.264 ที่เสียเป็นครั้งคราว และจะ reconnect เมื่ออ่านพลาดต่อเนื่อง 10 เฟรม ปรับได้ด้วย `--reconnect-after 20`

โปรแกรมอ่าน RTSP ใน thread แยกและเก็บเฉพาะเฟรมล่าสุด จึงไม่สะสมภาพเก่าจนเกิด delay หาก YOLO ช้ากว่ากล้อง เฟรมบางส่วนจะถูกข้ามโดยตั้งใจเพื่อรักษาภาพแบบ real-time

หากมี NVIDIA GPU ให้รันด้วย CUDA และ FP16:

```powershell
python box_counter.py --username admin --device 0 --quantize 16
```

หากใช้ CPU และภาพยังช้า ให้ลดขนาด inference:

```powershell
python box_counter.py --username admin --device cpu --imgsz 416
```

## กล่องหายชั่วคราวก่อนเข้า Area

โปรแกรมใช้ `box_bytetrack.yaml` เป็นค่าเริ่มต้นและจำ tracking ID ที่หายไปได้นาน 90 เฟรมประมวลผล เมื่อกล่องกลับมาปรากฏใน Area ด้วย ID เดิม โปรแกรมยังตรวจการเปลี่ยนสถานะจากนอกเข้าในได้ โดยไม่ต้องหน่วงวิดีโอ

หาก confidence ของกล่องตกชั่วคราวใกล้เส้น ให้ลองลด threshold:

```powershell
python box_counter.py --username admin --conf 0.20
```

หากต้องการจำวัตถุนานขึ้น แก้ค่า `track_buffer` ใน `box_bytetrack.yaml` เช่น `150` แต่ค่าที่สูงเกินไปอาจจับ ID ผิดเมื่อมีกล่องหลายใบอยู่ใกล้กัน
