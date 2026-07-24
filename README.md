# 🌊 RescuOpt AI — ระบบตรวจจับน้ำท่วมและวางแผนช่วยเหลือ

RescuOpt เป็นระบบที่ใช้ YOLO ตรวจจับความรุนแรงของน้ำท่วมจากภาพ/วิดีโอ แล้วส่งข้อมูลไปยังแดชบอร์ดบนเว็บเพื่อคำนวณเส้นทางอพยพและจุดปลอดภัยให้ผู้ประสบภัยแบบเรียลไทม์
โดย นนอ.อริญชย์ หุนตระนี เเละ นนอ.อภิรักษ์ สาจันทร์

ระบบประกอบด้วย 2 ส่วนที่ทำงานพร้อมกัน:

| ส่วน | ไฟล์ | หน้าที่ |
|---|---|---|
| 🖥️ Server (Dashboard) | `Server.py` | เว็บเซิร์ฟเวอร์ Flask รับข้อมูล hazard/survivor แล้วแสดงผลบนแผนที่ (`dashboard/disaster_nav.html`) |
| 🤖 Desktop App | `Flood-detection/main.py` | โปรแกรม Tkinter สำหรับอัปโหลดภาพ/วิดีโอ แล้วให้ YOLO วิเคราะห์ความรุนแรงน้ำท่วม |

---

## 📋 สิ่งที่ต้องมีก่อนเริ่ม

1. **Windows 10/11**
2. **Miniconda หรือ Anaconda** ติดตั้งแล้ว → ดาวน์โหลดที่ [conda.io](https://www.anaconda.com/download)
3. **Git** ติดตั้งแล้ว → ดาวน์โหลดที่ [git-scm.com](https://git-scm.com/downloads)
4. เนื้อที่ว่างในเครื่องอย่างน้อย **5 GB** (สำหรับ PyTorch และโมเดล YOLO)

ทดสอบว่าติดตั้งครบไหม เปิด **Anaconda Prompt** แล้วพิมพ์:
```cmd
conda --version
git --version
```
ถ้าขึ้นเลขเวอร์ชันทั้งคู่ แปลว่าพร้อมแล้ว

---

## 🚀 ขั้นตอนการติดตั้ง (ทำตามลำดับ ห้ามข้าม)

### ขั้นที่ 1: Clone โปรเจกต์

⚠️ **สำคัญมาก:** ตั้งชื่อโฟลเดอร์ปลายทางเอง อย่าปล่อยให้ Git ตั้งชื่ออัตโนมัติ เพื่อป้องกันปัญหาโฟลเดอร์ซ้อนกัน (ปัญหาที่พบบ่อยที่สุด — ดูหัวข้อแก้ปัญหาด้านล่าง)

```cmd
cd C:\Users\<ชื่อผู้ใช้ของคุณ>\Downloads
git clone https://github.com/apiraSJ/Flooding_DASHBOARD-.git RescuOpt
cd RescuOpt
```

### ขั้นที่ 2: สร้าง Conda Environment แยกต่างหาก

```cmd
conda create -n geoai python=3.10 -y
conda activate geoai
```

> ✅ หลังรันคำสั่ง `activate` แล้ว ต้องเห็นคำว่า `(geoai)` ขึ้นนำหน้าบรรทัดคำสั่งเสมอ ก่อนรันคำสั่ง `pip install` ใดๆ ต่อไปนี้ — ถ้าไม่เห็น แปลว่า package จะไปติดตั้งผิดที่

### ขั้นที่ 3: ติดตั้ง Dependency ของ Server (Flask)

```cmd
pip install flask flask-cors
```

### ขั้นที่ 4: แก้ปัญหา setuptools/pkg_resources ล่วงหน้า (กันไว้ก่อน)

Conda เวอร์ชันใหม่มักจะติดตั้ง `setuptools` เวอร์ชันที่ใหม่เกินไปมาให้ ซึ่งจะทำให้ YOLO ใช้งานไม่ได้ ให้ล็อกเวอร์ชันไว้ก่อนเลย **ก่อน**ติดตั้ง dependency ตัวอื่น (ไม่งั้นตอนขั้นถัดไป pip อาจ build `shapely` จากซอร์สแล้วดึง setuptools เวอร์ชันใหม่มาเองจนเจอ error `pkg_resources` ไปก่อนที่จะถึงขั้นแก้):

```cmd
pip install "setuptools==80.10.2"
```

### ขั้นที่ 5: ติดตั้ง Dependency ของระบบตรวจจับ (YOLO)

```cmd
pip install -r Flood-detection\requirements.txt
```

⏳ ขั้นตอนนี้ใช้เวลานาน (5-15 นาทีขึ้นกับความเร็วเน็ต) เพราะต้องโหลด PyTorch ที่มีขนาดหลายร้อย MB **ห้ามปิดหน้าต่างจนกว่าจะขึ้น prompt กลับมาแบบไม่มีบรรทัดสีแดง**

### ขั้นที่ 6: รันโปรแกรม

```cmd
start_rescuopt.bat
```

จะมีหน้าต่างเปิดขึ้น 2 บาน:
- **RescuOpt Server** — เซิร์ฟเวอร์ Flask (อย่าปิดหน้าต่างนี้)
- **RescuOpt Desktop App** — โปรแกรมตรวจจับน้ำท่วม (Tkinter)

เปิดเบราว์เซอร์ไปที่ `http://127.0.0.1:5000/dashboard/disaster_nav.html` เพื่อดูแดชบอร์ด

---

## 🔧 คู่มือแก้ปัญหา (Troubleshooting)

### ❌ ปัญหา 1: `can't open file '...\Flooding_DASHBOARD--main\Flooding_DASHBOARD--main\...'`

**อาการ:** path ในข้อความ error มีชื่อโฟลเดอร์ซ้ำกัน 2 ชั้น

**สาเหตุ:** ดาวน์โหลด/clone repo ซ้ำเข้าไปในโฟลเดอร์ที่มีชื่อเดียวกันอยู่แล้ว

**วิธีแก้:** ลบโฟลเดอร์เก่าทิ้งแล้ว clone ใหม่โดยตั้งชื่อโฟลเดอร์ปลายทางเอง (ดูขั้นที่ 1 ด้านบน) — **ปิดโปรแกรมทุกตัวที่เปิดโฟลเดอร์นั้นอยู่ก่อน** (VSCode, terminal) แล้วค่อยลบ:

```cmd
rmdir /s /q C:\Users\<ชื่อผู้ใช้>\Downloads\Flooding_DASHBOARD--main
```

---

### ❌ ปัญหา 2: `ModuleNotFoundError: No module named 'flask'`

**สาเหตุ:** ยังไม่ได้ติดตั้ง Flask หรือ Flask-CORS

**วิธีแก้:**
```cmd
conda activate geoai
pip install flask flask-cors
```

---

### ❌ ปัญหา 3: `ModuleNotFoundError: No module named 'PIL'`

**สาเหตุ:** ยังไม่ได้ติดตั้ง dependency ของฝั่ง YOLO/Tkinter (Pillow อยู่ใน `requirements.txt` แล้ว แต่ยังไม่เคยรันติดตั้ง)

**วิธีแก้:**
```cmd
conda activate geoai
pip install -r Flood-detection\requirements.txt
```

---

### ❌ ปัญหา 4: `Analysis error: No module named 'pkg_resources'`

**สาเหตุ:** `setuptools` เวอร์ชันใหม่เกินไป (81 ขึ้นไป) ตัดโมดูล `pkg_resources` ออก แต่ `ultralytics` (YOLO) ยังต้องใช้อยู่

**วิธีแก้:**
```cmd
conda activate geoai
pip install "setuptools==80.10.2"
```

ตรวจสอบว่าแก้สำเร็จจริงด้วยคำสั่งนี้:
```cmd
python -c "import pkg_resources; print('OK')"
```
ถ้าขึ้น `OK` (มี warning สีเหลืองก็ไม่เป็นไร) แปลว่าใช้ได้แล้ว

---

### ❌ ปัญหา 5: แก้ปัญหา 4 แล้วแต่ยัง error ซ้ำอยู่

**สาเหตุที่พบบ่อยที่สุด:** รันคำสั่ง `pip install` โดย**ไม่ได้ activate environment `geoai`** ก่อน ทำให้ package ไปติดตั้งผิดที่ (ไปอยู่ที่ Python เครื่องหลักแทนที่จะอยู่ใน environment ของโปรเจกต์)

**วิธีเช็ค:** ดูว่าหน้าต่าง Command Prompt มีคำว่า `(geoai)` ขึ้นนำหน้าบรรทัดคำสั่งหรือไม่

```
(geoai) C:\Users\apira\Downloads\RescuOpt>     ← ถูกต้อง ✅
C:\Users\apira\Downloads\RescuOpt>             ← ผิด ต้อง activate ก่อน ❌
```

ถ้าไม่ขึ้น ให้พิมพ์คำสั่งนี้ก่อนทุกครั้งที่จะ `pip install`:
```cmd
conda activate geoai
```

ยืนยันว่าติดตั้งถูกที่ด้วยคำสั่งนี้ (Location ต้องชี้ไปที่โฟลเดอร์ `envs\geoai`):
```cmd
pip show setuptools
```
ผลลัพธ์ที่ถูกต้องต้องมีบรรทัดประมาณนี้:
```
Location: C:\Users\<ชื่อผู้ใช้>\miniconda3\envs\geoai\Lib\site-packages
```
ถ้า Location ไม่มีคำว่า `envs\geoai` แปลว่าลงผิด environment ให้ activate ใหม่แล้วติดตั้งซ้ำ

---

### ❌ ปัญหา 6: `The system cannot find the path specified.`

**สาเหตุ:** มักเกิดจาก path ซ้อนกันเหมือนปัญหา 1 หรือชื่อโฟลเดอร์ผิด

**วิธีแก้:** ตรวจสอบว่าอยู่ในโฟลเดอร์ `RescuOpt` (ไม่ใช่โฟลเดอร์ย่อยผิดที่) ด้วยคำสั่ง:
```cmd
cd
dir
```
ต้องเห็นไฟล์ `Server.py`, `start_rescuopt.bat`, และโฟลเดอร์ `Flood-detection` อยู่ในระดับเดียวกัน

---

## 🩹 วิธีล้างและเริ่มต้นใหม่ทั้งหมด (Nuclear Option)

ถ้าลองแก้ทุกอย่างแล้วยังไม่หาย ให้ล้างทุกอย่างแล้วเริ่มใหม่:

```cmd
:: 1. ลบโฟลเดอร์โปรเจกต์
cd C:\Users\<ชื่อผู้ใช้>\Downloads
rmdir /s /q RescuOpt

:: 2. ลบ conda environment เดิม
conda deactivate
conda env remove -n geoai

:: 3. clone และติดตั้งใหม่ทั้งหมด (ทำตามขั้นที่ 1-6 ด้านบน)
```

---

## 📁 โครงสร้างโปรเจกต์

```
RescuOpt/
├── Server.py                  ← เซิร์ฟเวอร์ Flask (รันจาก root)
├── start_rescuopt.bat          ← ตัวเปิดโปรแกรมทั้งหมด
├── dashboard/
│   └── disaster_nav.html       ← หน้าเว็บแดชบอร์ด
└── Flood-detection/
    ├── main.py                 ← โปรแกรม Tkinter (รันจากในโฟลเดอร์นี้)
    ├── yolo.py                 ← โมดูลตรวจจับ YOLO
    ├── best.pt                 ← โมเดล YOLO ที่เทรนแล้ว
    ├── requirements.txt        ← dependency ของฝั่งตรวจจับ
    └── media/                  ← ไฟล์ตัวอย่างภาพ/วิดีโอสำหรับทดสอบ อันนี้เอาภาพน้ำท่วมอื่นมาใส่ได้นะครับ
```

---

การแยก environment (`geoai`) ออกจาก Python หลักของเครื่อง (`base`) ช่วยป้องกันไม่ให้ library เวอร์ชันของโปรเจกต์นี้ไปชนกับโปรเจกต์อื่นที่อาจต้องใช้เวอร์ชันต่างกัน — เหมือนแยกกล่องเครื่องมือของแต่ละงานไม่ให้ปนกัน ถ้าโปรเจกต์นี้พังก็แค่ลบ environment `geoai` ทิ้งแล้วสร้างใหม่ โดยไม่กระทบ Python ส่วนอื่นของเครื่อง

---
