# Optimization Flooding and Survive AI 🚩

ระบบการคาดเดาความรุนเเรงของน้ำท่วม เเละคำนวณเส้นทางการอพยพที่ปลอดภัยที่สุด
โดยการ optimization

🧐 จัดทำโดย : นนอ.อริญชย์ หุนตระนี เเละ นนอ.อภิรักษ์ สาจันทร์

## How to use 🧑‍💻

### Step 1: Start the Server 🌐

Open a terminal with the conda environment and run:

```bash
conda activate geoai
cd d:\geoai_train\geoai_train
python Server.py
```

Then open your browser and navigate to: <http://localhost:5000>

### Step 2: Start Flood Detection (in a new terminal)

Open a **new terminal** (do not close the first one) and run:

```bash
cd d:\geoai_train\geoai_train\Flood-detection
python main.py  

one click run : run start_rescuopt.bat this file is auto cmd file to start the flood detection process

optimize_test.py : this file is for testing the optimization process, you can run it to see how the optimization works
```
