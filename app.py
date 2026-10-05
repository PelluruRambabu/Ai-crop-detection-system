import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import cv2
import numpy as np
import pandas as pd
import requests
import pickle
import os
import io
from datetime import datetime

# ==============================================================================
# 1. PAGE SETUP & CYBER-HUD THEME
# ==============================================================================
st.set_page_config(
    page_title="SMART AGRICULTURE SYSTEM | AI HUB",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom HUD Styling with dark navy background, neon accents, and monospace fonts
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Share+Tech+Mono&family=Orbitron:wght@600;800&family=Inter:wght@400;600&display=swap');

    html, body, [class*="css"] {
        background-color: #070b14 !important;
        color: #e2e8f0;
        font-family: 'Inter', sans-serif;
    }
    .brand-header {
        font-family: 'Orbitron', monospace;
        color: #00f5d4;
        font-size: 2.1rem;
        text-align: center;
        letter-spacing: 2px;
        text-shadow: 0 0 12px rgba(0, 245, 212, 0.45);
        margin-top: -10px;
        margin-bottom: 2px;
    }
    .brand-sub {
        font-family: 'Share Tech Mono', monospace;
        color: #00bbf9;
        font-size: 0.88rem;
        text-align: center;
        letter-spacing: 1.5px;
        margin-bottom: 20px;
    }
    .hud-card {
        background: #0b1325;
        border: 1px solid #1a2744;
        border-radius: 8px;
        padding: 14px;
        margin-bottom: 12px;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.6);
    }
    .hud-header {
        font-family: 'Share Tech Mono', monospace;
        font-size: 0.82rem;
        color: #00bbf9;
        text-transform: uppercase;
        letter-spacing: 1.2px;
        border-bottom: 1px solid #1a2744;
        padding-bottom: 5px;
        margin-bottom: 10px;
        display: flex;
        align-items: center;
        gap: 6px;
    }
    .metric-container {
        background: #060a12;
        border: 1px solid #162035;
        border-radius: 6px;
        padding: 8px 10px;
        text-align: center;
    }
    .metric-label {
        font-size: 0.74rem;
        color: #94a3b8;
        text-transform: uppercase;
        font-weight: 600;
    }
    .metric-val {
        font-family: 'Share Tech Mono', monospace;
        font-size: 1.4rem;
        font-weight: 700;
        color: #f8fafc;
        margin-top: 2px;
    }
    .metric-unit { font-size: 0.82rem; color: #00bbf9; }
    
    .badge {
        display: inline-block;
        padding: 3px 9px;
        border-radius: 4px;
        font-family: 'Share Tech Mono', monospace;
        font-size: 0.78rem;
        font-weight: 700;
        margin-top: 4px;
    }
    .badge-on { background: rgba(0, 245, 212, 0.15); color: #00f5d4; border: 1px solid #00f5d4; }
    .badge-off { background: rgba(239, 68, 68, 0.15); color: #ef4444; border: 1px solid #ef4444; }
    .badge-warn { background: rgba(245, 158, 11, 0.15); color: #f59e0b; border: 1px solid #f59e0b; }

    div.stButton > button:first-child {
        background: linear-gradient(90deg, #00f5d4 0%, #00bbf9 100%) !important;
        color: #030712 !important;
        font-family: 'Share Tech Mono', monospace !important;
        font-size: 0.95rem !important;
        font-weight: 800 !important;
        letter-spacing: 1px !important;
        border: none !important;
        border-radius: 6px !important;
        padding: 10px 20px !important;
        box-shadow: 0 0 12px rgba(0, 245, 212, 0.35) !important;
    }
    div.stDownloadButton > button:first-child {
        background: linear-gradient(90deg, #f72585 0%, #7209b7 100%) !important;
        color: #ffffff !important;
        font-family: 'Share Tech Mono', monospace !important;
        font-weight: 700 !important;
        letter-spacing: 1.2px !important;
        border: none !important;
        border-radius: 6px !important;
        box-shadow: 0 0 12px rgba(247, 37, 133, 0.35) !important;
    }
</style>
""", unsafe_allow_html=True)

# Session state initialization
if "scan_logs" not in st.session_state:
    st.session_state.scan_logs = []
if "active_leaf_image" not in st.session_state:
    st.session_state.active_leaf_image = None
if "image_source_label" not in st.session_state:
    st.session_state.image_source_label = "None"
if "last_diagnosis" not in st.session_state:
    st.session_state.last_diagnosis = None

# ==============================================================================
# 2. LOAD TRAINED MODEL (.PKL) - CPU / GPU COMPATIBLE
# ==============================================================================
MODEL_FILE = "crop_disease_model.pkl"

class CPU_Unpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == 'torch.storage' and name == '_load_from_bytes':
            return lambda b: torch.load(io.BytesIO(b), map_location='cpu')
        return super().find_class(module, name)

@st.cache_resource
def load_pytorch_model(pkl_path):
    if not os.path.exists(pkl_path):
        return None, None
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    with open(pkl_path, 'rb') as f:
        if torch.cuda.is_available():
            payload = pickle.load(f)
        else:
            payload = CPU_Unpickler(f).load()
    
    classes = payload['class_names']
    net = models.mobilenet_v2(weights=None)
    net.classifier[1] = nn.Linear(net.last_channel, len(classes))
    net.load_state_dict(payload['model_state_dict'])
    net.to(device)
    net.eval()
    return net, classes

model, class_names = load_pytorch_model(MODEL_FILE)

# Evaluation image transformation
eval_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# Treatment recommendation database
REMEDY_DATABASE = {
    # Tomato
    "tomato healthy": ("None", "Plant tissue is vigorous. Maintain standard fertigation and drip irrigation schedule."),
    "tomato leaf blight": ("High", "Apply copper oxychloride (2.5g/L) or mancozeb. Prune lower diseased foliage."),
    "tomato leaf curl": ("Critical", "Vector-borne TYLCV. Control whiteflies with neem oil or imidacloprid spray."),
    "tomato septoria leaf spot": ("Medium", "Apply chlorothalonil spray; ensure drip spacing to lower canopy humidity."),
    "tomato verticilium wilt": ("High", "Vascular fungus. Solarize soil beds; rotate crop with non-solanaceous species."),
    
    # Maize
    "maize healthy": ("None", "Optimal canopy health. Maintain nitrogen top-dressing schedule."),
    "maize fall armyworm": ("Critical", "Apply emamectin benzoate (0.4g/L) directly into leaf whorls early morning."),
    "maize grasshoper": ("Medium", "Deploy border barrier trenches and spray registered bio-pesticides."),
    "maize leaf beetle": ("Medium", "Administer pyrethroid foliar contact insecticide at primary spotting."),
    "maize leaf blight": ("High", "Apply azoxystrobin + propiconazole spray upon primary lesion detection."),
    "maize leaf spot": ("Medium", "Improve crop spacing for aeration; apply foliar triazole formulations."),
    "maize streak virus": ("High", "Eliminate Cicadulina leafhopper vectors using systemic neonicotinoid sprays."),
    
    # Rice
    "rice healthy": ("None", "Optimal paddy condition. Maintain 2-5 cm standing water layer."),
    "rice bacterial blight": ("Critical", "Drain stagnant field water; apply Streptocycline (100 ppm) with copper fungicide."),
    "rice blast": ("Critical", "Immediately spray Tricyclazole 75% WP @ 0.6g/L upon spindle lesion spotting."),
    "rice brown spot": ("High", "Correct soil potassium deficiency; apply carbendazim or mancozeb.")
}

def get_remedy(disease_name):
    clean_key = disease_name.lower().strip()
    for k, v in REMEDY_DATABASE.items():
        if k in clean_key:
            return v
    return ("Moderate", "Apply broad-spectrum organic fungicide and monitor foliar progression.")

# ==============================================================================
# 3. SIDEBAR CONFIGURATION
# ==============================================================================
st.sidebar.markdown("<div class='hud-header'>📡 SYSTEM COMMUNICATION</div>", unsafe_allow_html=True)
esp32_ip = st.sidebar.text_input("Main ESP32 IP Address", "192.168.1.50")
esp32_cam_ip = st.sidebar.text_input("ESP32-CAM IP Address", "192.168.1.60")

cam_capture_url = f"http://{esp32_cam_ip}/capture"
cam_stream_url = f"http://{esp32_cam_ip}:81/stream"

st.sidebar.markdown("---")
st.sidebar.markdown("<div class='hud-header'>🌿 TARGET CROP SELECTION</div>", unsafe_allow_html=True)
selected_crop = st.sidebar.radio(
    "Select Crop Species:",
    ["Tomato", "Maize (Corn)", "Rice / Paddy"]
)

# ==============================================================================
# 4. TELEMETRY POLLING ENGINE
# ==============================================================================
def poll_esp32_sensors(ip):
    try:
        res = requests.get(f"http://{ip}/data", timeout=1.0)
        if res.status_code == 200:
            return res.json(), True
    except Exception:
        pass
    # Fallback simulation values if node is unreachable
    return {
        "dht11_temp": 28.5,
        "dht11_humidity": 68.0,
        "dht22_temp": 27.2,
        "dht22_humidity": 56.4,
        "soil_moisture": 35,
        "motor_relay": "ON",
        "humidifier_relay": "ON",
        "pir_motion": "MOTION DETECTED",
        "buzzer": "ON"
    }, False

telemetry, is_online = poll_esp32_sensors(esp32_ip)

# ==============================================================================
# 5. DASHBOARD LAYOUT & LIVE TELEMETRY
# ==============================================================================
st.markdown("<div class='brand-header'>SMART AGRICULTURE SYSTEM</div>", unsafe_allow_html=True)
st.markdown(f"<div class='brand-sub'>NODE STATUS: {'[ ONLINE - ' + esp32_ip + ' ]' if is_online else '[ OFFLINE / SIMULATION MODE ]'} // SYSTEM CLOCK: {datetime.now().strftime('%H:%M:%S')}</div>", unsafe_allow_html=True)

col_irrigation, col_humidity = st.columns(2)

with col_irrigation:
    st.markdown("""
    <div class='hud-card'>
        <div class='hud-header'>💧 AUTOMATIC PLANT WATERING SYSTEM (DHT11 + SOIL + MOTOR + PIR)</div>
    </div>
    """, unsafe_allow_html=True)
    c1, c2, c3, c4 = st.columns(4)
    c1.markdown(f"<div class='metric-container'><div class='metric-label'>Canopy Temp</div><div class='metric-val'>{telemetry.get('dht11_temp', 0)} <span class='metric-unit'>°C</span></div></div>", unsafe_allow_html=True)
    c2.markdown(f"<div class='metric-container'><div class='metric-label'>Soil Moisture</div><div class='metric-val'>{telemetry.get('soil_moisture', 0)} <span class='metric-unit'>%</span></div></div>", unsafe_allow_html=True)
    
    motor_on = telemetry.get('motor_relay') == "ON"
    c3.markdown(f"<div class='metric-container'><div class='metric-label'>Water Pump</div><div class='badge {'badge-on' if motor_on else 'badge-off'}'>{telemetry.get('motor_relay', 'OFF')}</div></div>", unsafe_allow_html=True)
    
    has_motion = "DETECTED" in telemetry.get('pir_motion', '')
    c4.markdown(f"<div class='metric-container'><div class='metric-label'>PIR Motion</div><div class='badge {'badge-warn' if has_motion else 'badge-off'}'>{'MOTION' if has_motion else 'CLEAR'}</div></div>", unsafe_allow_html=True)

with col_humidity:
    st.markdown("""
    <div class='hud-card'>
        <div class='hud-header'>💨 SMART HUMIDITY CONTROL SYSTEM (DHT22 + HUMIDIFIER RELAY)</div>
    </div>
    """, unsafe_allow_html=True)
    c5, c6, c7 = st.columns(3)
    c5.markdown(f"<div class='metric-container'><div class='metric-label'>Chamber Temp</div><div class='metric-val'>{telemetry.get('dht22_temp', 0)} <span class='metric-unit'>°C</span></div></div>", unsafe_allow_html=True)
    c6.markdown(f"<div class='metric-container'><div class='metric-label'>Chamber Humid</div><div class='metric-val'>{telemetry.get('dht22_humidity', 0)} <span class='metric-unit'>%</span></div></div>", unsafe_allow_html=True)
    
    humid_on = telemetry.get('humidifier_relay') == "ON"
    c7.markdown(f"<div class='metric-container'><div class='metric-label'>Humidifier</div><div class='badge {'badge-on' if humid_on else 'badge-off'}'>{telemetry.get('humidifier_relay', 'OFF')}</div></div>", unsafe_allow_html=True)

st.markdown("<div style='height: 8px;'></div>", unsafe_allow_html=True)

# ==============================================================================
# 6. DUAL IMAGE INGESTION (REAL-TIME ESP32-CAM + MANUAL UPLOAD)
# ==============================================================================
left_cam_col, right_ai_col = st.columns([1.1, 0.9])

with left_cam_col:
    st.markdown("<div class='hud-card'><div class='hud-header'>📷 CROP IMAGE INGESTION HUB</div></div>", unsafe_allow_html=True)
    
    input_channel = st.radio(
        "Choose Ingestion Channel:",
        ["ESP32-CAM Real-Time Stream", "Manual File Upload", "Laptop / USB Camera"],
        horizontal=True
    )

    # Mode 1: ESP32-CAM Real-Time Live Feed & Instant Capture
    if input_channel == "ESP32-CAM Real-Time Stream":
        st.markdown(f"""
        <div style='text-align:center; background:#000; border: 1px solid #1a2744; border-radius:6px; overflow:hidden;'>
            <img src="{cam_stream_url}" width="100%" style="min-height:260px; max-height:340px; object-fit:contain;" 
                 onerror="this.onerror=null; this.src='https://via.placeholder.com/640x360/060a12/00f5d4?text=CONNECTING+TO+ESP32-CAM+STREAM...';" />
        </div>
        """, unsafe_allow_html=True)
        st.caption(f"Stream: `{cam_stream_url}` | Direct Snapshot: `{cam_capture_url}`")
        
        if st.button("📸 CAPTURE FRAME FROM ESP32-CAM", use_container_width=True):
            with st.spinner("Acquiring real-time frame from ESP32-CAM..."):
                img_captured = False
                
                try:
                    resp = requests.get(cam_capture_url, timeout=2.5)
                    if resp.status_code == 200:
                        st.session_state.active_leaf_image = Image.open(io.BytesIO(resp.content)).convert('RGB')
                        st.session_state.image_source_label = "ESP32-CAM Live Capture"
                        st.success("Captured real-time snapshot via ESP32-CAM `/capture`.")
                        img_captured = True
                except Exception:
                    pass

                if not img_captured:
                    try:
                        cap = cv2.VideoCapture(cam_stream_url)
                        ret, frame = cap.read()
                        if ret:
                            st.session_state.active_leaf_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                            st.session_state.image_source_label = "ESP32-CAM Stream Buffer"
                            st.success("Captured real-time snapshot via MJPEG stream buffer.")
                            img_captured = True
                        cap.release()
                    except Exception as e:
                        st.error(f"Failed to acquire stream frame: {e}")

    # Mode 2: Manual File Upload
    elif input_channel == "Manual File Upload":
        st.markdown("""
        <div style='font-family:"Share Tech Mono", monospace; font-size:0.85rem; color:#94a3b8; margin-bottom:8px;'>
            UPLOAD TEST LEAF IMAGE FROM LOCAL DISK:
        </div>
        """, unsafe_allow_html=True)
        
        uploaded_file = st.file_uploader(
            "Choose a Leaf Image", 
            type=["jpg", "jpeg", "png"],
            help="Upload an image from your computer to run pathology classification."
        )
        
        if uploaded_file is not None:
            st.session_state.active_leaf_image = Image.open(uploaded_file).convert('RGB')
            st.session_state.image_source_label = f"Uploaded File ({uploaded_file.name})"
            st.success(f"Loaded: `{uploaded_file.name}`")

    # Mode 3: Laptop Webcam
    else:
        cam_snap = st.camera_input("Take Snapshot with Built-in Camera")
        if cam_snap:
            st.session_state.active_leaf_image = Image.open(cam_snap).convert('RGB')
            st.session_state.image_source_label = "Laptop Camera Snapshot"

with right_ai_col:
    st.markdown("<div class='hud-card'><div class='hud-header'>🔬 ML IMAGE ANALYSIS & DIAGNOSTICS</div></div>", unsafe_allow_html=True)

    if st.session_state.active_leaf_image is not None:
        st.image(st.session_state.active_leaf_image, caption=f"Active Sample [{selected_crop}] — Source: {st.session_state.image_source_label}", width=260)
        
        if st.button("⚡ START AI PATHOLOGY SCAN", use_container_width=True):
            if model is None:
                st.error(f"Cannot find `{MODEL_FILE}` in project directory! Please place your trained .pkl file here.")
            else:
                with st.spinner("Processing deep convolutional inference..."):
                    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                    img_tensor = eval_transform(st.session_state.active_leaf_image).unsqueeze(0).to(device)
                    
                    with torch.no_grad():
                        output = model(img_tensor)
                        probabilities = torch.nn.functional.softmax(output[0], dim=0)
                        top_conf, top_idx = torch.max(probabilities, 0)
                        predicted_label = class_names[top_idx.item()]
                        confidence = top_conf.item() * 100

                    severity, remedy_text = get_remedy(predicted_label)
                    is_healthy = "healthy" in predicted_label.lower()
                    color_accent = "#00f5d4" if is_healthy else "#ef4444"

                    st.session_state.last_diagnosis = {
                        "label": predicted_label,
                        "confidence": confidence,
                        "severity": severity,
                        "remedy": remedy_text,
                        "color": color_accent,
                        "source": st.session_state.image_source_label
                    }

                    # Append scan to in-memory dataset table
                    st.session_state.scan_logs.insert(0, {
                        "Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        "Crop": selected_crop,
                        "Diagnosed_Pathology": predicted_label,
                        "Confidence": f"{confidence:.2f}%",
                        "Severity": severity,
                        "Source": st.session_state.image_source_label,
                        "Canopy_Temp": f"{telemetry.get('dht11_temp')} °C",
                        "Soil_Moisture": f"{telemetry.get('soil_moisture')}%",
                        "Chamber_Humid": f"{telemetry.get('dht22_humidity')}%",
                        "Pump_State": telemetry.get('motor_relay'),
                        "Humidifier_State": telemetry.get('humidifier_relay'),
                        "PIR_Alert": telemetry.get('pir_motion')
                    })
    else:
        st.markdown("""
        <div style='background: #03070f; border: 1px dashed #1a2744; border-radius: 6px; padding: 40px 10px; text-align: center; color: #64748b; font-family: "Share Tech Mono", monospace;'>
            AWAITING OPTICAL FRAME...<br>
            [ CAPTURE FROM ESP32-CAM OR UPLOAD A LEAF IMAGE TO INITIALIZE DIAGNOSTICS ]
        </div>
        """, unsafe_allow_html=True)

    # Render latest diagnosis card
    if st.session_state.last_diagnosis:
        res = st.session_state.last_diagnosis
        st.markdown(f"""
        <div class='hud-card' style='border-left: 4px solid {res["color"]}; margin-top: 10px;'>
            <div style='font-family: "Orbitron", monospace; font-size: 1.15rem; font-weight: 700; color: {res["color"]};'>
                DIAGNOSIS: {res['label'].upper()}
            </div>
            <div style='font-size: 0.9rem; color: #cbd5e1; margin-top: 5px;'>
                <b>Target Crop:</b> {selected_crop} | 
                <b>Severity:</b> <span style='color: {res["color"]}; font-weight: bold;'>{res['severity']}</span> | 
                <b>Confidence:</b> <span style='color: #00bbf9; font-weight: bold;'>{res['confidence']:.2f}%</span>
            </div>
            <div style='font-size: 0.8rem; color: #64748b; margin-top: 3px;'>
                <b>Sample Origin:</b> {res.get('source', 'Unknown')}
            </div>
            <div style='margin-top: 8px; font-size: 0.84rem; color: #94a3b8; background: #060a12; padding: 8px; border-radius: 4px;'>
                <b>Agronomic Recommendation:</b><br>{res['remedy']}
            </div>
        </div>
        """, unsafe_allow_html=True)

# ==============================================================================
# 7. SCAN LOGS & CSV EXPORT (ML DATASET FRAMEWORK)
# ==============================================================================
st.markdown("<div style='height: 12px;'></div>", unsafe_allow_html=True)
st.markdown("<div class='hud-card'><div class='hud-header'>📋 SCAN LOGS (ML DATASET FRAMEWORK)</div></div>", unsafe_allow_html=True)

if len(st.session_state.scan_logs) > 0:
    df_logs = pd.DataFrame(st.session_state.scan_logs)
    st.dataframe(df_logs, use_container_width=True)

    csv_data = df_logs.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="📥 DOWNLOAD CSV (ML DATA)",
        data=csv_data,
        file_name=f"smart_agri_dataset_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
        mime="text/csv",
        use_container_width=True
    )
else:
    st.markdown("""
    <div style='background: #02050b; border: 1px solid #142036; border-radius: 6px; padding: 12px; font-family: "Share Tech Mono", monospace; font-size: 0.8rem; color: #00f5d4;'>
        Timestamp, Crop, Pathology, Confidence, Severity, Source, Soil_Moist, Chamber_Humid, Pump_State, Humidifier_State<br>
        [SYSTEM LOG READY] Awaiting first camera capture or uploaded scan sequence...
    </div>
    """, unsafe_allow_html=True)