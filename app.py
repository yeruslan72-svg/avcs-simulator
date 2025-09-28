import streamlit as st
import numpy as np
import plotly.graph_objects as go
import plotly.subplots as sp
import time
import io
from PIL import Image

# ==================== PDF Проверка ====================
try:
    from reportlab.lib.pagesizes import letter
    from reportlab.pdfgen import canvas
    REPORTLAB_AVAILABLE = True
except ImportError:
    REPORTLAB_AVAILABLE = False
    st.warning("📄 PDF reports disabled - install reportlab: `pip install reportlab`")

# ==================== Настройка страницы ====================
st.set_page_config(page_title="AVCS DNA Multi-Channel Simulator v5.1", layout="wide")
st.title("🏭 AVCS DNA Multi-Channel Industrial Simulator v5.1")
st.markdown("**Operational Excellence, Delivered** – Multi-sensor vibration control with angular damper configuration")

# ==================== Конфигурация оборудования ====================
class EquipmentConfig:
    def __init__(self):
        self.sensors = {
            'Motor_End': {'position': 'Motor Side'},
            'Pump_End': {'position': 'Pump Side'}
        }
        self.dampers = {
            'Damper_1': {'position': 'Front-Left', 'angle': 45},
            'Damper_2': {'position': 'Front-Right', 'angle': 45},
            'Damper_3': {'position': 'Rear-Left', 'angle': 45},
            'Damper_4': {'position': 'Rear-Right', 'angle': 45}
        }
        self.sample_rates = [1000, 2500, 5000, 10000, 20000]

config = EquipmentConfig()

# ==================== Генерация сигналов ====================
def generate_vibration_signal(time_points, fault_type, severity, sensor_position):
    base_signal = np.sin(2 * np.pi * 50 * time_points) + 0.05*np.random.randn(len(time_points))
    if fault_type == "Bearing_Fault":
        base_signal += np.random.choice([0, severity*0.5], size=len(time_points), p=[0.99,0.01])
    elif fault_type == "Imbalance":
        base_signal *= 1 + 0.3*severity*np.sin(2*np.pi*50*time_points)
    elif fault_type == "Misalignment":
        base_signal += 0.2*severity*np.sin(2*np.pi*100*time_points)
    return base_signal

def calculate_damper_force(fault_detected, severity):
    if not fault_detected:
        return {d:500 for d in config.dampers}
    base = min(8000, severity*1600)
    return {d:base*np.sin(np.radians(config.dampers[d]['angle'])) for d in config.dampers}

def suppress_signal(signal, damper_forces):
    total_force = sum(damper_forces.values())
    suppression = np.exp(-0.25*total_force/8000)
    return signal*suppression

def calculate_features(signal):
    rms = np.sqrt(np.mean(signal**2))
    pkpk = np.ptp(signal)
    crest = np.max(np.abs(signal))/rms if rms>0 else 0
    return rms, pkpk, crest

# ==================== Интерфейс ====================
st.sidebar.subheader("🎛️ Equipment & Fault Configuration")
sensor_motor = st.sidebar.checkbox("Motor End Sensor", value=True)
sensor_pump = st.sidebar.checkbox("Pump End Sensor", value=True)
fault_type = st.sidebar.selectbox("Fault Type", ["Normal Operation", "Bearing_Fault", "Imbalance", "Misalignment"])
severity = st.sidebar.slider("Fault Severity", 1,5,1)
dampers_enabled = st.sidebar.checkbox("Enable Dampers", True)
sample_rate = st.sidebar.selectbox("Sample Rate (Hz)", config.sample_rates, index=3)
show_animation = st.sidebar.checkbox("Show Animation", True)
run_sim = st.sidebar.button("▶️ Run Simulation")

# ==================== Симуляция ====================
if run_sim:
    active_sensors = []
    if sensor_motor: active_sensors.append('Motor_End')
    if sensor_pump: active_sensors.append('Pump_End')
    if not active_sensors: st.error("Select at least one sensor"); st.stop()

    time_points = np.linspace(0,0.1,int(sample_rate*0.05))
    signals = {}
    suppressed = {}
    damper_forces_dict = {}
    features_dict = {}

    for s in active_sensors:
        sig = generate_vibration_signal(time_points, fault_type, severity, s)
        fault_detected = fault_type!="Normal Operation"
        damper_forces = calculate_damper_force(fault_detected, severity)
        signals[s] = sig
        suppressed[s] = suppress_signal(sig, damper_forces) if dampers_enabled and fault_detected else sig
        damper_forces_dict[s] = damper_forces
        features_dict[s] = calculate_features(signals[s])

    # ==================== Анимация ====================
    if show_animation and len(active_sensors)>0:
        anim_placeholder = st.empty()
        num_frames = 5
        for frame in range(num_frames):
            fig = sp.make_subplots(rows=len(active_sensors), cols=1,
                subplot_titles=[f"{s} - {config.sensors[s]['position']}" for s in active_sensors])
            for i,s in enumerate(active_sensors):
                ratio = (frame+1)/num_frames
                current_signal = signals[s]*ratio + suppressed[s]*(1-ratio)
                fig.add_trace(go.Scatter(x=time_points*1000, y=current_signal, mode='lines',
                                         name='Signal', line=dict(color='blue')), row=i+1, col=1)
            fig.update_layout(height=300*len(active_sensors), title_text=f"Frame {frame+1}/{num_frames}")
            anim_placeholder.plotly_chart(fig, use_container_width=True)
            time.sleep(0.3)
    else:
        fig = sp.make_subplots(rows=len(active_sensors), cols=1,
            subplot_titles=[f"{s} - {config.sensors[s]['position']}" for s in active_sensors])
        for i,s in enumerate(active_sensors):
            fig.add_trace(go.Scatter(x=time_points*1000, y=signals[s], mode='lines', name='Original', line=dict(color='red')), row=i+1,col=1)
            if dampers_enabled and fault_type!="Normal Operation":
                fig.add_trace(go.Scatter(x=time_points*1000, y=suppressed[s], mode='lines', name='Suppressed', line=dict(color='blue')), row=i+1,col=1)
        fig.update_layout(height=300*len(active_sensors), title_text="Multi-Channel Simulation")
        st.plotly_chart(fig, use_container_width=True)

    # ==================== Панель Damper Forces ====================
    st.subheader("🔧 Damper Forces Overview")
    damper_cols = st.columns(len(config.dampers))
    for i, damper_id in enumerate(config.dampers.keys()):
        with damper_cols[i]:
            force = 0
            if len(active_sensors) > 0:
                force = max(damper_forces_dict[s].get(damper_id,0) for s in active_sensors)
            color = "normal"
            if force < 2000: color="normal"
            elif force < 5000: color="inverse"
            else: color="critical"
            st.metric(f"{damper_id}", f"{force:.0f} N", delta_color=color)

    # ==================== Бизнес-метрики ====================
    st.subheader("📈 Business Impact")
    downtime_cost = st.number_input("Estimated hourly downtime cost ($)",1000,100000,10000,1000)
    prevented_hours = severity*8
    savings = downtime_cost*prevented_hours
    roi = savings/120000
    st.metric("💾 Downtime Prevented", f"{prevented_hours} hours")
    st.metric("💰 Estimated Savings", f"${savings:,.0f}")
    st.metric("📊 ROI Multiplier", f"{roi:.1f}x")

    # ==================== LinkedIn ====================
    st.subheader("📢 LinkedIn Post Preview")
    post_text = f"""
🚀 Multi-Channel Predictive Maintenance – AVCS DNA

Fault: {fault_type} | Severity: {severity}/5
Downtime Prevented: {prevented_hours}h
Savings: ${savings:,.0f} | ROI: {roi:.1f}x

#PredictiveMaintenance #Industry40 #AVCSDNA
"""
    st.text_area("Suggested text:", post_text, height=200)

