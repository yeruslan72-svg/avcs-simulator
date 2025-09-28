import streamlit as st
import numpy as np
import plotly.graph_objects as go
import plotly.subplots as sp
import time
import io
from PIL import Image

# ==================== ПРОВЕРКА reportlab ====================
try:
    from reportlab.lib.pagesizes import letter
    from reportlab.pdfgen import canvas
    REPORTLAB_AVAILABLE = True
except ImportError:
    REPORTLAB_AVAILABLE = False
    st.warning("📄 PDF reports disabled - install reportlab: `pip install reportlab`")

# ==================== СТРАНИЦА ====================
st.set_page_config(page_title="AVCS DNA Multi-Channel Simulator", layout="wide")
st.title("🏭 AVCS DNA Multi-Channel Industrial Simulator")
st.markdown("**Operational Excellence, Delivered** - Multi-sensor vibration control with angular damper configuration")

# ==================== КОНФИГ ====================
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
        self.accelerometer_types = {
            'ICP_603C01': {},
            'PCB_352C23': {},
            'IMI_608A11': {}
        }

config = EquipmentConfig()

# ==================== ФУНКЦИИ ====================
def generate_vibration_signal(time_points, fault_type, severity, sensor_position):
    base_freq = 50
    base_signal = np.sin(2 * np.pi * base_freq * time_points) + 0.05*np.random.randn(len(time_points))
    
    if sensor_position == 'Motor_End':
        base_signal += 0.1*np.sin(2*np.pi*200*time_points)
    else:
        base_signal += 0.15*np.sin(2*np.pi*25*time_points)
    
    anomalies = np.zeros_like(time_points)
    
    if "Bearing" in fault_type:
        impulse_prob = 0.008 * severity
        anomalies = (np.random.rand(len(time_points)) < impulse_prob).astype(float) * severity * 0.6
        base_signal += anomalies
    elif fault_type == "Imbalance":
        base_signal *= 1 + 0.4*severity*np.sin(2*np.pi*base_freq*time_points)
    elif fault_type == "Misalignment":
        base_signal += 0.5*severity*np.sin(2*np.pi*2*base_freq*time_points + np.pi/4)
        base_signal += 0.3*severity*np.sin(2*np.pi*3*base_freq*time_points + np.pi/6)
    
    return base_signal, anomalies

def calculate_damper_forces(dampers, fault_detected, severity, sensor_position):
    forces = {}
    base_force = 500 if not fault_detected else min(8000, severity*1600)
    for d, cfg in dampers.items():
        angle_factor = np.sin(np.radians(cfg['angle']))
        position_factor = 1.2 if (('Front' in cfg['position'] and sensor_position=='Motor_End') or
                                  ('Rear' in cfg['position'] and sensor_position=='Pump_End')) else 1.0
        forces[d] = base_force * angle_factor * position_factor
    return forces

def apply_damper_suppression(signal, damper_forces):
    total_force = sum(damper_forces.values())
    factor = np.exp(-0.25*total_force/8000)
    suppressed = signal * factor
    return suppressed

def calculate_features(signals):
    features = {}
    for s, sig in signals.items():
        rms = np.sqrt(np.mean(sig**2))
        pkpk = np.ptp(sig)
        crest = np.max(np.abs(sig))/rms if rms>0 else 0
        features[s] = {'rms': rms, 'pkpk': pkpk, 'crest': crest}
    return features

def show_business_impact(severity):
    st.subheader("📈 Business Impact Estimation")
    col1, col2 = st.columns(2)
    with col1:
        cost = st.number_input("Hourly downtime cost ($)", 1000, 50000, 10000, step=1000)
    with col2:
        prevented = severity * 8
        savings = cost * prevented
        st.metric("💾 Downtime prevented", f"{prevented} hours")
        st.metric("💰 Estimated savings", f"${savings:,.0f}")
    return prevented, savings

def generate_text_report(features, fault_type, severity, prevented, savings, roi):
    report = f"""
AVCS DNA SIMULATION REPORT
Generated on: {time.strftime('%Y-%m-%d %H:%M:%S')}

FAULT SIMULATION:
- Type: {fault_type}
- Severity: {severity}/5
- Downtime Prevented: {prevented} hours
- Estimated Savings: ${savings:,.0f}
- ROI: {roi:.1f}x

TECHNICAL PARAMETERS:
"""
    for s,f in features.items():
        report += f"- {s}: RMS={f['rms']:.4f}, PkPk={f['pkpk']:.3f}, Crest={f['crest']:.2f}\n"
    return report

# ==================== ИНТЕРФЕЙС ====================
st.sidebar.subheader("🎛️ Equipment & Fault Configuration")
sensor_motor = st.sidebar.checkbox("Motor End Sensor", True)
sensor_pump = st.sidebar.checkbox("Pump End Sensor", True)
fault_type = st.sidebar.selectbox("Fault Type", ["Normal Operation","Bearing_Fault_Mild","Bearing_Fault_Severe","Imbalance","Misalignment"])
severity = st.sidebar.slider("Fault Severity", 1, 5, 1)
sample_rate = st.sidebar.selectbox("Sample Rate", config.sample_rates, index=3)
accel_type = st.sidebar.selectbox("Accelerometer Type", list(config.accelerometer_types.keys()))
dampers_enabled = st.sidebar.checkbox("Enable Dampers", True)
anim_speed = st.sidebar.slider("Animation Speed", 1, 5, 3)

run_sim = st.button("▶️ Start Simulation")

if run_sim:
    active_sensors = []
    if sensor_motor: active_sensors.append('Motor_End')
    if sensor_pump: active_sensors.append('Pump_End')
    if not active_sensors:
        st.error("Select at least one sensor")
        st.stop()
    
    time_points = np.linspace(0,0.1,int(sample_rate*0.1))
    final_signals = {}
    final_suppressed = {}
    final_damper_forces = {}
    
    anim_placeholder = st.empty()
    num_frames = 10
    progress_bar = st.progress(0)
    for frame in range(num_frames):
        progress_bar.progress((frame+1)/num_frames)
        signals = {}
        suppressed = {}
        forces_hist = {}
        for s in active_sensors:
            sig, anomalies = generate_vibration_signal(time_points, fault_type, severity, s)
            signals[s] = sig
            fault_detected = fault_type != "Normal Operation"
            forces = calculate_damper_forces(config.dampers, fault_detected, severity, s)
            forces_hist[s] = forces
            if dampers_enabled and fault_detected:
                suppressed[s] = apply_damper_suppression(sig, forces)
            else:
                suppressed[s] = sig
        final_signals = signals
        final_suppressed = suppressed
        final_damper_forces = forces_hist
        
        fig = sp.make_subplots(rows=len(active_sensors), cols=1, subplot_titles=active_sensors)
        for i, s in enumerate(active_sensors):
            fig.add_trace(go.Scatter(x=time_points*1000, y=final_signals[s], mode='lines', line=dict(color='red')), row=i+1, col=1)
            if dampers_enabled:
                fig.add_trace(go.Scatter(x=time_points*1000, y=final_suppressed[s], mode='lines', line=dict(color='green')), row=i+1, col=1)
        fig.update_layout(height=300*len(active_sensors), title_text=f"Multi-Channel Simulation - Frame {frame+1}/{num_frames}")
        anim_placeholder.plotly_chart(fig, use_container_width=True)
        time.sleep(0.5/anim_speed)
    progress_bar.empty()
    
    # Считаем метрики и сохраняем в session_state
    features = calculate_features(final_signals)
    prevented, savings = show_business_impact(severity)
    roi = savings/120000
    
    st.session_state['features'] = features
    st.session_state['fault_type'] = fault_type
    st.session_state['severity'] = severity
    st.session_state['prevented'] = prevented
    st.session_state['savings'] = savings
    st.session_state['roi'] = roi
    
    st.subheader("📢 Professional Report")
    if REPORTLAB_AVAILABLE:
        if st.button("📄 Generate PDF Report"):
            try:
                c = canvas.Canvas("report.pdf", pagesize=letter)
                y = 750
                c.drawString(50, y, "AVCS DNA Simulation Report")
                y -= 20
                for s,f in features.items():
                    c.drawString(50, y, f"{s}: RMS={f['rms']:.4f} PkPk={f['pkpk']:.3f} Crest={f['crest']:.2f}")
                    y -= 15
                c.save()
                with open("report.pdf","rb") as f:
                    st.download_button("📥 Download PDF", f, file_name="avcs_simulation_report.pdf")
            except Exception as e:
                st.error(f"PDF generation failed: {e}")
    else:
        if st.button("📄 Generate Text Report"):
            text_report = generate_text_report(st.session_state['features'], st.session_state['fault_type'], st.session_state['severity'], st.session_state['prevented'], st.session_state['savings'], st.session_state['roi'])
            st.text_area("Professional Report", text_report, height=300)

st.markdown("---")
st.markdown("**Operational Excellence, Delivered** | Multi-Channel AVCS DNA Simulator v5.3")
