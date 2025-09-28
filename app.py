# ==================== IMPORTS ====================
import streamlit as st
import numpy as np
import plotly.graph_objects as go
import plotly.subplots as sp
import time
import io
from PIL import Image

# ==================== PDF CHECK ====================
try:
    from reportlab.lib.pagesizes import letter
    from reportlab.pdfgen import canvas
    REPORTLAB_AVAILABLE = True
except ImportError:
    REPORTLAB_AVAILABLE = False
    st.warning("📄 PDF reports disabled - install reportlab: `pip install reportlab`")

# ==================== CONFIG ====================
st.set_page_config(page_title="AVCS DNA Multi-Channel Simulator", layout="wide")
st.title("🏭 AVCS DNA Multi-Channel Industrial Simulator")
st.markdown("**Operational Excellence, Delivered** - Multi-sensor vibration control with angular damper configuration")

class EquipmentConfig:
    def __init__(self):
        self.sensors = {
            'Motor_End': {'position': 'Motor Side', 'sensitivity': 100},
            'Pump_End': {'position': 'Pump Side', 'sensitivity': 100}
        }
        self.dampers = {
            'Damper_1': {'position': 'Front-Left', 'angle': 45, 'force_range': (0, 8000)},
            'Damper_2': {'position': 'Front-Right', 'angle': 45, 'force_range': (0, 8000)},
            'Damper_3': {'position': 'Rear-Left', 'angle': 45, 'force_range': (0, 8000)},
            'Damper_4': {'position': 'Rear-Right', 'angle': 45, 'force_range': (0, 8000)}
        }
        self.sample_rates = [1000, 2500, 5000, 10000, 20000]

config = EquipmentConfig()

# ==================== FUNCTIONS ====================
def generate_vibration_signal(time_points, fault_type, severity, sensor_position, base_freq=50):
    base_signal = np.sin(2*np.pi*base_freq*time_points)
    base_signal += 0.05*np.random.randn(len(time_points))
    if sensor_position == 'Motor_End':
        base_signal += 0.1*np.sin(2*np.pi*200*time_points)
    else:
        base_signal += 0.15*np.sin(2*np.pi*25*time_points)
    if fault_type=="Normal Operation":
        return base_signal
    elif "Bearing_Fault" in fault_type:
        impulses = (np.random.rand(len(time_points)) < 0.008*severity).astype(float)*severity*0.6
        return base_signal + impulses
    elif fault_type=="Imbalance":
        return base_signal*(1+0.4*severity*np.sin(2*np.pi*base_freq*time_points))
    elif fault_type=="Misalignment":
        harmonic_2x = 0.5*severity*np.sin(2*np.pi*2*base_freq*time_points + np.pi/4)
        harmonic_3x = 0.3*severity*np.sin(2*np.pi*3*base_freq*time_points + np.pi/6)
        return base_signal + harmonic_2x + harmonic_3x
    return base_signal

def calculate_damper_force(fault_detected, severity):
    base_force = min(8000, severity*1600)
    return {d: base_force for d in config.dampers.keys()}

def suppress_signal(signal, damper_forces):
    total_force = sum(damper_forces.values())
    factor = np.exp(-0.25*total_force/8000)
    suppressed = signal*factor
    return suppressed

def calculate_features(signal):
    rms = np.sqrt(np.mean(signal**2))
    pkpk = np.ptp(signal)
    crest = np.max(np.abs(signal))/rms if rms>0 else 0
    return {'rms': rms, 'pkpk': pkpk, 'crest': crest}

def signal_colors(signal):
    # простая анимация цветов для аномалий
    return 'red' if np.max(np.abs(signal))>0.8 else 'green'

# ==================== SIDEBAR ====================
st.sidebar.subheader("🎛️ Equipment & Fault Configuration")
sensor_motor = st.sidebar.checkbox("Motor End Sensor", True)
sensor_pump = st.sidebar.checkbox("Pump End Sensor", True)
fault_type = st.sidebar.selectbox("Fault Type", ["Normal Operation", "Bearing_Fault", "Imbalance", "Misalignment"])
severity = st.sidebar.slider("Fault Severity", 1, 5, 1)
dampers_enabled = st.sidebar.checkbox("Enable Dampers", True)
sample_rate = st.sidebar.selectbox("Sample Rate (Hz)", config.sample_rates, index=3)
show_animation = st.sidebar.checkbox("Show Animation", True)
animation_speed = st.sidebar.slider("Animation Speed", 0.1, 5.0, 1.0, step=0.1)
run_sim = st.sidebar.button("▶️ Run Simulation")

# ==================== SIMULATION ====================
if run_sim:
    active_sensors = []
    if sensor_motor: active_sensors.append('Motor_End')
    if sensor_pump: active_sensors.append('Pump_End')
    if not active_sensors: st.error("Select at least one sensor"); st.stop()

    time_points = np.linspace(0,0.1,int(sample_rate*0.05))
    signals, suppressed, damper_forces_dict, features_dict = {}, {}, {}, {}

    num_frames = 10
    animation_placeholder = st.empty()
    progress_bar = st.progress(0)

    for frame in range(num_frames):
        for s in active_sensors:
            sig = generate_vibration_signal(time_points, fault_type, severity, s)
            fault_detected = fault_type != "Normal Operation"
            damper_forces = calculate_damper_force(fault_detected, severity)
            signals[s] = sig
            suppressed[s] = suppress_signal(sig, damper_forces) if dampers_enabled and fault_detected else sig
            damper_forces_dict[s] = damper_forces
            features_dict[s] = calculate_features(sig)

        fig = sp.make_subplots(rows=len(active_sensors), cols=1,
                               subplot_titles=[f"{s} - {config.sensors[s]['position']}" for s in active_sensors])
        for i,s in enumerate(active_sensors):
            fig.add_trace(go.Scatter(
                x=time_points*1000,
                y=signals[s],
                mode='lines',
                line=dict(color='blue', width=2),
                name='Original'
            ), row=i+1, col=1)
            if dampers_enabled and fault_type!="Normal Operation":
                fig.add_trace(go.Scatter(
                    x=time_points*1000,
                    y=suppressed[s],
                    mode='lines',
                    line=dict(color='green', width=2),
                    name='Suppressed'
                ), row=i+1, col=1)
            fig.add_trace(go.Scatter(
                x=time_points*1000,
                y=signals[s],
                mode='lines',
                line=dict(color=signal_colors(signals[s]), width=3),
                opacity=0.5,
                name='Anomaly Overlay'
            ), row=i+1, col=1)

        fig.update_layout(height=300*len(active_sensors),
                          title_text=f"Multi-Channel Simulation - Frame {frame+1}/{num_frames}")
        animation_placeholder.plotly_chart(fig, use_container_width=True)
        progress_bar.progress((frame+1)/num_frames)
        time.sleep(0.5/animation_speed)

    progress_bar.empty()
    st.success("✅ Simulation completed!")

    # ==================== BUSINESS METRICS ====================
    st.subheader("📈 Business Impact Estimation")
    downtime_cost = st.number_input("Estimated hourly downtime cost ($)", 1000, 100000, 10000, 1000)
    prevented_hours = severity*8
    potential_savings = downtime_cost*prevented_hours
    system_cost = 120000
    roi = potential_savings/system_cost
    st.metric("💾 Downtime Prevented", f"{prevented_hours} hours")
    st.metric("💰 Estimated Savings", f"${potential_savings:,.0f}")
    st.metric("📊 ROI multiplier", f"{roi:.1f}x")
    # ==================== PROFESSIONAL REPORT GENERATOR ====================
st.subheader("📊 Professional Report Generator")

def generate_text_report(features_dict, fault_type, severity, prevented_hours, potential_savings, roi):
    report = f"""
AVCS DNA SIMULATION REPORT
Generated on: {time.strftime('%Y-%m-%d %H:%M:%S')}

FAULT SIMULATION:
- Type: {fault_type}
- Severity: {severity}/5
- Downtime Prevented: {prevented_hours} hours
- Estimated Savings: ${potential_savings:,.0f}
- ROI: {roi:.1f}x

TECHNICAL PARAMETERS:
"""
    for s, f in features_dict.items():
        report += f"- {s}: RMS={f['rms']:.4f}, PkPk={f['pkpk']:.3f}, Crest={f['crest']:.2f}\n"
    report += "\nConclusion: AVCS DNA shows effective vibration control and significant ROI potential."
    return report

def generate_pdf_report(filename="avcs_simulation_report.pdf"):
    if not REPORTLAB_AVAILABLE:
        return None
    try:
        c = canvas.Canvas(filename, pagesize=letter)
        width, height = letter
        c.setFont("Helvetica-Bold", 16)
        c.drawString(50, height-50, "AVCS DNA Simulation Report")
        c.setFont("Helvetica", 12)
        c.drawString(50, height-70, f"Generated on: {time.strftime('%Y-%m-%d %H:%M:%S')}")
        y = height-100
        c.setFont("Helvetica", 10)
        for s, f in features_dict.items():
            c.drawString(50, y, f"{s}: RMS={f['rms']:.4f}, PkPk={f['pkpk']:.3f}, Crest={f['crest']:.2f}")
            y -= 15
        c.showPage()
        c.save()
        return filename
    except Exception as e:
        st.error(f"PDF generation failed: {e}")
        return None

# Кнопки для генерации
if REPORTLAB_AVAILABLE:
    if st.button("📄 Generate PDF Report"):
        pdf_file = generate_pdf_report()
        if pdf_file:
            with open(pdf_file, "rb") as f:
                st.download_button("📥 Download PDF Report", f, file_name="avcs_simulation_report.pdf")
            st.success("✅ PDF report generated successfully!")
else:
    st.info("💡 Reportlab not installed, generating text report instead.")
    if st.button("📄 Generate Text Report"):
        text_report = generate_text_report(features_dict, fault_type, severity, prevented_hours, potential_savings, roi)
        st.text_area("Professional Report:", text_report, height=300)
