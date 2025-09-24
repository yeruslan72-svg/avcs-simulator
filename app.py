import streamlit as st
import numpy as np
import plotly.graph_objects as go
from datetime import datetime
import time
import io
from PIL import Image

# Настройка страницы
st.set_page_config(page_title="AVCS DNA Simulator | Engineering Panel", layout="wide")
st.title("🛠️ AVCS DNA Technology Simulator - Engineering Panel")
st.markdown("""
**Operational Excellence, Delivered** - Real-time industrial monitoring with full engineering visibility
""")

# ==================== ФУНКЦИИ ====================

def calculate_features(signal_data):
    if len(signal_data) == 0:
        return {'rms': 0, 'pkpk': 0, 'crest': 0, 'variance': 0, 'centroid': 0, 'dominant_freqs': [0, 0], 'kurtosis': 0}

    rms = np.sqrt(np.mean(signal_data**2))
    pkpk = np.ptp(signal_data)
    crest = np.max(np.abs(signal_data)) / rms if rms > 0 else 0
    variance = np.var(signal_data)

    try:
        fft = np.fft.fft(signal_data)
        freqs = np.fft.fftfreq(len(signal_data))
        magnitude = np.abs(fft)

        if np.sum(magnitude) > 0:
            centroid = np.sum(freqs * magnitude) / np.sum(magnitude)
        else:
            centroid = 0

        dominant_idx = np.argsort(magnitude)[-3:][::-1]
        dominant_freqs = [freqs[i] * 1000 for i in dominant_idx if i < len(freqs)]
        while len(dominant_freqs) < 2:
            dominant_freqs.append(0)

        kurtosis_val = np.mean(magnitude**4) / (np.mean(magnitude**2)**2) - 3 if np.mean(magnitude**2) > 0 else 0
    except:
        centroid = 0
        dominant_freqs = [0, 0]
        kurtosis_val = 0

    return {
        'rms': rms,
        'pkpk': pkpk,
        'crest': crest,
        'variance': variance,
        'centroid': abs(centroid * 1000),
        'dominant_freqs': dominant_freqs[:2],
        'kurtosis': kurtosis_val
    }

def simulate_dampers(signal_data, fault_detected, severity, enabled=True):
    n = len(signal_data)
    if n == 0:
        return np.array([]), np.array([])

    if not enabled or not fault_detected:
        return signal_data * 0.98, np.full(n, 500)

    damper_force_value = min(8000, severity * 1600)
    damper_force = np.full(n, damper_force_value)
    suppression_factor = np.exp(-0.3 * damper_force_value / 8000)
    suppressed_signal = signal_data * suppression_factor

    return suppressed_signal, damper_force

def show_engineering_panel(signal_data, suppressed_signal, fault_detected, 
                          severity, fault_type, dampers_enabled, features):
    st.subheader("🔧 Engineering Panel - Real-time Diagnostics")

    if dampers_enabled and fault_detected and len(signal_data) > 0 and len(suppressed_signal) > 0:
        std_signal = np.std(signal_data)
        std_suppressed = np.std(suppressed_signal)
        vibration_reduction = (1 - std_suppressed/std_signal) * 100 if std_signal > 0 else 0
    else:
        vibration_reduction = 0

    col_eng1, col_eng2, col_eng3 = st.columns(3)

    with col_eng1:
        st.markdown("**📈 Time-domain Features**")
        st.metric("RMS", f"{features['rms']:.4f}")
        st.metric("Peak-to-Peak", f"{features['pkpk']:.3f}")
        st.metric("Crest Factor", f"{features['crest']:.2f}")

    with col_eng2:
        st.markdown("**📊 Frequency-domain Features**")
        st.metric("Spectral Centroid", f"{features['centroid']:.1f} Hz")
        st.metric("Dominant Freq 1", f"{features['dominant_freqs'][0]:.1f} Hz")
        st.metric("Dominant Freq 2", f"{features['dominant_freqs'][1]:.1f} Hz")

    with col_eng3:
        st.markdown("**⚡ System Diagnosis**")
        fault_color = "🟢" if not fault_detected else "🔴"
        confidence = 0.98 if not fault_detected else min(0.3 + severity * 0.15, 0.95)
        st.metric("Fault Type", f"{fault_color} {fault_type}")
        st.metric("Severity", severity)
        st.metric("Confidence", f"{confidence:.1%}")
        st.metric("Vibration Reduction", f"{vibration_reduction:.1f}%")

def show_business_impact(severity):
    st.subheader("📈 Business Impact Estimation")

    col_cost, col_impact = st.columns(2)

    with col_cost:
        downtime_cost = st.number_input("Estimated hourly downtime cost ($)", 
                                      min_value=1000, value=10000, step=1000,
                                      key="downtime_cost")

    with col_impact:
        prevented_hours = severity * 8
        potential_savings = downtime_cost * prevented_hours
        system_cost = 120000

        st.metric("💾 Potential downtime prevented", f"{prevented_hours} hours")
        st.metric("💰 Estimated savings", f"${potential_savings:,.0f}")
        if system_cost > 0:
            st.metric("📊 ROI multiplier", f"{potential_savings/system_cost:.1f}x")

    return prevented_hours, potential_savings, system_cost

def generate_linkedin_post(fault_type, severity, prevented_hours, potential_savings, roi, fig):
    buf = io.BytesIO()
    fig.write_image(buf, format="png")
    buf.seek(0)
    img = Image.open(buf)

    post_text = f"""
🚀 **Predictive Maintenance in Action – AVCS DNA**

Fault simulated: **{fault_type}**  
Severity: **{severity}/5**  
Downtime prevented: **{prevented_hours:.1f} h**  
Estimated savings: **${potential_savings:,.0f}**  
ROI multiplier: **{roi:.1f}x**

At late-life operation stage, every prevented hour matters.  
AVCS DNA ensures asset integrity and eliminates last-minute failures.

#PredictiveMaintenance #AssetIntegrity #Decommissioning #ROI #OperationalExcellence
    """.strip()

    return post_text, img

# ==================== ОСНОВНОЙ ИНТЕРФЕЙС ====================

col1, col2 = st.columns([1, 2])

with col1:
    st.subheader("🎛️ Configuration")

    fault_type = st.selectbox(
        "**Fault Type**",
        ["Normal Operation", "Bearing_Fault_Mild", "Bearing_Fault_Severe", "Imbalance", "Misalignment"]
    )

    severity = st.slider("**Fault Severity**", 1, 5, 1)
    dampers_enabled = st.checkbox("**Enable Active Dampers**", value=True)

    show_animation = st.checkbox("**Show Live Animation**", value=True)

    run_simulation = st.button("▶️ Start Live Simulation", type="primary")

if run_simulation:
    # Статическая версия графика (для LinkedIn)
    t = np.linspace(0, 0.1, 1000)
    base_signal = np.sin(2 * np.pi * 50 * t) + 0.1 * np.random.randn(1000)

    if fault_type == "Normal Operation":
        signal_data = base_signal
        fault_detected = False
    elif "Bearing_Fault" in fault_type:
        impulses = (np.random.rand(1000) < 0.02 * severity).astype(float) * severity * 0.8
        signal_data = base_signal + impulses
        fault_detected = True
    else:
        signal_data = base_signal * (1 + 0.3 * severity * np.sin(2 * np.pi * 50 * t))
        fault_detected = severity >= 1

    suppressed_signal, damper_force = simulate_dampers(signal_data, fault_detected, severity, dampers_enabled)
    features = calculate_features(signal_data)

    fig = go.Figure()
    color = "green" if not fault_detected else "red"
    fig.add_trace(go.Scatter(y=signal_data, mode='lines', name='Vibration', line=dict(color=color, width=2)))

    if dampers_enabled and fault_detected:
        fig.add_trace(go.Scatter(y=suppressed_signal, mode='lines', name='Suppressed', line=dict(color='blue', width=2)))

    fig.update_layout(height=400, title=f"Simulation - {fault_type}")
    st.plotly_chart(fig, use_container_width=True)

    show_engineering_panel(signal_data, suppressed_signal, fault_detected, 
                         severity, fault_type, dampers_enabled, features)

    prevented_hours, potential_savings, system_cost = show_business_impact(severity)
    roi = potential_savings / system_cost if system_cost > 0 else 0

    # LinkedIn блок
    linkedin_text, linkedin_img = generate_linkedin_post(
        fault_type, severity, prevented_hours, potential_savings, roi, fig
    )

    st.subheader("📢 LinkedIn-ready Post")
    st.text_area("Suggested text:", linkedin_text, height=200)
    st.image(linkedin_img, caption="Attach this graph to your post", use_container_width=True)
    st.success("✅ Copy text & download image → Ready for LinkedIn")

# Technology Stack
with st.expander("🔧 Under the Hood: AVCS DNA Technology Stack"):
    st.markdown("""
    **Industrial-Grade Vibration Monitoring System**
    - Real-time signal processing at 10kHz
    - Machine Learning anomaly detection
    - Active vibration control with MR dampers (0-8000N)
    - 12 features per sensor for precise diagnostics
    - >2000% ROI from prevented downtime
    """)

# Call-to-Action
st.markdown("---")
st.subheader("🚀 Ready to Deploy AVCS DNA on Your Equipment?")

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("**📞 Technical Briefing**")
    st.markdown("Live demo with your data")

with col2:
    st.markdown("**📧 Contact**")
    st.markdown("yeruslan@operationalexcellence.com")

with col3:
    st.markdown("**📚 Resources**")
    st.markdown("Case studies & ROI analysis")

st.markdown("---")
st.markdown("**Operational Excellence, Delivered** | © 2024 AVCS DNA Technology Simulator v3.3")
