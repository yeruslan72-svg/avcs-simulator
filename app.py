import streamlit as st
import numpy as np
import plotly.graph_objects as go
import time
import io
from PIL import Image

# ==================== Настройка страницы ====================
st.set_page_config(page_title="AVCS DNA Simulator | Engineering Panel", layout="wide")
st.title("🛠️ AVCS DNA Technology Simulator - Engineering Panel")
st.markdown("""
**Operational Excellence, Delivered** - Real-time industrial monitoring with full engineering visibility
""")

# ==================== ФУНКЦИИ ====================

def calculate_features(signal_data):
    if len(signal_data) == 0:
        return {'rms':0,'pkpk':0,'crest':0,'variance':0,'centroid':0,'dominant_freqs':[0,0],'kurtosis':0}
    rms = np.sqrt(np.mean(signal_data**2))
    pkpk = np.ptp(signal_data)
    crest = np.max(np.abs(signal_data))/rms if rms>0 else 0
    variance = np.var(signal_data)
    try:
        fft = np.fft.fft(signal_data)
        freqs = np.fft.fftfreq(len(signal_data))
        mag = np.abs(fft)
        centroid = np.sum(freqs*mag)/np.sum(mag) if np.sum(mag)>0 else 0
        dom_idx = np.argsort(mag)[-3:][::-1]
        dom_freqs = [freqs[i]*1000 for i in dom_idx if i<len(freqs)]
        while len(dom_freqs)<2: dom_freqs.append(0)
        kurtosis_val = (np.mean(mag**4)/(np.mean(mag**2)**2)-3 if np.mean(mag**2)>0 else 0)
    except:
        centroid=0; dom_freqs=[0,0]; kurtosis_val=0
    return {'rms':rms,'pkpk':pkpk,'crest':crest,'variance':variance,'centroid':abs(centroid*1000),'dominant_freqs':dom_freqs[:2],'kurtosis':kurtosis_val}

def simulate_dampers(signal_data, fault_detected, severity, enabled=True):
    n = len(signal_data)
    if n==0: return np.array([]), np.array([])
    if not enabled or not fault_detected:
        return signal_data*0.98, np.full(n,500)
    damper_force_value = min(8000, severity*1600)
    damper_force = np.full(n, damper_force_value)
    suppression_factor = np.exp(-0.3*damper_force_value/8000)
    suppressed_signal = signal_data*suppression_factor
    return suppressed_signal, damper_force

def show_engineering_panel(signal_data, suppressed_signal, fault_detected,
                           severity, fault_type, dampers_enabled, features):
    st.subheader("🔧 Engineering Panel - Real-time Diagnostics")
    vibration_reduction = 0
    if dampers_enabled and fault_detected and len(signal_data)>0 and len(suppressed_signal)>0:
        std_signal = np.std(signal_data)
        std_suppressed = np.std(suppressed_signal)
        vibration_reduction = (1-std_suppressed/std_signal)*100 if std_signal>0 else 0

    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("**📈 Time-domain Features**")
        st.metric("RMS", f"{features['rms']:.4f}")
        st.metric("Peak-to-Peak", f"{features['pkpk']:.3f}")
        st.metric("Crest Factor", f"{features['crest']:.2f}")
    with col2:
        st.markdown("**📊 Frequency-domain Features**")
        st.metric("Spectral Centroid", f"{features['centroid']:.1f} Hz")
        st.metric("Dominant Freq 1", f"{features['dominant_freqs'][0]:.1f} Hz")
        st.metric("Dominant Freq 2", f"{features['dominant_freqs'][1]:.1f} Hz")
    with col3:
        st.markdown("**⚡ System Diagnosis**")
        fault_color = "🟢" if not fault_detected else "🔴"
        confidence = 0.98 if not fault_detected else min(0.3 + severity*0.15,0.95)
        st.metric("Fault Type", f"{fault_color} {fault_type}")
        st.metric("Severity", severity)
        st.metric("Confidence", f"{confidence:.1%}")
        st.metric("Vibration Reduction", f"{vibration_reduction:.1f}%")

def show_business_impact(severity):
    st.subheader("📈 Business Impact Estimation")
    col_cost, col_impact = st.columns(2)
    with col_cost:
        downtime_cost = st.number_input("Estimated hourly downtime cost ($)", min_value=1000,value=10000,step=1000,key="downtime_cost")
    with col_impact:
        prevented_hours = severity*8
        potential_savings = downtime_cost*prevented_hours
        system_cost = 120000
        st.metric("💾 Potential downtime prevented", f"{prevented_hours} hours")
        st.metric("💰 Estimated savings", f"${potential_savings:,.0f}")
        st.metric("📊 ROI multiplier", f"{potential_savings/system_cost:.1f}x")
    return prevented_hours, potential_savings, system_cost

def generate_linkedin_post(fault_type,severity,prevented_hours,potential_savings,roi,fig):
    buf = io.BytesIO()
    try:
        fig.write_image(buf, format="png")
        buf.seek(0)
        img = Image.open(buf)
    except:
        img = None
    text = f"""
🚀 **Predictive Maintenance – AVCS DNA**
Fault simulated: **{fault_type}**  
Severity: **{severity}/5**  
Downtime prevented: **{prevented_hours:.1f} h**  
Estimated savings: **${potential_savings:,.0f}**  
ROI multiplier: **{roi:.1f}x**
#PredictiveMaintenance #OperationalExcellence
    """.strip()
    return text,img

# ==================== Интерфейс ====================
col1,col2 = st.columns([1,2])
with col1:
    st.subheader("🎛️ Configuration")
    fault_type = st.selectbox("**Fault Type**", ["Normal Operation","Bearing_Fault_Mild","Bearing_Fault_Severe","Imbalance","Misalignment"])
    severity = st.slider("**Fault Severity**",1,5,1)
    dampers_enabled = st.checkbox("**Enable Active Dampers**",value=True)
    enable_animation = st.checkbox("Enable Live Animation", value=True)
    animation_speed = st.slider("Animation Speed",1,5,3)
    run_sim = st.button("▶️ Start Simulation",type="primary")

# ==================== Симуляция ====================
if run_sim:
    time_points = np.linspace(0,0.1,500)
    final_signal = None
    final_suppressed = None
    final_fault = None

    if enable_animation:
        animation_placeholder = st.empty()
        num_frames = 15
        for frame in range(num_frames):
            base_freq = 50 + 2*np.sin(frame*0.2)
            base_signal = np.sin(2*np.pi*base_freq*time_points) + 0.1*np.random.randn(len(time_points))

            if fault_type=="Normal Operation":
                signal_data = base_signal
                fault_detected = False
            elif "Bearing_Fault" in fault_type:
                prob = 0.01*severity*(1+0.5*np.sin(frame*0.3))
                impulses = (np.random.rand(len(time_points))<prob).astype(float)*severity*0.8
                signal_data = base_signal + impulses
                fault_detected = True
            elif fault_type=="Imbalance":
                signal_data = base_signal*(1 + 0.3*severity*np.sin(2*np.pi*50*time_points))
                fault_detected = True
            else:
                harmonic = 0.4*severity*np.sin(2*np.pi*100*time_points)
                signal_data = base_signal + harmonic
                fault_detected = True

            suppressed_signal, damper_force = simulate_dampers(signal_data, fault_detected, severity, dampers_enabled)

            if frame==num_frames-1:
                final_signal = signal_data
                final_suppressed = suppressed_signal
                final_fault = fault_detected

            fig = go.Figure()
            fig.add_trace(go.Scatter(x=time_points*1000, y=signal_data, mode='lines', name='Vibration', line=dict(color='blue')))
            if fault_detected:
                fig.add_trace(go.Scatter(x=time_points*1000, y=suppressed_signal, mode='lines', name='Suppressed', line=dict(color='green')))
            fig.update_layout(title=f"Live Simulation - Frame {frame+1}/{num_frames}", height=400)
            animation_placeholder.plotly_chart(fig,use_container_width=True)
            time.sleep(0.4/animation_speed)

    else:
        base_signal = np.sin(2*np.pi*50*time_points) + 0.1*np.random.randn(len(time_points))
        if fault_type=="Normal Operation":
            final_signal = base_signal
            final_fault = False
        elif "Bearing_Fault" in fault_type:
            impulses = (np.random.rand(len(time_points))<0.02*severity).astype(float)*severity*0.8
            final_signal = base_signal + impulses
            final_fault = True
        elif fault_type=="Imbalance":
            final_signal = base_signal*(1 + 0.3*severity*np.sin(2*np.pi*50*time_points))
            final_fault = True
        else:
            harmonic = 0.4*severity*np.sin(2*np.pi*100*time_points)
            final_signal = base_signal + harmonic
            final_fault = True

        final_suppressed = simulate_dampers(final_signal, final_fault, severity, dampers_enabled)[0]

        fig = go.Figure()
        color = "green" if not final_fault else "red"
        fig.add_trace(go.Scatter(x=time_points*1000, y=final_signal, mode='lines', name='Vibration', line=dict(color=color)))
        if dampers_enabled and final_fault:
            fig.add_trace(go.Scatter(x=time_points*1000, y=final_suppressed, mode='lines', name='Suppressed', line=dict(color='blue')))
        fig.update_layout(title=f"Static Simulation - {fault_type}", height=400)
        st.plotly_chart(fig,use_container_width=True)

    features = calculate_features(final_signal)
    show_engineering_panel(final_signal, final_suppressed, final_fault, severity, fault_type, dampers_enabled, features)
    prevented_hours, potential_savings, system_cost = show_business_impact(severity)
    roi = potential_savings/system_cost if system_cost>0 else 0

    linkedin_text, linkedin_img = generate_linkedin_post(fault_type, severity, prevented_hours, potential_savings, roi, fig)
    st.subheader("📢 LinkedIn-ready Post")
    st.text_area("Suggested text:", linkedin_text,height=200)
    if linkedin_img:
        st.image(linkedin_img,caption="Attach this graph to your post",use_container_width=True)

# ==================== Технологический стек ====================
with st.expander("🔧 Under the Hood: AVCS DNA Technology Stack"):
    st.markdown("""
    **Industrial-Grade Vibration Monitoring System**
    - Real-time signal processing at 10kHz
    - ML anomaly detection
    - Active vibration control with MR dampers (0-8000N)
    - 12 features per sensor
    - >2000% ROI from prevented downtime
    """)

st.markdown("---")
st.subheader("🚀 Ready to Deploy AVCS DNA on Your Equipment?")
c1,c2,c3 = st.columns(3)
with c1: st.markdown("**📞 Technical Briefing**\nLive demo with your data")
with c2: st.markdown("**📧 Contact**\nyeruslan@operationalexcellence.com")
with c3: st.markdown("**📚 Resources**\nCase studies & ROI analysis")
st.markdown("---")
st.markdown("**Operational Excellence, Delivered** | © 2025 AVCS DNA Technology Simulator v3.8")
