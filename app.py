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
    animation_speed = st.slider("**Animation Speed**", 1, 5, 3) if show_animation else 3

    run_simulation = st.button("▶️ Start Live Simulation", type="primary")

# Основная логика
if run_simulation:
    if show_animation:
        # ==================== АНИМАЦИОННАЯ ВЕРСИЯ ====================
        animation_placeholder = st.empty()
        progress_bar = st.progress(0)
        status_display = st.empty()
        
        num_frames = 10
        time_points = np.linspace(0, 0.1, 500)
        
        # Сохраняем финальные данные для LinkedIn
        final_signal_data = None
        final_suppressed_signal = None
        final_fault_detected = None
        
        for frame in range(num_frames):
            progress = (frame + 1) / num_frames
            progress_bar.progress(progress)
            status_display.text(f"🎬 Live Simulation: Frame {frame+1}/{num_frames}")
            
            # Генерация сигнала с анимацией
            base_frequency = 50 + 2 * np.sin(frame * 0.2)  # Плавное изменение частоты
            base_signal = np.sin(2 * np.pi * base_frequency * time_points)
            base_signal += 0.1 * np.random.randn(len(time_points))
            
            # Моделирование неисправности с анимацией
            if fault_type == "Normal Operation":
                signal_data = base_signal
                fault_detected = False
                impulses = np.zeros_like(time_points)
            elif "Bearing_Fault" in fault_type:
                impulse_prob = 0.01 * severity * (1 + 0.5 * np.sin(frame * 0.3))  # Пульсирующие импульсы
                impulses = (np.random.rand(len(time_points)) < impulse_prob).astype(float) * severity * 0.8
                signal_data = base_signal + impulses
                fault_detected = True
            elif fault_type == "Imbalance":
                imbalance_effect = 0.3 * severity * (1 + 0.2 * np.sin(frame * 0.4))
                signal_data = base_signal * (1 + imbalance_effect * np.sin(2 * np.pi * 50 * time_points))
                fault_detected = severity >= 1
                impulses = np.zeros_like(time_points)
            elif fault_type == "Misalignment":
                harmonic_strength = 0.4 * severity * (1 + 0.1 * np.sin(frame * 0.3))
                harmonic_2x = harmonic_strength * np.sin(2 * np.pi * 100 * time_points + frame * 0.2)
                signal_data = base_signal + harmonic_2x
                fault_detected = severity >= 1
                impulses = np.zeros_like(time_points)

            # Демпферы с анимацией
            suppressed_signal, damper_force = simulate_dampers(signal_data, fault_detected, severity, dampers_enabled)
            
            # Сохраняем последний кадр для LinkedIn
            if frame == num_frames - 1:
                final_signal_data = signal_data
                final_suppressed_signal = suppressed_signal
                final_fault_detected = fault_detected
            
            # Визуализация анимации
            fig_anim = go.Figure()
            fig_anim.add_trace(go.Scatter(
                x=time_points*1000, 
                y=signal_data, 
                mode='lines', 
                name='Vibration', 
                line=dict(color='blue', width=2)
            ))
            
            if "Bearing_Fault" in fault_type:
                fig_anim.add_trace(go.Scatter(
                    x=time_points*1000, 
                    y=impulses, 
                    mode='lines', 
                    name='Bearing Impacts', 
                    line=dict(color='orange', width=2)
                ))
            
            if dampers_enabled and fault_detected and len(suppressed_signal) > 0:
                fig_anim.add_trace(go.Scatter(
                    x=time_points*1000, 
                    y=suppressed_signal, 
                    mode='lines', 
                    name='Suppressed', 
                    line=dict(color='green', width=2)
                ))
                
                if len(damper_force) > 0:
                    fig_anim.add_trace(go.Scatter(
                        x=time_points*1000, 
                        y=damper_force/50,
                        mode='lines', 
                        name='Damper Force/50', 
                        line=dict(color='red', width=2, dash='dot'),
                        yaxis='y2'
                    ))
                    
                    fig_anim.update_layout(
                        yaxis2=dict(
                            title="Damper Force (N/50)",
                            overlaying='y',
                            side='right'
                        )
                    )
            
            # Статус аномалии
            status_color = "green" if not fault_detected else "red"
            status_text = "🟢 NORMAL" if not fault_detected else "🔴 FAULT DETECTED"
            
            fig_anim.add_annotation(
                x=0.02, y=0.98, xref="paper", yref="paper",
                text=status_text,
                showarrow=False, 
                bgcolor="white", 
                bordercolor=status_color,
                borderwidth=2,
                font=dict(color=status_color, size=12)
            )
            
            fig_anim.update_layout(
                title=f"Live Animation - Frame {frame+1}/{num_frames} - {fault_type}", 
                height=400,
                showlegend=True,
                xaxis_title="Time (ms)",
                yaxis_title="Amplitude"
            )
            
            animation_placeholder.plotly_chart(fig_anim, use_container_width=True)
            time.sleep(0.5 / animation_speed)
        
        progress_bar.empty()
        status_display.success("✅ Live simulation completed!")
        
        # Инженерная панель для финального кадра
        features = calculate_features(final_signal_data)
        show_engineering_panel(final_signal_data, final_suppressed_signal, final_fault_detected, 
                             severity, fault_type, dampers_enabled, features)
        
        # LinkedIn для анимационной версии
        fig_linkedin = go.Figure()
        color = "green" if not final_fault_detected else "red"
        fig_linkedin.add_trace(go.Scatter(
            x=time_points*1000, 
            y=final_signal_data, 
            mode='lines', 
            name='Vibration', 
            line=dict(color=color, width=3)
        ))
        
        if dampers_enabled and final_fault_detected:
            fig_linkedin.add_trace(go.Scatter(
                x=time_points*1000, 
                y=final_suppressed_signal, 
                mode='lines', 
                name='Suppressed', 
                line=dict(color='blue', width=2)
            ))
        
        fig_linkedin.update_layout(
            height=400, 
            title=f"AVCS DNA Simulation - {fault_type}",
            showlegend=True
        )
        
    else:
        # ==================== СТАТИЧЕСКАЯ ВЕРСИЯ ====================
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

        # График для статической версии
        fig_linkedin = go.Figure()
        color = "green" if not fault_detected else "red"
        fig_linkedin.add_trace(go.Scatter(
            y=signal_data, 
            mode='lines', 
            name='Vibration', 
            line=dict(color=color, width=2)
        ))

        if dampers_enabled and fault_detected:
            fig_linkedin.add_trace(go.Scatter(
                y=suppressed_signal, 
                mode='lines', 
                name='Suppressed', 
                line=dict(color='blue', width=2)
            ))

        fig_linkedin.update_layout(
            height=400, 
            title=f"Simulation - {fault_type}",
            showlegend=True
        )
        
        st.plotly_chart(fig_linkedin, use_container_width=True)
        show_engineering_panel(signal_data, suppressed_signal, fault_detected, 
                             severity, fault_type, dampers_enabled, features)
        final_fault_detected = fault_detected

    # Бизнес-метрики и LinkedIn (общие для обеих версий)
    prevented_hours, potential_savings, system_cost = show_business_impact(severity)
    roi = potential_savings / system_cost if system_cost > 0 else 0

    # LinkedIn блок
    linkedin_text, linkedin_img = generate_linkedin_post(
        fault_type, severity, prevented_hours, potential_savings, roi, fig_linkedin
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
st.markdown("**Operational Excellence, Delivered** | © 2024 AVCS DNA Technology Simulator v3.4")
