import streamlit as st
import numpy as np
import plotly.graph_objects as go
from datetime import datetime
import time

# Настройка страницы
st.set_page_config(page_title="AVCS DNA Simulator | Engineering Panel", layout="wide")
st.title("🛠️ AVCS DNA Technology Simulator - Engineering Panel")
st.markdown("""
**Operational Excellence, Delivered** - Real-time industrial monitoring with full engineering visibility
""")

# ==================== ВСЕ ФУНКЦИИ ДЛЯ ФИЧЕК ====================

def calculate_features(signal_data):
    """Расчет всех фич как в промышленной системе"""
    rms = np.sqrt(np.mean(signal_data**2))
    pkpk = np.ptp(signal_data)
    crest = np.max(np.abs(signal_data)) / rms if rms > 0 else 0
    variance = np.var(signal_data)
    
    # Упрощенный спектральный анализ
    fft = np.fft.fft(signal_data)
    freqs = np.fft.fftfreq(len(signal_data))
    magnitude = np.abs(fft)
    
    # Спектральный центроид
    if np.sum(magnitude) > 0:
        centroid = np.sum(freqs * magnitude) / np.sum(magnitude)
    else:
        centroid = 0
    
    # Доминантные частоты
    dominant_idx = np.argsort(magnitude)[-3:][::-1]
    dominant_freqs = freqs[dominant_idx] * 1000  # Нормализация
    
    return {
        'rms': rms,
        'pkpk': pkpk,
        'crest': crest,
        'variance': variance,
        'centroid': abs(centroid * 1000),  # Hz
        'dominant_freqs': dominant_freqs[:2],  # Top 2 frequencies
        'kurtosis': np.mean(magnitude**4) / (np.mean(magnitude**2)**2) - 3
    }

def simulate_dampers(signal_data, fault_detected, severity, enabled=True):
    """Полная модель демпферов"""
    if not enabled or not fault_detected:
        return signal_data * 0.98, 500  # Легкое демпфирование
    
    # Активное подавление
    damper_force = min(8000, severity * 1600)
    suppression_factor = np.exp(-0.3 * damper_force / 8000)
    suppressed_signal = signal_data * suppression_factor
    
    return suppressed_signal, damper_force

def show_engineering_panel(signal_data, suppressed_signal, fault_detected, 
                          severity, fault_type, dampers_enabled, features):
    """Инженерная панель со всеми структурами данных"""
    
    st.subheader("🔧 Engineering Panel - Real-time Diagnostics")
    
    # Расчет эффективности
    if dampers_enabled and fault_detected:
        vibration_reduction = (1 - np.std(suppressed_signal)/np.std(signal_data)) * 100
    else:
        vibration_reduction = 0
    
    # Показываем данные в инженерном формате
    col_eng1, col_eng2, col_eng3 = st.columns(3)
    
    with col_eng1:
        st.markdown("**📈 Time-domain Features**")
        st.metric("RMS", f"{features['rms']:.4f}")
        st.metric("Peak-to-Peak", f"{features['pkpk']:.3f}")
        st.metric("Crest Factor", f"{features['crest']:.2f}")
        st.metric("Variance", f"{features['variance']:.6f}")
        
    with col_eng2:
        st.markdown("**📊 Frequency-domain Features**")
        st.metric("Spectral Centroid", f"{features['centroid']:.1f} Hz")
        st.metric("Dominant Freq 1", f"{features['dominant_freqs'][0]:.1f} Hz")
        st.metric("Dominant Freq 2", f"{features['dominant_freqs'][1]:.1f} Hz" if len(features['dominant_freqs']) > 1 else "N/A")
        st.metric("Spectral Kurtosis", f"{features['kurtosis']:.2f}")
        
    with col_eng3:
        st.markdown("**⚡ System Diagnosis**")
        fault_color = "🟢" if not fault_detected else "🔴"
        confidence = 0.98 if not fault_detected else min(0.3 + severity * 0.15, 0.95)
        st.metric("Fault Type", f"{fault_color} {fault_type}")
        st.metric("Severity", severity)
        st.metric("Confidence", f"{confidence:.1%}")
        st.metric("Vibration Reduction", f"{vibration_reduction:.1f}%")
    
    # Детальная инженерная информация
    with st.expander("🔍 Detailed Engineering Data (TwinCAT Structures)"):
        st.code(f"""
// ST_SystemConfig (Industrial Configuration)
nSampleRate_Hz: 10000
nBufferSize: 1000  
nNumSensors: 4
nNumFeatures: 12

// ST_Features_V2 (Extracted Features)
rRMS: {features['rms']:.6f}
rPeakToPeak: {features['pkpk']:.6f}
rCrestFactor: {features['crest']:.4f}
rVariance: {features['variance']:.8f}
rSpectralCentroid: {features['centroid']:.2f}
rSpectralKurtosis: {features['kurtosis']:.4f}
aDominantFreqs[1]: {features['dominant_freqs'][0]:.2f}
aDominantFreqs[2]: {features['dominant_freqs'][1]:.2f if len(features['dominant_freqs']) > 1 else 0.0}

// ST_Diagnosis (ML Inference Result)
FaultType: {fault_type}
Severity: {severity}
Confidence: {confidence:.3f}
Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')}
        """, language='cpp')

def show_business_impact(severity, downtime_cost=10000):
    """Бизнес-метрики и ROI калькулятор"""
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
    
    # Режим анимации
    animation_speed = st.slider("**Animation Speed**", 1, 10, 5)
    show_animation = st.checkbox("**Show Live Animation**", value=True)
    
    run_simulation = st.button("▶️ Start Live Simulation", type="primary")

# Основная логика
if run_simulation:
    if show_animation:
        # АНИМАЦИОННАЯ ВЕРСИЯ
        animation_placeholder = st.empty()
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        num_frames = 15
        time_points = np.linspace(0, 0.1, 1000)
        
        for frame in range(num_frames):
            progress = (frame + 1) / num_frames
            progress_bar.progress(progress)
            status_text.text(f"🎬 Live Simulation: Frame {frame+1}/{num_frames}")
            
            # Генерация сигнала с анимацией
            base_frequency = 50 + 2 * np.sin(frame * 0.1)
            base_signal = np.sin(2 * np.pi * base_frequency * time_points)
            base_signal += 0.1 * np.random.randn(len(time_points))
            
            # Моделирование неисправности - ИСПРАВЛЕННАЯ ЛОГИКА!
            if fault_type == "Normal Operation":
                signal_data = base_signal
                fault_detected = False
            elif "Bearing_Fault" in fault_type:
                impulse_prob = 0.001 * severity
                impulses = (np.random.rand(len(time_points)) < impulse_prob).astype(float) * severity * 0.5
                signal_data = base_signal + impulses
                fault_detected = True  # ЛЮБОЙ Bearing_Fault это аномалия!
            elif fault_type == "Imbalance":
                imbalance_effect = 0.5 * severity
                signal_data = base_signal * (1 + imbalance_effect * np.sin(2 * np.pi * 50 * time_points))
                fault_detected = severity >= 1  # severity 1+ это аномалия
            elif fault_type == "Misalignment":
                harmonic_2x = 0.7 * severity * np.sin(2 * np.pi * 100 * time_points + np.pi/4)
                signal_data = base_signal + harmonic_2x
                fault_detected = severity >= 1  # severity 1+ это аномалия

            # Демпферы
            suppressed_signal, damper_force = simulate_dampers(signal_data, fault_detected, severity, dampers_enabled)
            
            # Визуализация
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=time_points*1000, y=signal_data, mode='lines', name='Vibration', line=dict(color='blue')))
            
            if dampers_enabled and fault_detected:
                fig.add_trace(go.Scatter(x=time_points*1000, y=suppressed_signal, mode='lines', name='Suppressed', line=dict(color='green')))
                fig.add_trace(go.Scatter(x=time_points*1000, y=damper_force/40, mode='lines', name='Damper Force/40', line=dict(color='red', dash='dot')))
            
            # Добавляем статус аномалии на график
            fig.add_annotation(
                x=0.02, y=0.98, xref="paper", yref="paper",
                text="🟢 NORMAL" if not fault_detected else "🔴 FAULT DETECTED",
                showarrow=False, bgcolor="white", bordercolor="black", borderwidth=1
            )
            
            fig.update_layout(
                title=f"Live Monitoring - Frame {frame+1}/{num_frames}", 
                height=400,
                showlegend=True
            )
            animation_placeholder.plotly_chart(fig, use_container_width=True)
            time.sleep(0.3 / animation_speed)
        
        progress_bar.empty()
        status_text.success("✅ Live simulation completed!")
        
        # Расчет фич для финального кадра
        features = calculate_features(signal_data)
        show_engineering_panel(signal_data, suppressed_signal, fault_detected, 
                             severity, fault_type, dampers_enabled, features)
        
    else:
        # СТАТИЧЕСКАЯ ВЕРСИЯ - ИСПРАВЛЕННАЯ ЛОГИКА!
        t = np.linspace(0, 0.1, 1000)
        base_signal = np.sin(2 * np.pi * 50 * t) + 0.1 * np.random.randn(1000)
        
        if fault_type == "Normal Operation":
            signal_data = base_signal
            fault_detected = False
        elif "Bearing_Fault" in fault_type:
            impulses = (np.random.rand(1000) < 0.01 * severity).astype(float) * severity * 0.5
            signal_data = base_signal + impulses
            fault_detected = True  # ЛЮБОЙ Bearing_Fault это аномалия!
        elif fault_type == "Imbalance":
            signal_data = base_signal * (1 + 0.3 * severity * np.sin(2 * np.pi * 50 * t))
            fault_detected = severity >= 1
        elif fault_type == "Misalignment":
            signal_data = base_signal + 0.5 * severity * np.sin(2 * np.pi * 100 * t)
            fault_detected = severity >= 1
            
        suppressed_signal, damper_force = simulate_dampers(signal_data, fault_detected, severity, dampers_enabled)
        features = calculate_features(signal_data)
        
        # График с индикацией статуса
        fig = go.Figure()
        fig.add_trace(go.Scatter(y=signal_data, mode='lines', name='Vibration', line=dict(color='green' if not fault_detected else 'red')))
        if dampers_enabled and fault_detected:
            fig.add_trace(go.Scatter(y=suppressed_signal, mode='lines', name='Suppressed', line=dict(color='blue')))
        
        fig.add_annotation(
            x=0.5, y=0.95, xref="paper", yref="paper",
            text="🟢 SYSTEM NORMAL" if not fault_detected else f"🔴 {fault_type} - Severity {severity}",
            showarrow=False, bgcolor="white", font=dict(size=16, color="black" if not fault_detected else "red")
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        show_engineering_panel(signal_data, suppressed_signal, fault_detected, 
                             severity, fault_type, dampers_enabled, features)

# Всегда показываем бизнес-метрики
current_severity = severity if run_simulation else 3
show_business_impact(current_severity)

# Technology Stack
with st.expander("🔧 Under the Hood: AVCS DNA Technology Stack"):
    st.markdown("""
    **Core Technologies:**
    - **Real-time signal processing**: Industrial-grade FFT analysis
    - **ML Anomaly Detection**: Isolation Forest + Gradient Boosting  
    - **Feature Extraction**: 12 parameters per sensor (RMS, Crest, Spectral features)
    - **Active Vibration Control**: MR dampers (0-8000N, <100ms response)
    - **Industrial Hardware**: Beckhoff PLCs, EtherCAT communication
    
    **Performance Metrics:**
    - Response time: <100 ms
    - Fault detection accuracy: >95%
    - Vibration reduction: up to 80%
    - ROI: >2000% from first prevented incident
    """)

# Call-to-Action
st.markdown("---")
st.subheader("🚀 Ready to Deploy AVCS DNA on Your Equipment?")

cta_col1, cta_col2, cta_col3 = st.columns(3)

with cta_col1:
    st.markdown("**📞 Schedule Technical Briefing**")
    st.markdown("""
    - Live demo with your data
    - Custom ROI calculation
    - Integration planning
    """)

with cta_col2:
    st.markdown("**📧 Contact Us**")
    st.markdown("""
    **Email:** yeruslan@operationalexcellence.com  
    **LinkedIn:** Yeruslan Chihachyov
    """)

with cta_col3:
    st.markdown("**📚 Resources**")
    st.markdown("""
    - Technical Specification PDF
    - Case Studies & ROI Analysis
    - Integration Guide
    """)

st.markdown("---")
st.markdown("**Operational Excellence, Delivered** | © 2024 AVCS DNA Technology Simulator v3.1")
