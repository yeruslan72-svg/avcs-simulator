import streamlit as st
import numpy as np
import plotly.graph_objects as go
from datetime import datetime
import time

# Настройка страницы
st.set_page_config(page_title="AVCS DNA Simulator | Live Animation", layout="wide")
st.title("🛠️ AVCS DNA Technology Simulator - Live Animation")
st.markdown("""
**Operational Excellence, Delivered** - Real-time vibration monitoring with live animation
""")

# Функция для инженерной панели
def show_engineering_panel(signal_data, suppressed_signal, fault_detected, severity, fault_type, dampers_enabled):
    """Функция для отображения инженерной панели после анимации"""
    
    st.subheader("🔧 Engineering Analysis - Final Frame")
    
    # Расчет параметров
    rms = np.sqrt(np.mean(signal_data**2))
    pkpk = np.ptp(signal_data)
    crest = np.max(np.abs(signal_data)) / rms if rms > 0 else 0
    
    if dampers_enabled and fault_detected:
        vibration_reduction = (1 - np.std(suppressed_signal)/np.std(signal_data)) * 100
    else:
        vibration_reduction = 0
    
    # Инженерные метрики
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("RMS Vibration", f"{rms:.4f}")
        st.metric("Peak-to-Peak", f"{pkpk:.3f}")
        
    with col2:
        st.metric("Crest Factor", f"{crest:.2f}")
        st.metric("Vibration Reduction", f"{vibration_reduction:.1f}%")
        
    with col3:
        status = "🟢 NORMAL" if not fault_detected else "🔴 FAULT"
        st.metric("System Status", status)
        st.metric("Fault Severity", severity)

# Создаем две колонки
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
    
    # Инженерные настройки
    with st.expander("⚙️ Engineering Settings"):
        sample_rate = st.number_input("Sample Rate (Hz)", 1000, 50000, 10000)
        buffer_size = st.number_input("Buffer Size", 256, 4096, 1000)
        num_sensors = st.selectbox("Number of Sensors", [1, 2, 4, 8], index=2)
    
    run_simulation = st.button("▶️ Start Live Simulation", type="primary")

# Основная логика приложения
if run_simulation:
    if show_animation:
        # АНИМАЦИОННАЯ ВЕРСИЯ
        animation_placeholder = st.empty()
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # Параметры анимации
        num_frames = 20  # Уменьшил для скорости
        time_points = np.linspace(0, 0.1, 1000)  # Фиксированный размер буфера
        
        for frame in range(num_frames):
            # Обновление прогресса
            progress = (frame + 1) / num_frames
            progress_bar.progress(progress)
            status_text.text(f"🎬 Live Simulation: Frame {frame+1}/{num_frames}")
            
            # Генерация сигнала
            base_frequency = 50 + 2 * np.sin(frame * 0.1)
            base_signal = np.sin(2 * np.pi * base_frequency * time_points)
            base_signal += 0.1 * np.random.randn(len(time_points))
            
            # Моделирование неисправности
            if fault_type == "Normal Operation":
                signal_data = base_signal
                fault_detected = False
            elif "Bearing_Fault" in fault_type:
                impulse_prob = 0.001 * severity
                impulses = (np.random.rand(len(time_points)) < impulse_prob).astype(float) * severity * 0.5
                signal_data = base_signal + impulses
                fault_detected = severity > 2
            elif fault_type == "Imbalance":
                imbalance_effect = 0.5 * severity
                signal_data = base_signal * (1 + imbalance_effect * np.sin(2 * np.pi * 50 * time_points))
                fault_detected = severity > 1
            elif fault_type == "Misalignment":
                harmonic_2x = 0.7 * severity * np.sin(2 * np.pi * 100 * time_points + np.pi/4)
                signal_data = base_signal + harmonic_2x
                fault_detected = severity > 1

            # Модель демпферов
            if dampers_enabled and fault_detected:
                damper_force = np.minimum(8000, severity * 1600 * np.ones_like(time_points))
                suppression_factor = np.exp(-0.3 * damper_force/8000)
                suppressed_signal = signal_data * suppression_factor
            else:
                damper_force = 500 * np.ones_like(time_points)
                suppressed_signal = signal_data

            # Создание анимированного графика
            fig = go.Figure()
            
            fig.add_trace(go.Scatter(
                x=time_points * 1000,
                y=signal_data,
                mode='lines',
                name='Vibration Signal',
                line=dict(color='blue', width=2)
            ))
            
            if dampers_enabled and fault_detected:
                fig.add_trace(go.Scatter(
                    x=time_points * 1000,
                    y=suppressed_signal,
                    mode='lines',
                    name='Suppressed Vibration',
                    line=dict(color='green', width=2)
                ))
            
            fig.update_layout(
                title=f"Live Vibration Monitoring - Frame {frame+1}/{num_frames}",
                xaxis_title="Time (milliseconds)",
                yaxis_title="Vibration Amplitude",
                height=400
            )
            
            animation_placeholder.plotly_chart(fig, use_container_width=True)
            time.sleep(0.5 / animation_speed)
        
        progress_bar.empty()
        status_text.success("✅ Live simulation completed!")
        
        # Показываем финальную инженерную панель
        show_engineering_panel(signal_data, suppressed_signal, fault_detected, 
                             severity, fault_type, dampers_enabled)
        
    else:
        # СТАТИЧЕСКАЯ ВЕРСИЯ (без анимации)
        st.info("🚫 Animation disabled - showing static analysis")
        
        # Простая генерация сигнала
        t = np.linspace(0, 0.1, 1000)
        base_signal = np.sin(2 * np.pi * 50 * t) + 0.1 * np.random.randn(1000)
        
        if "Bearing_Fault" in fault_type:
            impulses = (np.random.rand(1000) < 0.01 * severity).astype(float) * severity * 0.5
            signal_data = base_signal + impulses
            fault_detected = True
        else:
            signal_data = base_signal
            fault_detected = False
            
        suppressed_signal = signal_data * 0.7 if dampers_enabled and fault_detected else signal_data
        
        # Статический график
        fig = go.Figure()
        fig.add_trace(go.Scatter(y=signal_data, mode='lines', name='Vibration'))
        if dampers_enabled and fault_detected:
            fig.add_trace(go.Scatter(y=suppressed_signal, mode='lines', name='Suppressed'))
        st.plotly_chart(fig, use_container_width=True)
        
        # Инженерная панель
        show_engineering_panel(signal_data, suppressed_signal, fault_detected, 
                             severity, fault_type, dampers_enabled)

# Бизнес-метрики (всегда показываем)
st.markdown("---")
st.subheader("📈 Business Impact Estimation")

col_cost, col_impact = st.columns(2)

with col_cost:
    downtime_cost = st.number_input("Estimated hourly downtime cost ($)", 
                                  min_value=1000, value=10000, step=1000)

with col_impact:
    prevented_hours = severity * 8
    potential_savings = downtime_cost * prevented_hours
    system_cost = 120000
    
    st.metric("Potential downtime prevented", f"{prevented_hours} hours")
    st.metric("Estimated savings", f"${potential_savings:,.0f}")
    if system_cost > 0:
        st.metric("ROI multiplier", f"{potential_savings/system_cost:.1f}x")

# Technology Stack
with st.expander("🔧 Technology Stack"):
    st.markdown("""
    **Core Technologies:**
    - Real-time signal processing
    - ML Anomaly Detection  
    - Active Vibration Control
    - Industrial Hardware integration
    
    **Performance Metrics:**
    - Response time: <100 ms
    - Fault detection accuracy: >95%
    - ROI: >2000% from first prevented incident
    """)

# Footer
st.markdown("---")
st.markdown("**Operational Excellence, Delivered** | © 2024 All rights reserved")
