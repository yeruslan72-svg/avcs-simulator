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
    dominant_freqs = freqs[dominant_idx] * 1000
    
    return {
        'rms': rms,
        'pkpk': pkpk,
        'crest': crest,
        'variance': variance,
        'centroid': abs(centroid * 1000),
        'dominant_freqs': dominant_freqs[:2],
        'kurtosis': np.mean(magnitude**4) / (np.mean(magnitude**2)**2) - 3 if np.mean(magnitude**2) > 0 else 0
    }

def simulate_dampers(signal_data, fault_detected, severity, enabled=True):
    """Полная модель демпферов - ВОЗВРАЩАЕТ МАССИВ!"""
    n = len(signal_data)
    if not enabled or not fault_detected:
        # Легкое демпфирование - массив из 500
        return signal_data * 0.98, np.full(n, 500)
    
    # Активное подавление - массив сил
    damper_force_value = min(8000, severity * 1600)
    damper_force = np.full(n, damper_force_value)  # Массив одинаковых значений
    suppression_factor = np.exp(-0.3 * damper_force_value / 8000)
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
        
    with col_eng2:
        st.markdown("**📊 Frequency-domain Features**")
        st.metric("Spectral Centroid", f"{features['centroid']:.1f} Hz")
        if len(features['dominant_freqs']) > 0:
            st.metric("Dominant Freq 1", f"{features['dominant_freqs'][0]:.1f} Hz")
        if len(features['dominant_freqs']) > 1:
            st.metric("Dominant Freq 2", f"{features['dominant_freqs'][1]:.1f} Hz")
        
    with col_eng3:
        st.markdown("**⚡ System Diagnosis**")
        fault_color = "🟢" if not fault_detected else "🔴"
        confidence = 0.98 if not fault_detected else min(0.3 + severity * 0.15, 0.95)
        st.metric("Fault Type", f"{fault_color} {fault_type}")
        st.metric("Severity", severity)
        st.metric("Confidence", f"{confidence:.1%}")
        st.metric("Vibration Reduction", f"{vibration_reduction:.1f}%")

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
    
    show_animation = st.checkbox("**Show Live Animation**", value=True)
    
    run_simulation = st.button("▶️ Start Live Simulation", type="primary")

# Основная логика
if run_simulation:
    if show_animation:
        # АНИМАЦИОННАЯ ВЕРСИЯ
        animation_placeholder = st.empty()
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        num_frames = 10  # Уменьшил для скорости
        time_points = np.linspace(0, 0.1, 500)  # Уменьшил размер буфера
        
        for frame in range(num_frames):
            progress = (frame + 1) / num_frames
            progress_bar.progress(progress)
            status_text.text(f"🎬 Live Simulation: Frame {frame+1}/{num_frames}")
            
            # Генерация сигнала
            base_frequency = 50
            base_signal = np.sin(2 * np.pi * base_frequency * time_points)
            base_signal += 0.1 * np.random.randn(len(time_points))
            
            # Моделирование неисправности
            if fault_type == "Normal Operation":
                signal_data = base_signal
                fault_detected = False
                impulses = np.zeros_like(time_points)
            elif "Bearing_Fault" in fault_type:
                impulse_prob = 0.005 * severity  # Увеличил вероятность
                impulses = (np.random.rand(len(time_points)) < impulse_prob).astype(float) * severity * 0.8
                signal_data = base_signal + impulses
                fault_detected = True
            elif fault_type == "Imbalance":
                imbalance_effect = 0.3 * severity
                signal_data = base_signal * (1 + imbalance_effect * np.sin(2 * np.pi * 50 * time_points))
                fault_detected = severity >= 1
                impulses = np.zeros_like(time_points)
            elif fault_type == "Misalignment":
                harmonic_2x = 0.4 * severity * np.sin(2 * np.pi * 100 * time_points)
                signal_data = base_signal + harmonic_2x
                fault_detected = severity >= 1
                impulses = np.zeros_like(time_points)

            # Демпферы - ТЕПЕРЬ ВОЗВРАЩАЕТ МАССИВ!
            suppressed_signal, damper_force = simulate_dampers(signal_data, fault_detected, severity, dampers_enabled)
            
            # Визуализация - ТЕПЕРЬ damper_force это МАССИВ!
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=time_points*1000, 
                y=signal_data, 
                mode='lines', 
                name='Vibration', 
                line=dict(color='blue', width=2)
            ))
            
            if "Bearing_Fault" in fault_type:
                fig.add_trace(go.Scatter(
                    x=time_points*1000, 
                    y=impulses, 
                    mode='lines', 
                    name='Bearing Impacts', 
                    line=dict(color='orange', width=3)
                ))
            
            if dampers_enabled and fault_detected:
                fig.add_trace(go.Scatter(
                    x=time_points*1000, 
                    y=suppressed_signal, 
                    mode='lines', 
                    name='Suppressed', 
                    line=dict(color='green', width=2)
                ))
                
                # ТЕПЕРЬ damper_force это МАССИВ - ошибки нет!
                fig.add_trace(go.Scatter(
                    x=time_points*1000, 
                    y=damper_force/50,  # Масштабируем для визуализации
                    mode='lines', 
                    name='Damper Force/50', 
                    line=dict(color='red', width=2, dash='dot'),
                    yaxis='y2'
                ))
                
                fig.update_layout(
                    yaxis2=dict(
                        title="Damper Force (N/50)",
                        overlaying='y',
                        side='right'
                    )
                )
            
            # Статус аномалии
            status_color = "green" if not fault_detected else "red"
            status_text = "🟢 NORMAL" if not fault_detected else "🔴 FAULT DETECTED"
            
            fig.add_annotation(
                x=0.02, y=0.98, xref="paper", yref="paper",
                text=status_text,
                showarrow=False, 
                bgcolor="white", 
                bordercolor=status_color,
                borderwidth=2,
                font=dict(color=status_color, size=14)
            )
            
            fig.update_layout(
                title=f"Live Vibration Monitoring - Frame {frame+1}/{num_frames}", 
                height=400,
                showlegend=True,
                xaxis_title="Time (milliseconds)",
                yaxis_title="Vibration Amplitude"
            )
            
            animation_placeholder.plotly_chart(fig, use_container_width=True)
            time.sleep(0.5)  # Фиксированная задержка
        
        progress_bar.empty()
        status_text.success("✅ Live simulation completed!")
        
        # Финальный анализ
        features = calculate_features(signal_data)
        show_engineering_panel(signal_data, suppressed_signal, fault_detected, 
                             severity, fault_type, dampers_enabled, features)
        
    else:
        # СТАТИЧЕСКАЯ ВЕРСИЯ
        t = np.linspace(0, 0.1, 1000)
        base_signal = np.sin(2 * np.pi * 50 * t) + 0.1 * np.random.randn(1000)
        
        if fault_type == "Normal Operation":
            signal_data = base_signal
            fault_detected = False
        elif "Bearing_Fault" in fault_type:
            impulses = (np.random.rand(1000) < 0.01 * severity).astype(float) * severity * 0.8
            signal_data = base_signal + impulses
            fault_detected = True
        else:
            signal_data = base_signal * (1 + 0.3 * severity * np.sin(2 * np.pi * 50 * t))
            fault_detected = severity >= 1
            
        suppressed_signal, damper_force = simulate_dampers(signal_data, fault_detected, severity, dampers_enabled)
        features = calculate_features(signal_data)
        
        # График
        fig = go.Figure()
        color = "green" if not fault_detected else "red"
        fig.add_trace(go.Scatter(y=signal_data, mode='lines', name='Vibration', line=dict(color=color)))
        
        if dampers_enabled and fault_detected:
            fig.add_trace(go.Scatter(y=suppressed_signal, mode='lines', name='Suppressed', line=dict(color='blue')))
        
        st.plotly_chart(fig, use_container_width=True)
        show_engineering_panel(signal_data, suppressed_signal, fault_detected, 
                             severity, fault_type, dampers_enabled, features)

# Бизнес-метрики
if not run_simulation:
    st.subheader("📈 Business Impact Estimation")
    st.info("Run simulation to see ROI calculations based on fault severity")

# Technology Stack
with st.expander("🔧 Technology Stack"):
    st.markdown("""
    **Industrial-Grade Vibration Monitoring System**
    - Real-time signal processing at 10kHz
    - Machine Learning anomaly detection
    - Active vibration control with MR dampers
    - >2000% ROI from prevented downtime
    """)

st.markdown("---")
st.markdown("**Operational Excellence, Delivered** | © 2024 AVCS DNA Technology Simulator v3.2")
