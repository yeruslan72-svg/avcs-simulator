import streamlit as st
import numpy as np
import plotly.graph_objects as go
import plotly.subplots as sp
from datetime import datetime
import time

# Настройка страницы
st.set_page_config(page_title="AVCS DNA Simulator | Live Animation", layout="wide")
st.title("🛠️ AVCS DNA Technology Simulator - Live Animation")
st.markdown("""
**Operational Excellence, Delivered** - Real-time vibration monitoring with live animation
""")

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

# Место для анимации
animation_placeholder = st.empty()
engineering_placeholder = st.empty()

if run_simulation and show_animation:
    # Инициализация анимации
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    # Параметры анимации
    num_frames = 50
    time_points = np.linspace(0, 0.1, buffer_size)
    
    for frame in range(num_frames):
        # Обновление прогресса
        progress = (frame + 1) / num_frames
        progress_bar.progress(progress)
        status_text.text(f"🎬 Live Simulation: Frame {frame+1}/{num_frames}")
        
        # Генерация сигнала с "дрейфом" для анимации
        base_frequency = 50 + 2 * np.sin(frame * 0.1)  # Плавное изменение частоты
        base_signal = np.sin(2 * np.pi * base_frequency * time_points)
        
        # Добавляем случайный шум (меняется каждый кадр)
        noise_level = 0.1 + 0.05 * np.sin(frame * 0.2)
        base_signal += noise_level * np.random.randn(buffer_size)
        
        # Моделирование неисправности с анимационными эффектами
        if fault_type == "Normal Operation":
            signal_data = base_signal
            fault_detected = False
            impulses = np.zeros_like(time_points)
            
        elif "Bearing_Fault" in fault_type:
            # Анимированные импульсы (появляются/исчезают)
            impulse_phase = frame * 0.3
            impulse_times = np.arange(0.0, time_points[-1] + 1e-9, 0.02)
            impulses = np.zeros_like(time_points)
            
            for t in impulse_times:
                idx = np.argmin(np.abs(time_points - t))
                decay = np.exp(-80.0 * (time_points - t) ** 2)
                impulse_strength = severity * (0.3 + 0.2 * np.sin(impulse_phase))
                impulses += impulse_strength * decay
                
            signal_data = base_signal + impulses
            fault_detected = severity > 2
            
        elif fault_type == "Imbalance":
            # Анимированный дисбаланс (пульсирующая амплитуда)
            imbalance_strength = 0.3 * severity * (1 + 0.2 * np.sin(frame * 0.4))
            signal_data = base_signal * (1 + imbalance_strength * np.sin(2 * np.pi * base_frequency * time_points))
            
            # Случайные всплески
            spikes = (np.random.rand(buffer_size) < 0.01 * severity).astype(float) * severity * 0.2
            signal_data += spikes
            fault_detected = severity > 1
            
        elif fault_type == "Misalignment":
            # Анимированная расцентровка (меняющаяся гармоника)
            harmonic_strength = 0.5 * severity * (1 + 0.1 * np.sin(frame * 0.3))
            harmonic_2x = harmonic_strength * np.sin(2 * np.pi * 2 * base_frequency * time_points + frame * 0.2)
            
            # Случайные удары
            impacts = (np.random.rand(buffer_size) < 0.005 * severity).astype(float) * severity * 0.4
            signal_data = base_signal + harmonic_2x + impacts
            fault_detected = severity > 1

        # Модель демпферов с анимацией
        if dampers_enabled and fault_detected:
            # Демпферы "включаются" постепенно в анимации
            damper_progress = min(1.0, frame / 10.0)
            max_force = severity * 1600 * damper_progress
            
            damper_force = np.zeros_like(time_points)
            for i in range(len(time_points)):
                if i > 50:  # Задержка отклика
                    damper_force[i] = min(max_force, severity * 1600 * (1 - np.exp(-i/100)))
            
            suppression_factor = np.exp(-0.3 * damper_force/8000)
            suppressed_signal = signal_data * suppression_factor
            
        else:
            damper_force = 500 * np.ones_like(time_points)
            suppressed_signal = signal_data * 0.98  # Легкое демпфирование

        # Создание анимированного графика
        fig = go.Figure()
        
        # Основной сигнал с анимационными эффектами
        fig.add_trace(go.Scatter(
            x=time_points * 1000,  # в миллисекундах
            y=signal_data,
            mode='lines',
            name='Vibration Signal',
            line=dict(color='blue', width=2),
            opacity=0.8
        ))
        
        # Импульсы (если есть)
        if "Bearing_Fault" in fault_type:
            fig.add_trace(go.Scatter(
                x=time_points * 1000,
                y=impulses,
                mode='lines',
                name='Bearing Impacts',
                line=dict(color='orange', width=3),
                opacity=0.6
            ))
        
        # Подавленный сигнал (если демпферы активны)
        if dampers_enabled and fault_detected:
            fig.add_trace(go.Scatter(
                x=time_points * 1000,
                y=suppressed_signal,
                mode='lines',
                name='Suppressed Vibration',
                line=dict(color='green', width=3),
                opacity=0.7
            ))
            
            # Сила демпфирования (второстепенная ось)
            fig.add_trace(go.Scatter(
                x=time_points * 1000,
                y=damper_force/50,  # Масштабирование для визуализации
                mode='lines',
                name='Damper Force (N/50)',
                line=dict(color='red', width=2, dash='dot'),
                yaxis='y2',
                opacity=0.6
            ))

        # Настройка анимационного графика
        fig.update_layout(
            title=f"🎬 Live Vibration Monitoring - Frame {frame+1}/{num_frames}",
            xaxis_title="Time (milliseconds)",
            yaxis_title="Vibration Amplitude",
            yaxis2=dict(
                title="Damper Force (N/50)",
                overlaying='y',
                side='right',
                range=[0, 200]  # Фиксированный диапазон для силы
            ),
            height=400,
            showlegend=True,
            template="plotly_white"
        )
        
        # Добавляем индикаторы состояния
        fig.add_annotation(
            x=0.02, y=0.98,
            xref="paper", yref="paper",
            text=f"Frequency: {base_frequency:.1f} Hz<br>Frame: {frame+1}",
            showarrow=False,
            bgcolor="white",
            bordercolor="black",
            borderwidth=1
        )
        
        if fault_detected:
            fig.add_annotation(
                x=0.98, y=0.98,
                xref="paper", yref="paper",
                text="🚨 FAULT DETECTED",
                showarrow=False,
                bgcolor="red",
                font=dict(color="white")
            )

        # Отображаем анимацию
        animation_placeholder.plotly_chart(fig, use_container_width=True)
        
        # Небольшая задержка для анимационного эффекта
        time.sleep(0.5 / animation_speed)
    
    # Завершение анимации
    progress_bar.empty()
    status_text.success("✅ Live simulation completed!")
    
    # Показываем итоговую инженерную панель
    with engineering_placeholder.container():
        show_engineering_panel(signal_data, suppressed_signal, fault_detected, 
                             severity, fault_type, dampers_enabled)

else:
    # Статическая версия (как раньше)
    if run_simulation:
        # ... существующий код статической симуляции ...
        pass

def show_engineering_panel(signal_data, suppressed_signal, fault_detected, 
                          severity, fault_type, dampers_enabled):
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

# Остальной код (бизнес-метрики, CTA) остается без изменений
