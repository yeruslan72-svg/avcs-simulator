import streamlit as st
import numpy as np
from scipy import signal
import plotly.graph_objects as go

# Настройка страницы
st.set_page_config(page_title="AVCS DNA Simulator", layout="wide")
st.title("🛠️ AVCS DNA Technology Simulator")
st.markdown("""
**Experience Active Vibration Control with Real-time Damper Response**
This simulator demonstrates how our system detects AND suppresses faults in real-time.
""")

# Создаем две колонки
col1, col2 = st.columns([1, 2])

with col1:
    st.subheader("Configuration")
    
    fault_type = st.selectbox(
        "**Select Fault Type**",
        ["Normal Operation", "Bearing Fault (Mild)", "Bearing Fault (Severe)", "Imbalance", "Misalignment"]
    )

    severity = st.slider("**Fault Severity**", 1, 5, 1)
    
    # НОВЫЙ ПАРАМЕТР: Включение демпферов
    dampers_enabled = st.checkbox("**Enable Active Dampers**", value=True)
    
    run_simulation = st.button("▶️ Run Simulation", type="primary")

with col2:
    st.subheader("Simulation Output")

    if run_simulation:
        # Генерация сигнала вибрации
        sample_rate = 10000
        duration = 0.1
        t = np.linspace(0, duration, int(sample_rate * duration))
        base_signal = np.sin(2 * np.pi * 50 * t) + 0.1 * np.random.randn(len(t))

        # Имитация выбранной неисправности
        if fault_type == "Normal Operation":
            signal_data = base_signal
            fault_detected = False
        elif "Bearing Fault" in fault_type:
            impulse_prob = 0.001 * severity
            impulses = (np.random.rand(len(t)) < impulse_prob).astype(float) * severity * 0.5
            signal_data = base_signal + impulses
            fault_detected = severity > 2
        elif fault_type == "Imbalance":
            imbalance_effect = 0.5 * severity
            signal_data = base_signal * (1 + imbalance_effect * np.sin(2 * np.pi * 50 * t))
            impulses = (np.random.rand(len(t)) < 0.003 * severity).astype(float) * severity * 0.3
            signal_data = signal_data + impulses
            fault_detected = severity > 1
        elif fault_type == "Misalignment":
            harmonic_2x = 0.7 * severity * np.sin(2 * np.pi * 100 * t + np.pi/4)
            impulses = (np.random.rand(len(t)) < 0.005 * severity).astype(float) * severity * 0.8
            signal_data = base_signal + harmonic_2x + impulses
            fault_detected = severity > 1

        # НОВАЯ СЕКЦИЯ: МОДЕЛЬ РАБОТЫ ДЕМПФЕРОВ
        if dampers_enabled:
            if fault_detected:
                # АКТИВНОЕ ПОДАВЛЕНИЕ - демпферы работают на полную мощность
                damper_response_time = 0.02  # 20 ms response
                response_samples = int(damper_response_time * sample_rate)
                
                # Моделируем постепенное включение демпферов
                damper_force = np.zeros_like(t)
                for i in range(len(t)):
                    if i > response_samples:
                        damper_force[i] = min(8000, severity * 1600 * (1 - np.exp(-i/response_samples)))
                
                # Эффект подавления вибрации (упрощенная модель)
                suppression_factor = np.exp(-0.5 * damper_force/8000)
                suppressed_signal = signal_data * suppression_factor
                
            else:
                # АДАПТИВНЫЙ РЕЖИМ - легкое демпфирование
                damper_force = 500 * np.ones_like(t)  # Базовая сила 500 Н
                suppressed_signal = signal_data * 0.95  # Легкое подавление
        else:
            # Демпферы отключены
            damper_force = np.zeros_like(t)
            suppressed_signal = signal_data

        # ВИЗУАЛИЗАЦИЯ РЕЗУЛЬТАТОВ
        fig = go.Figure()
        
        # График вибрации
        fig.add_trace(go.Scatter(
            y=signal_data, 
            mode='lines', 
            name='Original Vibration', 
            line=dict(color='blue', width=1)
        ))
        
        if dampers_enabled:
            fig.add_trace(go.Scatter(
                y=suppressed_signal, 
                mode='lines', 
                name='Suppressed Vibration', 
                line=dict(color='green', width=2)
            ))
            
            # График силы демпфирования (в масштабе)
            fig.add_trace(go.Scatter(
                y=damper_force/20,  # Масштабируем для визуализации
                mode='lines', 
                name='Damper Force (N/20)', 
                line=dict(color='red', width=2, dash='dot'),
                yaxis='y2'
            ))
        
        fig.update_layout(
            title="Vibration Control System Response",
            xaxis_title="Time (samples)",
            yaxis_title="Vibration Amplitude",
            yaxis2=dict(
                title="Damper Force (N/20)",
                overlaying='y',
                side='right'
            )
        )
        
        st.plotly_chart(fig, use_container_width=True)

        # ИНДИКАТОРЫ СИСТЕМЫ
        col_status, col_force, col_efficiency = st.columns(3)
        
        with col_status:
            if fault_detected:
                if dampers_enabled:
                    st.error("🚨 **FAULT DETECTED - ACTIVE SUPPRESSION**")
                else:
                    st.error("🚨 **FAULT DETECTED - DAMPERS OFF**")
            else:
                st.success("✅ **SYSTEM NORMAL**")
        
        with col_force:
            max_force = np.max(damper_force) if dampers_enabled else 0
            st.metric("🔧 Max Damper Force", f"{max_force:.0f} N")
        
        with col_efficiency:
            if dampers_enabled and fault_detected:
                vibration_reduction = (1 - np.std(suppressed_signal)/np.std(signal_data)) * 100
                st.metric("📉 Vibration Reduction", f"{vibration_reduction:.1f}%")

        # ТЕХНИЧЕСКИЕ МЕТРИКИ
        st.subheader("Technical Metrics")
        col_rms, col_crest, col_peak = st.columns(3)
        
        original_rms = np.sqrt(np.mean(signal_data**2))
        suppressed_rms = np.sqrt(np.mean(suppressed_signal**2)) if dampers_enabled else original_rms
        
        col_rms.metric("RMS Vibration", f"{original_rms:.4f}", f"{-((original_rms - suppressed_rms)/original_rms*100):.1f}%")
        col_crest.metric("Crest Factor", f"{np.max(np.abs(signal_data))/original_rms:.2f}")
        col_peak.metric("Peak Reduction", f"{np.max(np.abs(suppressed_signal))/np.max(np.abs(signal_data))*100:.1f}%")

# ОСТАЛЬНЫЕ СЕКЦИИ (Business Impact, CTA) остаются без изменений
