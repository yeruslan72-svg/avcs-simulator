import streamlit as st
import numpy as np
import plotly.graph_objects as go
from datetime import datetime

# Настройка страницы
st.set_page_config(page_title="AVCS DNA Simulator | Engineering Panel", layout="wide")
st.title("🛠️ AVCS DNA Technology Simulator - Engineering Panel")
st.markdown("""
**Operational Excellence, Delivered** - Real-time industrial monitoring with full engineering visibility
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
    
    # Инженерные настройки
    with st.expander("⚙️ Engineering Settings"):
        sample_rate = st.number_input("Sample Rate (Hz)", 1000, 50000, 10000)
        buffer_size = st.number_input("Buffer Size", 256, 4096, 1000)
        num_sensors = st.selectbox("Number of Sensors", [1, 2, 4, 8], index=2)
    
    run_simulation = st.button("▶️ Run Simulation", type="primary")

with col2:
    st.subheader("📊 Simulation Output")
    
    if run_simulation:
        # Генерация сигнала (упрощенная версия)
        t = np.linspace(0, 0.1, buffer_size)
        base_signal = np.sin(2 * np.pi * 50 * t) + 0.1 * np.random.randn(buffer_size)
        
        # Моделирование неисправности
        if "Bearing_Fault" in fault_type:
            impulses = np.random.poisson(0.01 * severity, buffer_size) * severity * 0.3
            signal_data = base_signal + impulses
            fault_detected = True
        elif fault_type == "Imbalance":
            signal_data = base_signal * (1 + 0.3 * severity * np.sin(2 * np.pi * 50 * t))
            fault_detected = True
        else:
            signal_data = base_signal
            fault_detected = False

        # Расчет фич (упрощенный)
        rms = np.sqrt(np.mean(signal_data**2))
        pkpk = np.ptp(signal_data)
        crest = np.max(np.abs(signal_data)) / rms if rms > 0 else 0
        centroid = 50 + severity * 10  # Упрощенный расчет
        
        # Визуализация
        fig = go.Figure()
        fig.add_trace(go.Scatter(y=signal_data, mode='lines', name='Vibration', line=dict(color='blue')))
        fig.update_layout(title="Raw Vibration Signal", xaxis_title="Samples", yaxis_title="Amplitude")
        st.plotly_chart(fig, use_container_width=True)

        # ==================== ИНЖЕНЕРНАЯ ПАНЕЛЬ ====================
        st.subheader("🔧 Engineering Panel - Real-time Diagnostics")
        
        # Создаем структуры данных как в TwinCAT коде
        class ST_Features_V2:
            def __init__(self, rms, pkpk, crest, centroid, fault_type, severity, confidence):
                self.rms = rms
                self.pkpk = pkpk
                self.crest = crest
                self.centroid = centroid
                self.fault_type = fault_type
                self.severity = severity
                self.confidence = confidence
                self.timestamp = datetime.now()
        
        # Создаем диагностическую структуру
        if fault_detected:
            confidence = min(0.3 + severity * 0.15, 0.95)
            diagnosis = ST_Features_V2(rms, pkpk, crest, centroid, fault_type, severity, confidence)
        else:
            diagnosis = ST_Features_V2(rms, pkpk, crest, centroid, "Normal", 0, 0.98)
        
        # Показываем данные в инженерном формате
        col_eng1, col_eng2, col_eng3 = st.columns(3)
        
        with col_eng1:
            st.markdown("**📈 Time-domain Features**")
            st.metric("RMS", f"{diagnosis.rms:.4f}")
            st.metric("Peak-to-Peak", f"{diagnosis.pkpk:.3f}")
            st.metric("Crest Factor", f"{diagnosis.crest:.2f}")
            
        with col_eng2:
            st.markdown("**📊 Frequency-domain Features**")
            st.metric("Spectral Centroid", f"{diagnosis.centroid:.1f} Hz")
            st.metric("Dominant Frequency", "85.0 Hz")  # Фиксировано для демо
            st.metric("Spectral Kurtosis", f"{severity * 0.5:.2f}")
            
        with col_eng3:
            st.markdown("**⚡ System Diagnosis**")
            fault_color = "🟢" if diagnosis.fault_type == "Normal" else "🔴"
            st.metric("Fault Type", f"{fault_color} {diagnosis.fault_type}")
            st.metric("Severity", diagnosis.severity)
            st.metric("Confidence", f"{diagnosis.confidence:.1%}")
        
        # Дополнительная инженерная информация
        with st.expander("🔍 Detailed Engineering Data"):
            st.code(f"""
// ST_SystemConfig (TwinCAT Structure)
nSampleRate_Hz: {sample_rate}
nBufferSize: {buffer_size}  
nNumSensors: {num_sensors}
nNumFeatures: 12

// ST_Features_V2 (Current Sensor 1)
rRMS: {diagnosis.rms:.6f}
rPeakToPeak: {diagnosis.pkpk:.6f}
rCrestFactor: {diagnosis.crest:.4f}
rSpectralCentroid: {diagnosis.centroid:.2f}

// ST_Diagnosis
FaultType: {diagnosis.fault_type}
Severity: {diagnosis.severity}
Confidence: {diagnosis.confidence:.3f}
Timestamp: {diagnosis.timestamp.strftime('%Y-%m-%d %H:%M:%S.%f')}
            """, language='cpp')
        
        # Визуализация решения системы
        st.markdown("**🎯 System Decision Logic**")
        
        if fault_detected and dampers_enabled:
            if severity >= 4:
                force = 8000
                action = "🟥 SEVERE FAULT - Full damping (8000N)"
            elif severity >= 2:
                force = 4000  
                action = "🟨 MILD FAULT - Moderate damping (4000N)"
            else:
                force = 1000
                action = "🟦 MINOR ISSUE - Light damping (1000N)"
                
            st.success(f"**{action}** - Confidence: {diagnosis.confidence:.1%}")
            
            # График демпфирования
            fig_force = go.Figure()
            fig_force.add_trace(go.Indicator(
                mode = "gauge+number+delta",
                value = force,
                domain = {'x': [0, 1], 'y': [0, 1]},
                title = {'text': "Damper Force Command (N)"},
                gauge = {
                    'axis': {'range': [None, 8000]},
                    'bar': {'color': "red"},
                    'steps': [
                        {'range': [0, 1000], 'color': "lightgray"},
                        {'range': [1000, 4000], 'color': "yellow"},
                        {'range': [4000, 8000], 'color': "red"}]
                }
            ))
            st.plotly_chart(fig_force, use_container_width=True)
        else:
            st.info("🟢 NORMAL OPERATION - Monitoring mode (500N baseline)")

# Бизнес-метрики и CTA (оставляем без изменений)
st.markdown("---")
# ... остальной код с бизнес-метриками ...
