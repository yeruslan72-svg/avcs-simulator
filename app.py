import streamlit as st
import numpy as np
from scipy import signal
from scipy.stats import kurtosis
import plotly.graph_objects as go

# Настройка страницы
st.set_page_config(page_title="AVCS Simulator", layout="wide")
st.title("🛠️ AVCS Technology Simulator")
st.markdown("""
**Experience the power of Machine Learning-driven Active Vibration Control.**
This simulator demonstrates how our system detects faults in real-time on FPSO rotating equipment.
""")

# Создаем две колонки для компактности
col1, col2 = st.columns([1, 2])

with col1:
    st.subheader("Configuration")
    # 1. Выбор типа неисправности
    fault_type = st.selectbox(
        "**Select Fault Type**",
        ["Normal Operation", "Bearing Fault (Mild)", "Bearing Fault (Severe)", "Imbalance", "Misalignment"]
    )

    # 2. Ползунок для тяжести неисправности
    severity = st.slider("**Fault Severity**", 1, 5, 1)

    # 3. Кнопка запуска симуляции
    run_simulation = st.button("▶️ Run Simulation", type="primary")

with col2:
    st.subheader("Simulation Output")

    if run_simulation:
        # Генерация сигнала
        sample_rate = 10000
        duration = 0.1
        t = np.linspace(0, duration, int(sample_rate * duration))
        base_signal = np.sin(2 * np.pi * 50 * t) + 0.1 * np.random.randn(len(t))

        # Имитация выбранной неисправности
        if fault_type == "Normal Operation":
            signal_data = base_signal
        elif "Bearing Fault" in fault_type:
            impulse_prob = 0.001 * severity
            impulses = (np.random.rand(len(t)) < impulse_prob).astype(float) * severity * 0.5
            signal_data = base_signal + impulses
        elif fault_type == "Imbalance":
            signal_data = base_signal * (1 + 0.1 * severity * np.sin(2 * np.pi * 50 * t))
        elif fault_type == "Misalignment":
        # Усиливаем эффект misalignment: добавляем мощную гармонику 2X
            misalignment_effect = 0.3 * severity * np.sin(2 * np.pi * 100 * t + np.pi/4)
            signal_data = base_signal + misalignment_effect

        # Извлечение признаков
        rms = np.sqrt(np.mean(signal_data**2))
        peak_to_peak = np.ptp(signal_data)
        crest_factor = np.max(np.abs(signal_data)) / rms if rms > 0 else 0

        # Визуализация
        fig = go.Figure()
        fig.add_trace(go.Scatter(y=signal_data, mode='lines', name='Vibration Signal', line=dict(color='blue')))
        fig.update_layout(title="Raw Vibration Signal", xaxis_title="Samples", yaxis_title="Amplitude")
        st.plotly_chart(fig, use_container_width=True)

        # Имитация работы ML-модели
        if crest_factor > 3.0:
            st.error(f"🚨 **ANOMALY DETECTED!**")
            st.success(f"**Diagnosis:** {fault_type} (Confidence: {severity/5*100:.0f}%)")
            st.markdown("> **On a real AVCS:** MR dampers would be activated to suppress vibration.")
        else:
            st.success("✅ **SYSTEM NORMAL**")
            st.markdown("> **On a real AVCS:** Dampers in adaptive mode.")

        # Показать извлеченные фичи
        st.subheader("Extracted Features")
        col_metric1, col_metric2, col_metric3 = st.columns(3)
        col_metric1.metric("RMS", f"{rms:.4f}")
        col_metric2.metric("Peak-to-Peak", f"{peak_to_peak:.2f}")
        col_metric3.metric("Crest Factor", f"{crest_factor:.2f}")

st.markdown("---")
st.caption("© Yeruslan Technologies | Active Vibration Control System (AVCS) Simulator")
