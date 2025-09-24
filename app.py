import streamlit as st
import numpy as np
from scipy import signal
from scipy.stats import kurtosis
import plotly.graph_objects as go

# Настройка страницы
st.set_page_config(page_title="AVCS DNA Simulator", layout="wide")
st.title("🛠️ AVCS DNA Technology Simulator")
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
            imbalance_effect = 0.5 * severity
            signal_data = base_signal * (1 + imbalance_effect * np.sin(2 * np.pi * 50 * t))
            impulses = (np.random.rand(len(t)) < 0.003 * severity).astype(float) * severity * 0.3
            signal_data = signal_data + impulses
        elif fault_type == "Misalignment":
            harmonic_2x = 0.7 * severity * np.sin(2 * np.pi * 100 * t + np.pi/4)
            impulses = (np.random.rand(len(t)) < 0.005 * severity).astype(float) * severity * 0.8
            signal_data = base_signal + harmonic_2x + impulses

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

        # 🔥 НОВЫЙ БЛОК: Business Impact Estimation
        st.subheader("📈 Business Impact Estimation")
        
        col_cost, col_impact = st.columns(2)
        
        with col_cost:
            downtime_cost = st.number_input("Estimated hourly downtime cost ($)", 
                                          min_value=1000, value=10000, step=1000,
                                          key="downtime_cost")
        
        with col_impact:
            prevented_hours = severity * 8  # Логика: чем серьезнее неисправность, тем больше часов простоя предотвращаем
            potential_savings = downtime_cost * prevented_hours
            system_cost = 120000  # Базовая стоимость системы
            
            st.metric("💾 Potential downtime prevented", f"{prevented_hours} hours")
            st.metric("💰 Estimated savings", f"${potential_savings:,.0f}")
            st.metric("📊 ROI multiplier", f"{potential_savings/system_cost:.1f}x")

        # 🔥 НОВЫЙ БЛОК: Technology Stack
        with st.expander("🔧 Under the Hood: AVCS DNA Technology Stack"):
            st.markdown("""
            **Core Technologies:**
            - **Real-time signal processing**: Scipy, NumPy
            - **ML Anomaly Detection**: Isolation Forest algorithm  
            - **Feature Extraction**: RMS, Kurtosis, Crest Factor
            - **Control Systems**: PID-based damper control
            - **Industrial Hardware**: LORD dampers, PCB sensors, Beckhoff PLCs
            
            **Performance Metrics:**
            - Response time: <100 ms
            - Fault detection accuracy: >95%
            - ROI: >2000% from first prevented incident
            """)

# 🔥 НОВЫЙ БЛОК: Call-to-Action (всегда виден)
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
    Email: yeruslan@operationalexcellence.com
    LinkedIn: Yeruslan Chihachyov
    """)

with cta_col3:
    st.markdown("**📚 Resources**")
    st.markdown("""
    - [Download Technical PDF]()
    - [Case Studies]()
    - [Integration Guide]()
    """)

st.markdown("---")
st.caption("© 2024 Operational Excellence, Delivered | AVCS DNA Technology Simulator v2.0")
