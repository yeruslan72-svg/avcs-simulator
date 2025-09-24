import streamlit as st
import numpy as np
from scipy import signal
import plotly.graph_objects as go

# Настройка страницы с брендингом
st.set_page_config(
    page_title="AVCS DNA Simulator | Operational Excellence, Delivered", 
    layout="wide",
    page_icon="⚙️"
)

st.title("🛠️ AVCS DNA Technology Simulator")
st.markdown("""
**Operational Excellence, Delivered** presents an interactive demonstration of our Machine Learning-driven Active Vibration Control System.
Experience how we detect AND suppress faults in real-time on FPSO rotating equipment.
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
    
    dampers_enabled = st.checkbox("**Enable Active Dampers**", value=True)
    
    run_simulation = st.button("▶️ Run Simulation", type="primary")

with col2:
    st.subheader("Simulation Output")

    if run_simulation:
        # [ОСТАВЛЯЕМ ВСЮ ЛОГИКУ СИМУЛЯТОРА БЕЗ ИЗМЕНЕНИЙ]
        # Генерация сигнала вибрации
        sample_rate = 10000
        duration = 0.1
        t = np.linspace(0, duration, int(sample_rate * duration))
        base_signal = np.sin(2 * np.pi * 50 * t) + 0.1 * np.random.randn(len(t))

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

        # Модель работы демпферов
        if dampers_enabled:
            if fault_detected:
                damper_response_time = 0.02
                response_samples = int(damper_response_time * sample_rate)
                
                damper_force = np.zeros_like(t)
                for i in range(len(t)):
                    if i > response_samples:
                        damper_force[i] = min(8000, severity * 1600 * (1 - np.exp(-i/response_samples)))
                
                suppression_factor = np.exp(-0.5 * damper_force/8000)
                suppressed_signal = signal_data * suppression_factor
                
            else:
                damper_force = 500 * np.ones_like(t)
                suppressed_signal = signal_data * 0.95
        else:
            damper_force = np.zeros_like(t)
            suppressed_signal = signal_data

        # Визуализация результатов
        fig = go.Figure()
        
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
            
            fig.add_trace(go.Scatter(
                y=damper_force/20,
                mode='lines', 
                name='Damper Force (N/20)', 
                line=dict(color='red', width=2, dash='dot'),
                yaxis='y2'
            ))
        
        fig.update_layout(
            title="Operational Excellence, Delivered | Vibration Control System Response",
            xaxis_title="Time (samples)",
            yaxis_title="Vibration Amplitude",
            yaxis2=dict(
                title="Damper Force (N/20)",
                overlaying='y',
                side='right'
            )
        )
        
        st.plotly_chart(fig, use_container_width=True)

        # Индикаторы системы
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

        # Технические метрики
        st.subheader("Technical Metrics")
        col_rms, col_crest, col_peak = st.columns(3)
        
        original_rms = np.sqrt(np.mean(signal_data**2))
        suppressed_rms = np.sqrt(np.mean(suppressed_signal**2)) if dampers_enabled else original_rms
        
        col_rms.metric("RMS Vibration", f"{original_rms:.4f}", f"{-((original_rms - suppressed_rms)/original_rms*100):.1f}%")
        col_crest.metric("Crest Factor", f"{np.max(np.abs(signal_data))/original_rms:.2f}")
        col_peak.metric("Peak Reduction", f"{np.max(np.abs(suppressed_signal))/np.max(np.abs(signal_data))*100:.1f}%")

        # Business Impact Calculator
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

        # Technology Stack
        with st.expander("🔧 Under the Hood: AVCS DNA Technology Stack by Operational Excellence, Delivered"):
            st.markdown("""
            **Core Technologies:**
            - **Real-time signal processing**: Scipy, NumPy
            - **ML Anomaly Detection**: Isolation Forest algorithm  
            - **Feature Extraction**: RMS, Kurtosis, Crest Factor
            - **Active Vibration Control**: MR dampers (0-8000N, <100ms response)
            - **Industrial Hardware**: LORD dampers, PCB sensors, Beckhoff PLCs
            
            **Performance Metrics:**
            - Response time: <100 ms
            - Fault detection accuracy: >95%
            - Vibration reduction: up to 80%
            - ROI: >2000% from first prevented incident
            
            *Developed by Yeruslan Chihachyov, Founder & FSO Operations & Reliability Architect*
            """)

# Call-to-Action с брендингом
st.markdown("---")
st.subheader("🚀 Ready to Deploy AVCS DNA on Your Equipment?")

st.markdown("""
**Operational Excellence, Delivered** provides proven engineering solutions to eliminate unplanned downtime on FSO/FPSO vessels. 
Our flagship AVC DNA System is available for outright sale and integration with performance-backed guarantees.
""")

cta_col1, cta_col2, cta_col3 = st.columns(3)

with cta_col1:
    st.markdown("**📞 Schedule Technical Briefing**")
    st.markdown("""
    - Live demo with your operational data
    - Custom ROI calculation for your fleet
    - Integration planning and timeline
    """)

with cta_col2:
    st.markdown("**📧 Contact Our Team**")
    st.markdown("""
    **Email:** yeruslan@operationalexcellence.com  
    **LinkedIn:** Yeruslan Chihachyov  
    **Website:** operationalexcellence.com *(coming soon)*
    """)

with cta_col3:
    st.markdown("**📚 Technical Resources**")
    st.markdown("""
    - Download Technical Specification PDF
    - View Case Studies and ROI Analysis
    - Request Integration Guide
    - Schedule Pilot Project Discussion
    """)

st.markdown("---")
st.markdown("""
**Operational Excellence, Delivered** | *Bridging Frontline Experience with Cutting-Edge AVC Technology*  
© 2024 All rights reserved. AVCS DNA Technology Simulator v2.2 | Delivering >2000% ROI & Eliminating Unplanned Downtime
""")
