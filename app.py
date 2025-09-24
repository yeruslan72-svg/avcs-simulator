import streamlit as st
import numpy as np
import plotly.graph_objects as go
import plotly.subplots as sp
import time
import io
from PIL import Image
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from reportlab.lib.units import inch

# ==================== НАСТРОЙКА СТРАНИЦЫ ====================
st.set_page_config(page_title="AVCS DNA Multi-Channel Simulator", layout="wide")
st.title("🏭 AVCS DNA Multi-Channel Industrial Simulator")
st.markdown("""
**Operational Excellence, Delivered** - Multi-sensor vibration control with angular damper configuration
""")

# ==================== КОНФИГУРАЦИЯ ОБОРУДОВАНИЯ ====================
class EquipmentConfig:
    def __init__(self):
        self.sensors = {
            'Motor_End': {'position': 'Motor Side', 'sensitivity': 100, 'channel': 1},
            'Pump_End': {'position': 'Pump Side', 'sensitivity': 100, 'channel': 2}
        }
        self.dampers = {
            'Damper_1': {'position': 'Front-Left', 'angle': 45, 'force_range': (0, 8000)},
            'Damper_2': {'position': 'Front-Right', 'angle': 45, 'force_range': (0, 8000)},
            'Damper_3': {'position': 'Rear-Left', 'angle': 45, 'force_range': (0, 8000)},
            'Damper_4': {'position': 'Rear-Right', 'angle': 45, 'force_range': (0, 8000)}
        }
        self.sample_rates = [1000, 2500, 5000, 10000, 20000]
        self.accelerometer_types = {
            'ICP_603C01': {'sensitivity': 100, 'range': 50, 'freq_range': (0.5, 10000)},
            'PCB_352C23': {'sensitivity': 10, 'range': 500, 'freq_range': (0.2, 10000)},
            'IMI_608A11': {'sensitivity': 1000, 'range': 5, 'freq_range': (1, 2000)}
        }

# ==================== ФУНКЦИИ СИМУЛЯЦИИ ====================
def generate_vibration_signal(time_points, fault_type, severity, sensor_position, base_freq=50):
    base_signal = np.sin(2 * np.pi * base_freq * time_points)
    base_signal += 0.05 * np.random.randn(len(time_points))
    
    if sensor_position == 'Motor_End':
        high_freq = 0.3 * np.sin(2 * np.pi * 200 * time_points)
        base_signal += 0.1 * high_freq
    else:
        low_freq = 0.4 * np.sin(2 * np.pi * 25 * time_points)
        base_signal += 0.15 * low_freq
    
    if fault_type == "Normal Operation":
        return base_signal, np.zeros_like(time_points)
    elif "Bearing_Fault" in fault_type:
        impulse_prob = 0.008 * severity
        impulses = (np.random.rand(len(time_points)) < impulse_prob).astype(float) * severity * 0.6
        return base_signal + impulses, impulses
    elif fault_type == "Imbalance":
        imbalance_effect = 0.4 * severity
        modulated = base_signal * (1 + imbalance_effect * np.sin(2 * np.pi * base_freq * time_points))
        return modulated, np.zeros_like(time_points)
    elif fault_type == "Misalignment":
        harmonic_2x = 0.5 * severity * np.sin(2 * np.pi * 2 * base_freq * time_points + np.pi/4)
        harmonic_3x = 0.3 * severity * np.sin(2 * np.pi * 3 * base_freq * time_points + np.pi/6)
        return base_signal + harmonic_2x + harmonic_3x, np.zeros_like(time_points)
    return base_signal, np.zeros_like(time_points)

def calculate_angular_damper_force(dampers_config, fault_detected, severity, sensor_position):
    forces = {}
    if not fault_detected:
        for damper_id in dampers_config:
            forces[damper_id] = 500
    else:
        base_force = min(8000, severity * 1600)
        for damper_id, config in dampers_config.items():
            angle_factor = np.sin(np.radians(config['angle']))
            position_factor = 1.0
            if 'Front' in config['position'] and sensor_position == 'Motor_End':
                position_factor = 1.2
            elif 'Rear' in config['position'] and sensor_position == 'Pump_End':
                position_factor = 1.2
            forces[damper_id] = base_force * angle_factor * position_factor
    return forces

def apply_damper_suppression(signal_data, damper_forces, time_points):
    total_force = sum(damper_forces.values())
    suppression_factor = np.exp(-0.25 * total_force / 8000)
    response_delay = int(0.02 * len(time_points))
    suppressed_signal = np.copy(signal_data)
    if response_delay < len(signal_data):
        suppressed_signal[response_delay:] = signal_data[response_delay:] * suppression_factor
    return suppressed_signal

def calculate_multi_channel_features(signals_dict):
    features_dict = {}
    for sensor_id, signal_data in signals_dict.items():
        if len(signal_data) == 0:
            features_dict[sensor_id] = {'rms': 0, 'pkpk': 0, 'crest': 0}
            continue
        rms = np.sqrt(np.mean(signal_data**2))
        pkpk = np.ptp(signal_data)
        crest = np.max(np.abs(signal_data)) / rms if rms > 0 else 0
        features_dict[sensor_id] = {'rms': rms, 'pkpk': pkpk, 'crest': crest, 'position': sensor_id}
    return features_dict

def show_business_impact(severity):
    st.subheader("📈 Business Impact Estimation")
    col_cost, col_impact = st.columns(2)
    with col_cost:
        downtime_cost = st.number_input("Estimated hourly downtime cost ($)", 
                                      min_value=1000, value=10000, step=1000, key="downtime_cost")
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
    try:
        buf = io.BytesIO()
        fig.write_image(buf, format="png")
        buf.seek(0)
        img = Image.open(buf)
    except:
        img = None
    post_text = f"""
🚀 **Multi-Channel Predictive Maintenance – AVCS DNA**

**Fault Simulated:** {fault_type}
**Severity Level:** {severity}/5
**Downtime Prevented:** {prevented_hours:.1f} hours
**Estimated Savings:** ${potential_savings:,.0f}
**ROI Multiplier:** {roi:.1f}x

Advanced multi-sensor monitoring with angular damper configuration.
Real-time vibration control across Motor End and Pump End.

#PredictiveMaintenance #AssetIntegrity #Industry40 #ROI #OperationalExcellence
    """.strip()
    return post_text, img

# ==================== НОВАЯ ФУНКЦИЯ: PDF ГЕНЕРАТОР ====================
def generate_pdf_report(fault_type, severity, prevented_hours, potential_savings, roi, 
                       active_sensors, dampers_enabled, sample_rate, accelerometer_type,
                       features_dict, filename="avcs_simulation_report.pdf"):
    """Генератор профессиональных PDF отчетов"""
    c = canvas.Canvas(filename, pagesize=letter)
    width, height = letter

    # Заголовок
    c.setFont("Helvetica-Bold", 16)
    c.drawString(50, height - 50, "AVCS DNA Simulation Report")
    c.setFont("Helvetica", 12)
    c.drawString(50, height - 70, f"Generated on: {time.strftime('%Y-%m-%d %H:%M:%S')}")

    # Конфигурация системы
    c.setFont("Helvetica-Bold", 14)
    c.drawString(50, height - 100, "System Configuration:")
    c.setFont("Helvetica", 10)
    
    y_position = height - 120
    config_lines = [
        f"Fault Type: {fault_type}",
        f"Severity Level: {severity}/5",
        f"Active Sensors: {', '.join(active_sensors)}",
        f"Dampers: {'Enabled' if dampers_enabled else 'Disabled'}",
        f"Sample Rate: {sample_rate} Hz",
        f"Accelerometer: {accelerometer_type}"
    ]
    
    for line in config_lines:
        c.drawString(50, y_position, line)
        y_position -= 15

    # Результаты
    c.setFont("Helvetica-Bold", 14)
    c.drawString(50, y_position - 20, "Simulation Results:")
    c.setFont("Helvetica", 10)
    
    y_position -= 40
    result_lines = [
        f"Potential Downtime Prevented: {prevented_hours} hours",
        f"Estimated Savings: ${potential_savings:,.0f}",
        f"ROI Multiplier: {roi:.1f}x"
    ]
    
    for line in result_lines:
        c.drawString(50, y_position, line)
        y_position -= 15

    # Технические параметры
    c.setFont("Helvetica-Bold", 14)
    c.drawString(50, y_position - 20, "Technical Parameters:")
    c.setFont("Helvetica", 10)
    
    y_position -= 40
    for sensor_id, features in features_dict.items():
        c.drawString(50, y_position, f"{sensor_id}:")
        c.drawString(200, y_position, f"RMS: {features['rms']:.4f} | PkPk: {features['pkpk']:.3f} | Crest: {features['crest']:.2f}")
        y_position -= 15

    # Заключение
    c.setFont("Helvetica-Bold", 12)
    c.drawString(50, y_position - 30, "Conclusion:")
    c.setFont("Helvetica", 10)
    c.drawString(50, y_position - 45, "AVCS DNA system demonstrates effective vibration control with significant ROI potential.")
    
    c.showPage()
    c.save()

# ==================== ИНТЕРФЕЙС ====================
config = EquipmentConfig()

col1, col2 = st.columns([1, 2])

with col1:
    st.subheader("🎛️ Equipment Configuration")
    
    # ВСЕ СУЩЕСТВУЮЩИЕ ФИШКИ СОХРАНЕНЫ:
    show_animation = st.checkbox("Show Live Animation", value=True, key="animation")
    if show_animation:
        animation_speed = st.slider("Animation Speed", 1, 5, 3, key="anim_speed")
    
    st.markdown("**📡 Sensor Configuration**")
    sensor_motor = st.checkbox("Motor End Sensor", value=True, key="sensor_motor")
    sensor_pump = st.checkbox("Pump End Sensor", value=True, key="sensor_pump")
    
    accelerometer_type = st.selectbox("Accelerometer Type", 
                                     list(config.accelerometer_types.keys()), key="accel_type")
    sample_rate = st.selectbox("Sample Rate (Hz)", config.sample_rates, index=3, key="sample_rate")
    
    st.markdown("**⚡ Fault Configuration**")
    fault_type = st.selectbox("Fault Type", [
        "Normal Operation", "Bearing_Fault_Mild", "Bearing_Fault_Severe", 
        "Imbalance", "Misalignment"
    ], key="fault_type")
    
    severity = st.slider("Fault Severity", 1, 5, 1, key="severity")
    dampers_enabled = st.checkbox("Enable Active Dampers", value=True, key="dampers")
    
    run_simulation = st.button("▶️ Start Multi-Channel Simulation", type="primary")

with col2:
    st.subheader("📊 Multi-Channel Monitoring")
    
    if run_simulation:
        active_sensors = []
        if sensor_motor: active_sensors.append('Motor_End')
        if sensor_pump: active_sensors.append('Pump_End')
        
        if not active_sensors:
            st.error("❌ Please select at least one sensor")
            st.stop()
        
        time_points = np.linspace(0, 0.1, int(sample_rate * 0.1))
        
        # СОХРАНЯЕМ ВСЮ ЛОГИКУ АНИМАЦИИ/СТАТИКИ
        if show_animation:
            animation_placeholder = st.empty()
            progress_bar = st.progress(0)
            status_display = st.empty()
            
            num_frames = 10
            final_signals = {}
            final_suppressed = {}
            final_damper_forces = {}
            
            for frame in range(num_frames):
                progress = (frame + 1) / num_frames
                progress_bar.progress(progress)
                status_display.text(f"🎬 Multi-Channel Simulation: Frame {frame+1}/{num_frames}")
                
                signals = {}
                suppressed_signals = {}
                damper_forces_history = {}
                
                for sensor_id in active_sensors:
                    signal_data, impulses = generate_vibration_signal(
                        time_points, fault_type, severity, sensor_id
                    )
                    signals[sensor_id] = signal_data
                    
                    fault_detected = fault_type != "Normal Operation"
                    damper_forces = calculate_angular_damper_force(
                        config.dampers, fault_detected, severity, sensor_id
                    )
                    damper_forces_history[sensor_id] = damper_forces
                    
                    if dampers_enabled and fault_detected:
                        suppressed_signals[sensor_id] = apply_damper_suppression(
                            signal_data, damper_forces, time_points
                        )
                    else:
                        suppressed_signals[sensor_id] = signal_data
                
                if frame == num_frames - 1:
                    final_signals = signals
                    final_suppressed = suppressed_signals
                    final_damper_forces = damper_forces_history
                
                fig = sp.make_subplots(
                    rows=len(active_sensors), cols=1,
                    subplot_titles=[f"{sensor_id} - {config.sensors[sensor_id]['position']}" 
                                   for sensor_id in active_sensors],
                    vertical_spacing=0.1
                )
                
                for i, sensor_id in enumerate(active_sensors):
                    row = i + 1
                    fig.add_trace(
                        go.Scatter(x=time_points*1000, y=signals[sensor_id],
                                  mode='lines', name=f'{sensor_id} Original',
                                  line=dict(color='blue', width=2)),
                        row=row, col=1
                    )
                    
                    if dampers_enabled and fault_detected:
                        fig.add_trace(
                            go.Scatter(x=time_points*1000, y=suppressed_signals[sensor_id],
                                      mode='lines', name=f'{sensor_id} Suppressed',
                                      line=dict(color='green', width=2)),
                            row=row, col=1
                        )
                
                fig.update_layout(height=300 * len(active_sensors), 
                                title_text=f"Multi-Channel Monitoring - Frame {frame+1}/{num_frames}")
                fig.update_xaxes(title_text="Time (ms)")
                fig.update_yaxes(title_text="Amplitude")
                
                animation_placeholder.plotly_chart(fig, use_container_width=True)
                time.sleep(0.5 / animation_speed)
            
            progress_bar.empty()
            status_display.success("✅ Multi-channel simulation completed!")
            current_fig = fig
            
        else:
            st.info("📊 Static Analysis Mode")
            final_signals = {}
            final_suppressed = {}
            final_damper_forces = {}
            
            for sensor_id in active_sensors:
                signal_data, impulses = generate_vibration_signal(
                    time_points, fault_type, severity, sensor_id
                )
                final_signals[sensor_id] = signal_data
                
                fault_detected = fault_type != "Normal Operation"
                damper_forces = calculate_angular_damper_force(
                    config.dampers, fault_detected, severity, sensor_id
                )
                final_damper_forces[sensor_id] = damper_forces
                
                if dampers_enabled and fault_detected:
                    final_suppressed[sensor_id] = apply_damper_suppression(
                        signal_data, damper_forces, time_points
                    )
                else:
                    final_suppressed[sensor_id] = signal_data
            
            fig_static = sp.make_subplots(
                rows=len(active_sensors), cols=1,
                subplot_titles=[f"{sensor_id} - {config.sensors[sensor_id]['position']}" 
                               for sensor_id in active_sensors],
                vertical_spacing=0.1
            )
            
            for i, sensor_id in enumerate(active_sensors):
                row = i + 1
                color = 'green' if fault_type == "Normal Operation" else 'red'
                fig_static.add_trace(
                    go.Scatter(x=time_points*1000, y=final_signals[sensor_id],
                              mode='lines', name=f'{sensor_id} Vibration',
                              line=dict(color=color, width=2)),
                    row=row, col=1
                )
                
                if dampers_enabled and fault_type != "Normal Operation":
                    fig_static.add_trace(
                        go.Scatter(x=time_points*1000, y=final_suppressed[sensor_id],
                                  mode='lines', name=f'{sensor_id} Suppressed',
                                  line=dict(color='blue', width=2)),
                        row=row, col=1
                    )
            
            fig_static.update_layout(height=300 * len(active_sensors), 
                                   title_text="Multi-Channel Static Analysis")
            st.plotly_chart(fig_static, use_container_width=True)
            current_fig = fig_static
        
        # ВСЕ СУЩЕСТВУЮЩИЕ ПАНЕЛИ СОХРАНЕНЫ:
        st.subheader("🔧 Damper Control Panel")
        damper_cols = st.columns(4)
        motor_forces = final_damper_forces.get('Motor_End', final_damper_forces.get(active_sensors[0], {}))
        
        for i, damper_id in enumerate(config.dampers.keys()):
            with damper_cols[i]:
                config_data = config.dampers[damper_id]
                force = motor_forces.get(damper_id, 0)
                st.metric(f"{damper_id}", f"{force:.0f} N", f"Angle: {config_data['angle']}°")
        
        features_dict = calculate_multi_channel_features(final_signals)
        
        st.subheader("🔍 Multi-Sensor Engineering Data")
        sensor_cols = st.columns(len(active_sensors))
        for i, sensor_id in enumerate(active_sensors):
            with sensor_cols[i]:
                st.markdown(f"**{sensor_id}**")
                features = features_dict[sensor_id]
                st.metric("RMS", f"{features['rms']:.4f}")
                st.metric("Peak-to-Peak", f"{features['pkpk']:.3f}")
                st.metric("Crest Factor", f"{features['crest']:.2f}")
        
        # БИЗНЕС-МЕТРИКИ СОХРАНЕНЫ:
        prevented_hours, potential_savings, system_cost = show_business_impact(severity)
        roi = potential_savings / system_cost if system_cost > 0 else 0
        
        # LINKEDIN ГЕНЕРАТОР СОХРАНЕН:
        linkedin_text, linkedin_img = generate_linkedin_post(
            fault_type, severity, prevented_hours, potential_savings, roi, current_fig
        )
        
        st.subheader("📢 LinkedIn-ready Post")
        st.text_area("Suggested text:", linkedin_text, height=200)
        if linkedin_img:
            st.image(linkedin_img, caption="Attach this graph to your post", use_container_width=True)
        else:
            st.warning("⚠️ Install kaleido for image export: pip install kaleido")
        
        # НОВАЯ ФИЧКА: PDF ГЕНЕРАТОР
        st.subheader("📊 Professional PDF Report")
        if st.button("📄 Generate PDF Report"):
            generate_pdf_report(
                fault_type, severity, prevented_hours, potential_savings, roi,
                active_sensors, dampers_enabled, sample_rate, accelerometer_type,
                features_dict
            )
            with open("avcs_simulation_report.pdf", "rb") as f:
                st.download_button(
                    "📥 Download Professional Report", 
                    f, 
                    file_name="avcs_simulation_report.pdf"
                )
            st.success("✅ PDF report generated successfully!")

# ИНФОРМАЦИЯ О СИСТЕМЕ СОХРАНЕНА:
with st.expander("🏭 Industrial System Overview"):
    st.markdown("""
    **Multi-Channel Vibration Monitoring System v4.2**
    - **All previous features preserved**
    - **New PDF report generator** 
    - **Professional documentation** for clients
    - **Complete simulation workflow**
    """)

st.markdown("---")
st.markdown("**Operational Excellence, Delivered** | Multi-Channel AVCS DNA Simulator v4.2")
