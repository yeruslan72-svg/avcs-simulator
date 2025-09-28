import streamlit as st
import numpy as np
import plotly.graph_objects as go
import plotly.subplots as sp
import time
import io
from PIL import Image

# ==================== ПРОВЕРКА reportlab ====================
try:
    from reportlab.lib.pagesizes import letter
    from reportlab.pdfgen import canvas
    from reportlab.lib.styles import getSampleStyleSheet
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
    REPORTLAB_AVAILABLE = True
except ImportError:
    REPORTLAB_AVAILABLE = False

# ==================== НАСТРОЙКА СТРАНИЦЫ ====================
st.set_page_config(
    page_title="AVCS DNA Multi-Channel Simulator", 
    layout="wide",
    page_icon="🏭"
)

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
    """Генерация сигналов вибрации с различными типами неисправностей"""
    base_signal = np.sin(2 * np.pi * base_freq * time_points)
    base_signal += 0.05 * np.random.randn(len(time_points))
    
    # Добавляем характеристики в зависимости от позиции датчика
    if sensor_position == 'Motor_End':
        high_freq = 0.3 * np.sin(2 * np.pi * 200 * time_points)
        base_signal += 0.1 * high_freq
    else:
        low_freq = 0.4 * np.sin(2 * np.pi * 25 * time_points)
        base_signal += 0.15 * low_freq
    
    anomalies = np.zeros_like(time_points)
    
    if fault_type == "Normal Operation":
        return base_signal, anomalies
        
    elif "Bearing_Fault" in fault_type:
        impulse_prob = 0.008 * severity
        impulses = (np.random.rand(len(time_points)) < impulse_prob).astype(float) * severity * 0.6
        anomalies = impulses
        return base_signal + impulses, anomalies
        
    elif fault_type == "Imbalance":
        imbalance_effect = 0.4 * severity
        modulated = base_signal * (1 + imbalance_effect * np.sin(2 * np.pi * base_freq * time_points))
        return modulated, anomalies
        
    elif fault_type == "Misalignment":
        harmonic_2x = 0.5 * severity * np.sin(2 * np.pi * 2 * base_freq * time_points + np.pi/4)
        harmonic_3x = 0.3 * severity * np.sin(2 * np.pi * 3 * base_freq * time_points + np.pi/6)
        return base_signal + harmonic_2x + harmonic_3x, anomalies
        
    return base_signal, anomalies

def calculate_angular_damper_force(dampers_config, fault_detected, severity, sensor_position):
    """Расчет сил демпфирования с учетом угловой конфигурации"""
    forces = {}
    if not fault_detected:
        for damper_id in dampers_config:
            forces[damper_id] = 500  # Базовая сила при нормальной работе
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
    """Применение подавления вибраций демпферами"""
    total_force = sum(damper_forces.values())
    suppression_factor = np.exp(-0.25 * total_force / 8000)
    response_delay = int(0.02 * len(time_points))
    suppressed_signal = np.copy(signal_data)
    if response_delay < len(signal_data):
        suppressed_signal[response_delay:] = signal_data[response_delay:] * suppression_factor
    return suppressed_signal

def calculate_multi_channel_features(signals_dict):
    """Расчет характеристик сигналов для всех каналов"""
    features_dict = {}
    for sensor_id, signal_data in signals_dict.items():
        if len(signal_data) == 0:
            features_dict[sensor_id] = {'rms': 0, 'pkpk': 0, 'crest': 0}
            continue
            
        rms = np.sqrt(np.mean(signal_data**2))
        pkpk = np.ptp(signal_data)
        crest = np.max(np.abs(signal_data)) / rms if rms > 0 else 0
        
        features_dict[sensor_id] = {
            'rms': rms, 
            'pkpk': pkpk, 
            'crest': crest,
            'position': sensor_id
        }
        
    return features_dict

def show_business_impact(severity):
    """Отображение экономического воздействия"""
    st.subheader("📈 Business Impact Estimation")
    col_cost, col_impact = st.columns(2)
    
    with col_cost:
        downtime_cost = st.number_input(
            "Estimated hourly downtime cost ($)", 
            min_value=1000, 
            value=10000, 
            step=1000, 
            key="downtime_cost"
        )
    
    with col_impact:
        prevented_hours = severity * 8
        potential_savings = downtime_cost * prevented_hours
        system_cost = 120000
        
        st.metric("💾 Potential downtime prevented", f"{prevented_hours} hours")
        st.metric("💰 Estimated savings", f"${potential_savings:,.0f}")
        
        if system_cost > 0:
            roi = potential_savings / system_cost
            st.metric("📊 ROI multiplier", f"{roi:.1f}x")
        else:
            roi = 0
            
    return prevented_hours, potential_savings, system_cost, roi

def generate_text_report(fault_type, severity, prevented_hours, potential_savings, roi,
                       active_sensors, dampers_enabled, sample_rate, accelerometer_type,
                       features_dict):
    """Генерация текстового отчета"""
    report_text = f"""
AVCS DNA SIMULATION REPORT
Generated on: {time.strftime('%Y-%m-%d %H:%M:%S')}

SYSTEM CONFIGURATION:
- Fault Type: {fault_type}
- Severity Level: {severity}/5
- Active Sensors: {', '.join(active_sensors)}
- Dampers: {'Enabled' if dampers_enabled else 'Disabled'}
- Sample Rate: {sample_rate} Hz
- Accelerometer: {accelerometer_type}

SIMULATION RESULTS:
- Potential Downtime Prevented: {prevented_hours} hours
- Estimated Savings: ${potential_savings:,.0f}
- ROI Multiplier: {roi:.1f}x

TECHNICAL PARAMETERS:
"""
    
    for sensor_id, features in features_dict.items():
        report_text += f"- {sensor_id}: RMS={features['rms']:.4f}, PkPk={features['pkpk']:.3f}, Crest={features['crest']:.2f}\n"
    
    report_text += "\nCONCLUSION:\nAVCS DNA system demonstrates effective vibration control with significant ROI potential."
    
    return report_text

def generate_pdf_report(fault_type, severity, prevented_hours, potential_savings, roi,
                       active_sensors, dampers_enabled, sample_rate, accelerometer_type,
                       features_dict, filename="avcs_simulation_report.pdf"):
    """Генерация PDF отчета"""
    if not REPORTLAB_AVAILABLE:
        return generate_text_report(fault_type, severity, prevented_hours, potential_savings, roi,
                                  active_sensors, dampers_enabled, sample_rate, accelerometer_type,
                                  features_dict)
    
    try:
        doc = SimpleDocTemplate(filename, pagesize=letter)
        styles = getSampleStyleSheet()
        story = []
        
        # Заголовок
        title = Paragraph("AVCS DNA Simulation Report", styles['Heading1'])
        story.append(title)
        story.append(Spacer(1, 12))
        
        # Дата генерации
        date_text = Paragraph(f"Generated on: {time.strftime('%Y-%m-%d %H:%M:%S')}", styles['Normal'])
        story.append(date_text)
        story.append(Spacer(1, 20))
        
        # Конфигурация системы
        config_title = Paragraph("System Configuration", styles['Heading2'])
        story.append(config_title)
        
        config_data = f"""
        Fault Type: {fault_type}<br/>
        Severity Level: {severity}/5<br/>
        Active Sensors: {', '.join(active_sensors)}<br/>
        Dampers: {'Enabled' if dampers_enabled else 'Disabled'}<br/>
        Sample Rate: {sample_rate} Hz<br/>
        Accelerometer: {accelerometer_type}
        """
        config_para = Paragraph(config_data, styles['Normal'])
        story.append(config_para)
        story.append(Spacer(1, 20))
        
        # Результаты
        results_title = Paragraph("Simulation Results", styles['Heading2'])
        story.append(results_title)
        
        results_data = f"""
        Potential Downtime Prevented: {prevented_hours} hours<br/>
        Estimated Savings: ${potential_savings:,.0f}<br/>
        ROI Multiplier: {roi:.1f}x
        """
        results_para = Paragraph(results_data, styles['Normal'])
        story.append(results_para)
        story.append(Spacer(1, 20))
        
        # Технические параметры
        tech_title = Paragraph("Technical Parameters", styles['Heading2'])
        story.append(tech_title)
        
        for sensor_id, features in features_dict.items():
            tech_data = f"{sensor_id}: RMS={features['rms']:.4f}, PkPk={features['pkpk']:.3f}, Crest={features['crest']:.2f}"
            tech_para = Paragraph(tech_data, styles['Normal'])
            story.append(tech_para)
        
        story.append(Spacer(1, 20))
        conclusion = Paragraph("CONCLUSION: AVCS DNA system demonstrates effective vibration control with significant ROI potential.", styles['Normal'])
        story.append(conclusion)
        
        doc.build(story)
        return True
        
    except Exception as e:
        st.error(f"PDF generation failed: {str(e)}")
        return False

# ==================== ИНИЦИАЛИЗАЦИЯ СЕССИИ ====================
if 'simulation_run' not in st.session_state:
    st.session_state.simulation_run = False
if 'features' not in st.session_state:
    st.session_state.features = {}
if 'report_data' not in st.session_state:
    st.session_state.report_data = {}

# ==================== ИНТЕРФЕЙС ====================
config = EquipmentConfig()

col1, col2 = st.columns([1, 2])

with col1:
    st.subheader("🎛️ Equipment Configuration")
    
    # Конфигурация сенсоров
    st.markdown("**📡 Sensor Configuration**")
    sensor_motor = st.checkbox("Motor End Sensor", value=True, key="sensor_motor")
    sensor_pump = st.checkbox("Pump End Sensor", value=True, key="sensor_pump")
    
    # Выбор оборудования
    accelerometer_type = st.selectbox(
        "Accelerometer Type", 
        list(config.accelerometer_types.keys()), 
        key="accel_type"
    )
    sample_rate = st.selectbox(
        "Sample Rate (Hz)", 
        config.sample_rates, 
        index=3, 
        key="sample_rate"
    )
    
    # Конфигурация неисправностей
    st.markdown("**⚡ Fault Configuration**")
    fault_type = st.selectbox(
        "Fault Type", 
        [
            "Normal Operation", 
            "Bearing_Fault_Mild", 
            "Bearing_Fault_Severe", 
            "Imbalance", 
            "Misalignment"
        ], 
        key="fault_type"
    )
    
    severity = st.slider("Fault Severity", 1, 5, 1, key="severity")
    dampers_enabled = st.checkbox("Enable Active Dampers", value=True, key="dampers")
    
    # Настройки анимации
    show_animation = st.checkbox("Show Live Animation", value=True, key="animation")
    if show_animation:
        animation_speed = st.slider("Animation Speed", 1, 5, 3, key="anim_speed")
    
    # Кнопка запуска симуляции
    run_simulation = st.button("▶️ Start Multi-Channel Simulation", type="primary", use_container_width=True)

with col2:
    st.subheader("📊 Multi-Channel Monitoring")
    
    if run_simulation or st.session_state.simulation_run:
        active_sensors = []
        if sensor_motor: 
            active_sensors.append('Motor_End')
        if sensor_pump: 
            active_sensors.append('Pump_End')
        
        if not active_sensors:
            st.error("❌ Please select at least one sensor")
            st.stop()
        
        # Генерация временных точек
        time_points = np.linspace(0, 0.1, int(sample_rate * 0.1))
        
        if show_animation:
            # Режим анимации
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
                
                # Создание графика
                fig = sp.make_subplots(
                    rows=len(active_sensors), 
                    cols=1,
                    subplot_titles=[
                        f"{sensor_id} - {config.sensors[sensor_id]['position']}" 
                        for sensor_id in active_sensors
                    ],
                    vertical_spacing=0.1
                )
                
                for i, sensor_id in enumerate(active_sensors):
                    row = i + 1
                    fig.add_trace(
                        go.Scatter(
                            x=time_points*1000, 
                            y=signals[sensor_id], 
                            mode='lines', 
                            name=f'{sensor_id} Original', 
                            line=dict(color='blue', width=2)
                        ),
                        row=row, col=1
                    )
                    
                    if dampers_enabled and fault_detected:
                        fig.add_trace(
                            go.Scatter(
                                x=time_points*1000, 
                                y=suppressed_signals[sensor_id], 
                                mode='lines',
                                name=f'{sensor_id} Suppressed', 
                                line=dict(color='green', width=2)
                            ),
                            row=row, col=1
                        )
                
                fig.update_layout(
                    height=300 * len(active_sensors), 
                    title_text=f"Multi-Channel Monitoring - Frame {frame+1}/{num_frames}",
                    showlegend=True
                )
                fig.update_xaxes(title_text="Time (ms)")
                fig.update_yaxes(title_text="Amplitude")
                
                animation_placeholder.plotly_chart(fig, use_container_width=True)
                time.sleep(0.5 / animation_speed)
            
            progress_bar.empty()
            status_display.success("✅ Multi-channel simulation completed!")
            
        else:
            # Статический режим
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
            
            # Создание статического графика
            fig_static = sp.make_subplots(
                rows=len(active_sensors), 
                cols=1,
                subplot_titles=[
                    f"{sensor_id} - {config.sensors[sensor_id]['position']}" 
                    for sensor_id in active_sensors
                ],
                vertical_spacing=0.1
            )
            
            for i, sensor_id in enumerate(active_sensors):
                row = i + 1
                color = 'green' if fault_type == "Normal Operation" else 'red'
                fig_static.add_trace(
                    go.Scatter(
                        x=time_points*1000, 
                        y=final_signals[sensor_id], 
                        mode='lines',
                        name=f'{sensor_id} Vibration', 
                        line=dict(color=color, width=2)
                    ),
                    row=row, col=1
                )
                
                if dampers_enabled and fault_type != "Normal Operation":
                    fig_static.add_trace(
                        go.Scatter(
                            x=time_points*1000, 
                            y=final_suppressed[sensor_id], 
                            mode='lines',
                            name=f'{sensor_id} Suppressed', 
                            line=dict(color='blue', width=2)
                        ),
                        row=row, col=1
                    )
            
            fig_static.update_layout(
                height=300 * len(active_sensors), 
                title_text="Multi-Channel Static Analysis",
                showlegend=True
            )
            st.plotly_chart(fig_static, use_container_width=True)
        
        # Сохранение результатов в session state
        st.session_state.simulation_run = True
        st.session_state.features = calculate_multi_channel_features(final_signals)
        st.session_state.final_damper_forces = final_damper_forces
        
        # Панель управления демпферами
        st.subheader("🔧 Damper Control Panel")
        damper_cols = st.columns(4)
        motor_forces = final_damper_forces.get('Motor_End', final_damper_forces.get(active_sensors[0], {}))
        
        for i, damper_id in enumerate(config.dampers.keys()):
            with damper_cols[i]:
                config_data = config.dampers[damper_id]
                force = motor_forces.get(damper_id, 0)
                st.metric(
                    f"{damper_id}", 
                    f"{force:.0f} N", 
                    f"Angle: {config_data['angle']}°"
                )
        
        # Инженерные данные
        st.subheader("🔍 Multi-Sensor Engineering Data")
        sensor_cols = st.columns(len(active_sensors))
        for i, sensor_id in enumerate(active_sensors):
            with sensor_cols[i]:
                st.markdown(f"**{sensor_id}**")
                features = st.session_state.features[sensor_id]
                st.metric("RMS", f"{features['rms']:.4f}")
                st.metric("Peak-to-Peak", f"{features['pkpk']:.3f}")
                st.metric("Crest Factor", f"{features['crest']:.2f}")
        
        # Бизнес-метрики
        prevented_hours, potential_savings, system_cost, roi = show_business_impact(severity)
        
        # Сохранение данных для отчетов
        st.session_state.report_data = {
            'fault_type': fault_type,
            'severity': severity,
            'prevented_hours': prevented_hours,
            'potential_savings': potential_savings,
            'roi': roi,
            'active_sensors': active_sensors,
            'dampers_enabled': dampers_enabled,
            'sample_rate': sample_rate,
            'accelerometer_type': accelerometer_type,
            'features': st.session_state.features
        }

# ==================== СЕКЦИЯ ОТЧЕТОВ ====================
if st.session_state.simulation_run:
    st.markdown("---")
    st.subheader("📊 Professional Report Generator")
    
    if st.button("📄 Generate Text Report", key="text_report"):
        report_data = st.session_state.report_data
        text_report = generate_text_report(
            report_data['fault_type'],
            report_data['severity'],
            report_data['prevented_hours'],
            report_data['potential_savings'],
            report_data['roi'],
            report_data['active_sensors'],
            report_data['dampers_enabled'],
            report_data['sample_rate'],
            report_data['accelerometer_type'],
            report_data['features']
        )
        st.text_area("Professional Report:", text_report, height=300, key="report_display")
        
        # Кнопка скачивания текстового отчета
        st.download_button(
            label="📥 Download Text Report",
            data=text_report,
            file_name=f"avcs_report_{time.strftime('%Y%m%d_%H%M%S')}.txt",
            mime="text/plain",
            key="text_download"
        )
    
    if REPORTLAB_AVAILABLE:
        if st.button("📄 Generate PDF Report", key="pdf_report"):
            report_data = st.session_state.report_data
            success = generate_pdf_report(
                report_data['fault_type'],
                report_data['severity'],
                report_data['prevented_hours'],
                report_data['potential_savings'],
                report_data['roi'],
                report_data['active_sensors'],
                report_data['dampers_enabled'],
                report_data['sample_rate'],
                report_data['accelerometer_type'],
                report_data['features']
            )
            
            if success:
                with open("avcs_simulation_report.pdf", "rb") as f:
                    st.download_button(
                        "📥 Download PDF Report",
                        f,
                        file_name=f"avcs_report_{time.strftime('%Y%m%d_%H%M%S')}.pdf",
                        key="pdf_download"
                    )
                st.success("✅ PDF report generated successfully!")
    else:
        st.info("💡 Install reportlab for PDF export: `pip install reportlab`")

# ==================== ИНФОРМАЦИЯ О СИСТЕМЕ ====================
with st.expander("🏭 System Information & Help"):
    st.markdown("""
    **AVCS DNA Multi-Channel Simulator v6.0**
    
    *Key Features:*
    - Multi-sensor vibration monitoring (Motor End, Pump End)
    - Real-time active damping simulation
    - Advanced fault detection algorithms
    - Business impact analysis with ROI calculation
    - Professional report generation
    
    *Supported Fault Types:*
    - Normal Operation
    - Bearing Faults (Mild/Severe)
    - Rotor Imbalance
    - Shaft Misalignment
    
    *Technical Capabilities:*
    - Sample rates: 1kHz to 20kHz
    - Professional accelerometer models
    - Angular damper configuration
    - Real-time signal processing
    """)
    
    if not REPORTLAB_AVAILABLE:
        st.warning("**PDF Reports Disabled** - Install reportlab for full functionality: `pip install reportlab`")

st.markdown("---")
st.markdown("**Operational Excellence, Delivered** | Multi-Channel AVCS DNA Simulator v6.0")

# ==================== СБРОС СИМУЛЯЦИИ ====================
if st.session_state.simulation_run:
    if st.button("🔄 Reset Simulation", key="reset"):
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        st.rerun()
