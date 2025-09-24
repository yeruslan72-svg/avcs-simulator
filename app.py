import streamlit as st
import numpy as np
import plotly.graph_objects as go
import plotly.subplots as sp
import time
import io

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
    """Генерация вибрационного сигнала с учетом позиции датчика"""
    base_signal = np.sin(2 * np.pi * base_freq * time_points)
    base_signal += 0.05 * np.random.randn(len(time_points))
    
    # Разные характеристики вибрации на Motor End vs Pump End
    if sensor_position == 'Motor_End':
        # Motor End: больше высокочастотных компонентов
        high_freq = 0.3 * np.sin(2 * np.pi * 200 * time_points)
        base_signal += 0.1 * high_freq
    else:  # Pump_End
        # Pump End: больше низкочастотных компонентов
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
    """Расчет сил демпферов с учетом углового расположения"""
    forces = {}
    if not fault_detected:
        # Нормальный режим - минимальные силы
        for damper_id in dampers_config:
            forces[damper_id] = 500  # Базовая сила
    else:
        # Активное подавление - разные силы для разных демпферов
        base_force = min(8000, severity * 1600)
        
        # Распределение сил в зависимости от позиции датчика и демпфера
        for damper_id, config in dampers_config.items():
            position_factor = 1.0
            
            # Учет угла демпфера (45 градусов)
            angle_factor = np.sin(np.radians(config['angle']))  # Синус 45° = 0.707
            
            # Позиционные коэффициенты
            if 'Front' in config['position'] and sensor_position == 'Motor_End':
                position_factor = 1.2  # Более сильное воздействие на передние демпферы со стороны мотора
            elif 'Rear' in config['position'] and sensor_position == 'Pump_End':
                position_factor = 1.2  # Более сильное воздействие на задние демпферы со стороны насоса
            
            forces[damper_id] = base_force * angle_factor * position_factor
    
    return forces

def apply_damper_suppression(signal_data, damper_forces, time_points):
    """Применение подавления от всех демпферов"""
    total_force = sum(damper_forces.values())
    suppression_factor = np.exp(-0.25 * total_force / 8000)
    
    # Добавляем небольшую задержку отклика демпферов
    response_delay = int(0.02 * len(time_points))  # 20ms задержка
    suppressed_signal = np.copy(signal_data)
    if response_delay < len(signal_data):
        suppressed_signal[response_delay:] = signal_data[response_delay:] * suppression_factor
    
    return suppressed_signal

def calculate_multi_channel_features(signals_dict):
    """Расчет фич для многоканальной системы"""
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

# ==================== ИНТЕРФЕЙС ====================
config = EquipmentConfig()

col1, col2 = st.columns([1, 2])

with col1:
    st.subheader("🎛️ Equipment Configuration")
    
    # Выбор датчиков
    st.markdown("**📡 Sensor Configuration**")
    sensor_motor = st.checkbox("Motor End Sensor", value=True, key="sensor_motor")
    sensor_pump = st.checkbox("Pump End Sensor", value=True, key="sensor_pump")
    
    # Тип акселерометра
    accelerometer_type = st.selectbox("Accelerometer Type", 
                                     list(config.accelerometer_types.keys()),
                                     key="accel_type")
    
    # Частота дискретизации
    sample_rate = st.selectbox("Sample Rate (Hz)", config.sample_rates, index=3, key="sample_rate")
    
    # Конфигурация неисправности
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
        # Инициализация анимации
        animation_placeholder = st.empty()
        status_display = st.empty()
        
        # Активные сенсоры
        active_sensors = []
        if sensor_motor: active_sensors.append('Motor_End')
        if sensor_pump: active_sensors.append('Pump_End')
        
        if not active_sensors:
            st.error("❌ Please select at least one sensor")
            st.stop()
        
        num_frames = 12
        time_points = np.linspace(0, 0.1, int(sample_rate * 0.1))
        
        for frame in range(num_frames):
            status_display.text(f"🎬 Multi-Channel Simulation: Frame {frame+1}/{num_frames}")
            
            # Генерация сигналов для каждого канала
            signals = {}
            suppressed_signals = {}
            damper_forces_history = {}
            
            for sensor_id in active_sensors:
                # Генерация вибрации с учетом позиции датчика
                signal_data, impulses = generate_vibration_signal(
                    time_points, fault_type, severity, sensor_id
                )
                signals[sensor_id] = signal_data
                
                # Расчет сил демпферов
                fault_detected = fault_type != "Normal Operation"
                damper_forces = calculate_angular_damper_force(
                    config.dampers, fault_detected, severity, sensor_id
                )
                damper_forces_history[sensor_id] = damper_forces
                
                # Применение подавления
                if dampers_enabled and fault_detected:
                    suppressed_signals[sensor_id] = apply_damper_suppression(
                        signal_data, damper_forces, time_points
                    )
                else:
                    suppressed_signals[sensor_id] = signal_data
            
            # Визуализация многоканальной системы
            fig = sp.make_subplots(
                rows=len(active_sensors), cols=1,
                subplot_titles=[f"{sensor_id} - {config.sensors[sensor_id]['position']}" 
                               for sensor_id in active_sensors],
                vertical_spacing=0.1
            )
            
            for i, sensor_id in enumerate(active_sensors):
                row = i + 1
                
                # Оригинальный сигнал
                fig.add_trace(
                    go.Scatter(x=time_points*1000, y=signals[sensor_id],
                              mode='lines', name=f'{sensor_id} Original',
                              line=dict(color='blue', width=2)),
                    row=row, col=1
                )
                
                # Подавленный сигнал (если есть)
                if dampers_enabled and fault_detected:
                    fig.add_trace(
                        go.Scatter(x=time_points*1000, y=suppressed_signals[sensor_id],
                                  mode='lines', name=f'{sensor_id} Suppressed',
                                  line=dict(color='green', width=2)),
                        row=row, col=1
                    )
            
            fig.update_layout(height=300 * len(active_sensors), 
                            title_text="Multi-Channel Vibration Monitoring")
            fig.update_xaxes(title_text="Time (ms)")
            fig.update_yaxes(title_text="Amplitude")
            
            animation_placeholder.plotly_chart(fig, use_container_width=True)
            time.sleep(0.3)
        
        # Финальный анализ
        status_display.success("✅ Multi-channel simulation completed!")
        
        # Панель управления демпферами
        st.subheader("🔧 Damper Control Panel")
        damper_cols = st.columns(4)
        
        final_damper_forces = calculate_angular_damper_force(
            config.dampers, fault_detected, severity, 'Motor_End'
        )
        
        for i, (damper_id, force) in enumerate(final_damper_forces.items()):
            with damper_cols[i]:
                config_data = config.dampers[damper_id]
                st.metric(
                    f"{damper_id}",
                    f"{force:.0f} N",
                    f"Angle: {config_data['angle']}°"
                )
        
        # Инженерная панель для каждого канала
        features_dict = calculate_multi_channel_features(signals)
        
        st.subheader("🔍 Multi-Sensor Engineering Data")
        sensor_cols = st.columns(len(active_sensors))
        
        for i, sensor_id in enumerate(active_sensors):
            with sensor_cols[i]:
                st.markdown(f"**{sensor_id}**")
                features = features_dict[sensor_id]
                st.metric("RMS", f"{features['rms']:.4f}")
                st.metric("Peak-to-Peak", f"{features['pkpk']:.3f}")
                st.metric("Crest Factor", f"{features['crest']:.2f}")

# ==================== ИНФОРМАЦИЯ О СИСТЕМЕ ====================
with st.expander("🏭 Industrial System Overview"):
    st.markdown("""
    **Equipment Configuration:**
    - **Motor End Sensor**: Vibration monitoring at drive unit
    - **Pump End Sensor**: Vibration monitoring at driven unit  
    - **4 Angular Dampers**: 45° positioning for optimal force distribution
    - **Control Cabinet**: Real-time signal processing and damper control
    
    **Damper Positioning:**
    ```
    Front-Left (45°)      Front-Right (45°)
          [ Motor ] → [ Pump ]
    Rear-Left (45°)       Rear-Right (45°)
    ```
    
    **Sensor Specifications:**
    - ICP/Accelerometer types with different sensitivity ranges
    - Sample rates from 1kHz to 20kHz
    - Multi-channel synchronous acquisition
    """)

st.markdown("---")
st.markdown("**Operational Excellence, Delivered** | Multi-Channel AVCS DNA Simulator v4.0")
