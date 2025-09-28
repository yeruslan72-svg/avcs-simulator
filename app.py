# ==================== Интерфейс ====================
st.sidebar.subheader("🎛️ Equipment & Fault Configuration")
sensor_motor = st.sidebar.checkbox("Motor End Sensor", value=True)
sensor_pump = st.sidebar.checkbox("Pump End Sensor", value=True)
fault_type = st.sidebar.selectbox("Fault Type", ["Normal Operation", "Bearing_Fault", "Imbalance", "Misalignment"])
severity = st.sidebar.slider("Fault Severity", 1,5,1)
dampers_enabled = st.sidebar.checkbox("Enable Dampers", True)
sample_rate = st.sidebar.selectbox("Sample Rate (Hz)", config.sample_rates, index=3)
show_animation = st.sidebar.checkbox("Show Animation", True)

# ⚡ Новый ползунок скорости анимации
animation_speed = st.sidebar.slider("Animation Speed", min_value=0.1, max_value=5.0, value=1.0, step=0.1)

run_sim = st.sidebar.button("▶️ Run Simulation")

# ==================== Симуляция с анимацией ====================
if run_sim:
    active_sensors = []
    if sensor_motor: active_sensors.append('Motor_End')
    if sensor_pump: active_sensors.append('Pump_End')
    if not active_sensors: st.error("Select at least one sensor"); st.stop()

    time_points = np.linspace(0,0.1,int(sample_rate*0.05))
    signals, suppressed, damper_forces_dict, features_dict = {}, {}, {}, {}

    num_frames = 10  # количество кадров для анимации
    animation_placeholder = st.empty()
    progress_bar = st.progress(0)

    for frame in range(num_frames):
        for s in active_sensors:
            sig = generate_vibration_signal(time_points, fault_type, severity, s)
            fault_detected = fault_type != "Normal Operation"
            damper_forces = calculate_damper_force(fault_detected, severity)
            signals[s] = sig
            suppressed[s] = suppress_signal(sig, damper_forces) if dampers_enabled and fault_detected else sig
            damper_forces_dict[s] = damper_forces
            features_dict[s] = calculate_features(sig)

        fig = sp.make_subplots(rows=len(active_sensors), cols=1,
                               subplot_titles=[f"{s} - {config.sensors[s]['position']}" for s in active_sensors])
        for i,s in enumerate(active_sensors):
            colors = signal_colors(signals[s])
            fig.add_trace(go.Scatter(
                x=time_points*1000,
                y=signals[s],
                mode='lines',
                line=dict(color='blue', width=2),
                name='Original'
            ), row=i+1, col=1)
            if dampers_enabled and fault_type!="Normal Operation":
                fig.add_trace(go.Scatter(
                    x=time_points*1000,
                    y=suppressed[s],
                    mode='lines',
                    line=dict(color='green', width=2),
                    name='Suppressed'
                ), row=i+1, col=1)
            fig.add_trace(go.Scatter(
                x=time_points*1000,
                y=signals[s],
                mode='lines',
                line=dict(color=colors, width=3),
                opacity=0.5,
                name='Anomaly Overlay'
            ), row=i+1, col=1)

        fig.update_layout(height=300*len(active_sensors),
                          title_text=f"Multi-Channel Simulation - Frame {frame+1}/{num_frames}")
        animation_placeholder.plotly_chart(fig, use_container_width=True)
        progress_bar.progress((frame+1)/num_frames)
        time.sleep(0.5/animation_speed)  # скорость регулируется ползунком

    progress_bar.empty()
    st.success("✅ Simulation completed!")
