"""
Live Audio Visualizer - Real-time audio visualization during recording
"""

import streamlit as st
import numpy as np
import sounddevice as sd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import librosa
import time
from collections import deque
import threading
import queue

class LiveAudioVisualizer:
    """Real-time audio visualization system"""
    
    def __init__(self, sample_rate=44100, chunk_size=1024):
        self.sample_rate = sample_rate
        self.chunk_size = chunk_size
        self.audio_queue = queue.Queue()
        self.recording = False
        self.recorded_chunks = []
        
        # Buffers for visualization
        self.waveform_buffer = deque(maxlen=sample_rate * 2)  # 2 seconds buffer
        self.volume_buffer = deque(maxlen=100)  # Last 100 volume readings
        self.frequency_data = None
        
    def audio_callback(self, indata, frames, time_info, status):
        """Callback function for audio stream"""
        if status:
            print(f"Audio callback status: {status}")
        
        # Add to queue for processing
        self.audio_queue.put(indata.copy())
        
        # Store for final recording
        if self.recording:
            self.recorded_chunks.append(indata.copy())
    
    def start_live_recording(self, duration_seconds, placeholder_wave, placeholder_volume, 
                            placeholder_freq, placeholder_metrics):
        """Start live recording with real-time visualization"""
        
        # Reset buffers
        self.waveform_buffer.clear()
        self.volume_buffer.clear()
        self.recorded_chunks = []
        self.recording = True
        
        # Calculate chunks needed
        chunks_needed = int(duration_seconds * self.sample_rate / self.chunk_size)
        
        try:
            # Start audio stream
            with sd.InputStream(
                samplerate=self.sample_rate,
                channels=1,
                callback=self.audio_callback,
                blocksize=self.chunk_size
            ):
                start_time = time.time()
                chunk_count = 0
                
                while chunk_count < chunks_needed and self.recording:
                    try:
                        # Get audio chunk with timeout
                        audio_chunk = self.audio_queue.get(timeout=1.0)
                        chunk_count += 1
                        
                        # Update buffers
                        self.waveform_buffer.extend(audio_chunk.flatten())
                        
                        # Calculate volume (RMS)
                        volume = np.sqrt(np.mean(audio_chunk**2))
                        self.volume_buffer.append(volume)
                        
                        # Update visualizations every 5 chunks (reduce overhead)
                        if chunk_count % 5 == 0:
                            elapsed = time.time() - start_time
                            remaining = max(0, duration_seconds - elapsed)
                            
                            # Update waveform
                            self._update_waveform(placeholder_wave)
                            
                            # Update volume meter
                            self._update_volume_meter(placeholder_volume, volume)
                            
                            # Update frequency spectrum
                            self._update_frequency_spectrum(placeholder_freq, audio_chunk)
                            
                            # Update metrics
                            self._update_metrics(placeholder_metrics, elapsed, remaining, volume)
                    
                    except queue.Empty:
                        continue
                
                self.recording = False
                
                # Combine all recorded chunks
                if self.recorded_chunks:
                    full_recording = np.concatenate(self.recorded_chunks, axis=0)
                    return full_recording.flatten()
                else:
                    return None
                    
        except Exception as e:
            st.error(f"Recording error: {e}")
            self.recording = False
            return None
    
    def _update_waveform(self, placeholder):
        """Update real-time waveform display"""
        if len(self.waveform_buffer) > 0:
            waveform_array = np.array(self.waveform_buffer)
            time_axis = np.arange(len(waveform_array)) / self.sample_rate
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=time_axis,
                y=waveform_array,
                mode='lines',
                line=dict(color='#1f77b4', width=1),
                name='Waveform'
            ))
            
            fig.update_layout(
                title="🌊 Live Waveform",
                xaxis_title="Time (s)",
                yaxis_title="Amplitude",
                height=250,
                margin=dict(l=50, r=20, t=40, b=40),
                showlegend=False,
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(240,240,240,0.5)'
            )
            
            placeholder.plotly_chart(fig, use_container_width=True, key=f"waveform_{time.time()}")
    
    def _update_volume_meter(self, placeholder, current_volume):
        """Update real-time volume meter"""
        # Convert to dB
        volume_db = 20 * np.log10(max(current_volume, 1e-10))
        volume_db = max(-60, min(0, volume_db))  # Clamp between -60 and 0 dB
        
        # Create gauge chart
        fig = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=volume_db,
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': "🔊 Volume Level (dB)", 'font': {'size': 16}},
            delta={'reference': -20},
            gauge={
                'axis': {'range': [-60, 0], 'tickwidth': 1},
                'bar': {'color': "darkblue"},
                'bgcolor': "white",
                'borderwidth': 2,
                'bordercolor': "gray",
                'steps': [
                    {'range': [-60, -40], 'color': 'lightgray'},
                    {'range': [-40, -20], 'color': 'yellow'},
                    {'range': [-20, 0], 'color': 'lightgreen'}
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': -10
                }
            }
        ))
        
        fig.update_layout(
            height=200,
            margin=dict(l=20, r=20, t=40, b=20)
        )
        
        placeholder.plotly_chart(fig, use_container_width=True, key=f"volume_{time.time()}")
    
    def _update_frequency_spectrum(self, placeholder, audio_chunk):
        """Update real-time frequency spectrum"""
        # Compute FFT
        audio_flat = audio_chunk.flatten()
        
        if len(audio_flat) < 512:
            return
        
        # Apply window
        windowed = audio_flat * np.hanning(len(audio_flat))
        
        # FFT
        fft_data = np.fft.rfft(windowed)
        fft_magnitude = np.abs(fft_data)
        fft_db = 20 * np.log10(fft_magnitude + 1e-10)
        
        # Frequency bins
        freqs = np.fft.rfftfreq(len(audio_flat), 1/self.sample_rate)
        
        # Limit to audible range (20 Hz - 8000 Hz)
        mask = (freqs >= 20) & (freqs <= 8000)
        freqs = freqs[mask]
        fft_db = fft_db[mask]
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=freqs,
            y=fft_db,
            mode='lines',
            fill='tozeroy',
            line=dict(color='#ff7f0e', width=2),
            name='Spectrum'
        ))
        
        fig.update_layout(
            title="📊 Live Frequency Spectrum",
            xaxis_title="Frequency (Hz)",
            yaxis_title="Magnitude (dB)",
            height=250,
            margin=dict(l=50, r=20, t=40, b=40),
            showlegend=False,
            xaxis_type="log",
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(240,240,240,0.5)'
        )
        
        placeholder.plotly_chart(fig, use_container_width=True, key=f"freq_{time.time()}")
    
    def _update_metrics(self, placeholder, elapsed, remaining, volume):
        """Update recording metrics"""
        # Calculate signal quality metrics
        volume_db = 20 * np.log10(max(volume, 1e-10))
        
        # Determine signal quality
        if volume_db > -20:
            quality = "🟢 Excellent"
            quality_color = "green"
        elif volume_db > -35:
            quality = "🟡 Good"
            quality_color = "orange"
        else:
            quality = "🔴 Too Quiet"
            quality_color = "red"
        
        # Display metrics
        placeholder.markdown(f"""
        <div style='background-color: rgba(240,240,240,0.5); padding: 15px; border-radius: 10px;'>
            <h4 style='margin-top: 0;'>📈 Recording Metrics</h4>
            <table style='width: 100%;'>
                <tr>
                    <td><b>⏱️ Elapsed:</b></td>
                    <td>{elapsed:.1f}s</td>
                    <td><b>⏳ Remaining:</b></td>
                    <td>{remaining:.1f}s</td>
                </tr>
                <tr>
                    <td><b>🔊 Volume:</b></td>
                    <td>{volume_db:.1f} dB</td>
                    <td><b>✨ Quality:</b></td>
                    <td><span style='color: {quality_color};'>{quality}</span></td>
                </tr>
            </table>
        </div>
        """, unsafe_allow_html=True)
    
    def stop_recording(self):
        """Stop the recording"""
        self.recording = False


class LiveAudioInterface:
    """Streamlit interface for live audio visualization"""
    
    def __init__(self):
        self.visualizer = LiveAudioVisualizer()
    
    def record_with_visualization(self, duration_seconds):
        """Record audio with real-time visualization - no extra UI, just recording.
        
        Use this method when you already have your own button/UI and just want
        the recording functionality with live visualization.
        
        Returns:
            tuple: (audio_data, sample_rate) or (None, None) if failed
        """
        # Create placeholders for live visualization
        st.markdown("### 🔴 **RECORDING IN PROGRESS**")
        
        col1, col2 = st.columns(2)
        
        with col1:
            placeholder_wave = st.empty()
            placeholder_freq = st.empty()
        
        with col2:
            placeholder_volume = st.empty()
            placeholder_metrics = st.empty()
        
        # Start live recording
        st.info("🎙️ **Recording... Speak now!**")
        
        audio_data = self.visualizer.start_live_recording(
            duration_seconds,
            placeholder_wave,
            placeholder_volume,
            placeholder_freq,
            placeholder_metrics
        )
        
        if audio_data is not None and len(audio_data) > 0:
            st.success("✅ **Recording Complete!**")
            return audio_data, self.visualizer.sample_rate
        else:
            st.error("❌ Recording failed. Please try again.")
            return None, None
    
    def display_live_recording_interface(self, duration=5):
        """Display the live recording interface with real-time visualization"""
        
        st.subheader("🎙️ Live Audio Recording")
        
        # Recording controls
        col1, col2 = st.columns([3, 1])
        
        with col1:
            duration = st.slider("Recording Duration (seconds):", 3, 30, duration, 1)
        
        with col2:
            st.markdown("<br>", unsafe_allow_html=True)
            record_button = st.button("🔴 Start Live Recording", type="primary", use_container_width=True)
        
        if record_button:
            with st.spinner("🎤 Preparing live recording..."):
                time.sleep(0.5)
            
            # Create placeholders for live visualization
            st.markdown("### 🔴 **RECORDING IN PROGRESS**")
            
            col1, col2 = st.columns(2)
            
            with col1:
                placeholder_wave = st.empty()
                placeholder_freq = st.empty()
            
            with col2:
                placeholder_volume = st.empty()
                placeholder_metrics = st.empty()
            
            # Start live recording
            st.info("🎙️ **Recording... Speak now!**")
            
            audio_data = self.visualizer.start_live_recording(
                duration,
                placeholder_wave,
                placeholder_volume,
                placeholder_freq,
                placeholder_metrics
            )
            
            if audio_data is not None:
                st.success("✅ **Recording Complete!**")
                
                # Store in session state
                timestamp = time.time()
                st.session_state[f'live_audio_{timestamp}'] = audio_data
                st.session_state[f'live_audio_sr_{timestamp}'] = self.visualizer.sample_rate
                st.session_state['last_live_recording'] = timestamp
                
                # Provide download option
                self._display_download_option(audio_data, self.visualizer.sample_rate)
                
                # Final analysis
                st.markdown("---")
                st.subheader("📊 Post-Recording Analysis")
                self._display_post_analysis(audio_data, self.visualizer.sample_rate)
                
                return audio_data, self.visualizer.sample_rate
            else:
                st.error("❌ Recording failed. Please try again.")
                return None, None
        
        return None, None
    
    def _display_download_option(self, audio_data, sample_rate):
        """Provide download option for recorded audio"""
        import scipy.io.wavfile as wav
        import io
        
        # Convert to WAV
        buffer = io.BytesIO()
        wav.write(buffer, sample_rate, (audio_data * 32767).astype(np.int16))
        buffer.seek(0)
        
        st.download_button(
            label="💾 Download Recording",
            data=buffer,
            file_name=f"live_recording_{int(time.time())}.wav",
            mime="audio/wav"
        )
    
    def _display_post_analysis(self, audio_data, sample_rate):
        """Display comprehensive analysis after recording"""
        
        col1, col2, col3 = st.columns(3)
        
        # Basic metrics
        duration = len(audio_data) / sample_rate
        rms = np.sqrt(np.mean(audio_data**2))
        peak = np.max(np.abs(audio_data))
        
        with col1:
            st.metric("⏱️ Duration", f"{duration:.2f}s")
        
        with col2:
            st.metric("🔊 RMS Volume", f"{20*np.log10(rms):.1f} dB")
        
        with col3:
            st.metric("📈 Peak Amplitude", f"{peak:.3f}")
        
        # Detailed visualizations
        tab1, tab2, tab3 = st.tabs(["🌊 Waveform", "📊 Spectrogram", "🎵 Pitch"])
        
        with tab1:
            self._plot_full_waveform(audio_data, sample_rate)
        
        with tab2:
            self._plot_spectrogram(audio_data, sample_rate)
        
        with tab3:
            self._plot_pitch_contour(audio_data, sample_rate)
    
    def _plot_full_waveform(self, audio_data, sample_rate):
        """Plot full waveform"""
        time_axis = np.arange(len(audio_data)) / sample_rate
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=time_axis,
            y=audio_data,
            mode='lines',
            line=dict(color='#1f77b4', width=1),
            name='Waveform'
        ))
        
        fig.update_layout(
            title="Complete Waveform",
            xaxis_title="Time (s)",
            yaxis_title="Amplitude",
            height=300
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    def _plot_spectrogram(self, audio_data, sample_rate):
        """Plot spectrogram"""
        # Compute spectrogram
        n_fft = 2048
        hop_length = 512
        
        D = librosa.stft(audio_data, n_fft=n_fft, hop_length=hop_length)
        S_db = librosa.amplitude_to_db(np.abs(D), ref=np.max)
        
        # Create time and frequency axes
        times = librosa.frames_to_time(np.arange(S_db.shape[1]), sr=sample_rate, hop_length=hop_length)
        freqs = librosa.fft_frequencies(sr=sample_rate, n_fft=n_fft)
        
        fig = go.Figure(data=go.Heatmap(
            z=S_db,
            x=times,
            y=freqs,
            colorscale='Viridis',
            colorbar=dict(title="dB")
        ))
        
        fig.update_layout(
            title="Spectrogram",
            xaxis_title="Time (s)",
            yaxis_title="Frequency (Hz)",
            height=400
        )
        
        fig.update_yaxis(range=[0, 8000])  # Limit to 8kHz
        
        st.plotly_chart(fig, use_container_width=True)
    
    def _plot_pitch_contour(self, audio_data, sample_rate):
        """Plot pitch contour"""
        try:
            # Extract pitch using librosa
            pitches, magnitudes = librosa.piptrack(
                y=audio_data,
                sr=sample_rate,
                fmin=50,
                fmax=400
            )
            
            # Get pitch contour
            pitch_contour = []
            for t in range(pitches.shape[1]):
                index = magnitudes[:, t].argmax()
                pitch = pitches[index, t]
                if pitch > 0:
                    pitch_contour.append(pitch)
                else:
                    pitch_contour.append(None)
            
            times = librosa.frames_to_time(np.arange(len(pitch_contour)), sr=sample_rate)
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=times,
                y=pitch_contour,
                mode='lines+markers',
                line=dict(color='#2ca02c', width=2),
                marker=dict(size=4),
                name='Pitch'
            ))
            
            fig.update_layout(
                title="Pitch Contour",
                xaxis_title="Time (s)",
                yaxis_title="Frequency (Hz)",
                height=300
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
        except Exception as e:
            st.warning(f"Could not extract pitch: {e}")


# Create global instance
live_audio_interface = LiveAudioInterface()
