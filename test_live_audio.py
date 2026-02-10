"""
Test script for Live Audio Features
Quick test to verify live audio visualization is working
"""

import streamlit as st
from live_audio_visualizer import live_audio_interface

st.set_page_config(
    page_title="Live Audio Test",
    page_icon="🎙️",
    layout="wide"
)

st.title("🎙️ Live Audio Features Test")
st.markdown("---")

st.markdown("""
### Welcome to the Live Audio Test!

This is a standalone test for the new live audio visualization features.

**Features to test:**
1. 🌊 Real-time waveform display
2. 🔊 Live volume meter
3. 📊 Live frequency spectrum
4. 📈 Recording metrics
5. 💾 Download recording
6. 📊 Post-recording analysis

**Instructions:**
1. Click the "Start Live Recording" button below
2. Speak into your microphone
3. Watch the real-time visualizations
4. Review the post-recording analysis
""")

st.markdown("---")

# Test the live audio interface
st.header("🎤 Live Recording Test")

col1, col2 = st.columns([3, 1])

with col1:
    st.info("👇 Click the button below to start a live recording test")

with col2:
    duration = st.number_input("Duration (seconds):", 3, 30, 5)

st.markdown("---")

# Display the live recording interface
audio_data, sample_rate = live_audio_interface.display_live_recording_interface(duration)

if audio_data is not None:
    st.success("✅ Test completed successfully!")
    
    # Display some basic info
    st.markdown("### 📊 Recording Information")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Sample Rate", f"{sample_rate} Hz")
    
    with col2:
        st.metric("Duration", f"{len(audio_data)/sample_rate:.2f}s")
    
    with col3:
        st.metric("Samples", len(audio_data))
    
    st.balloons()
    st.success("🎉 All live audio features are working correctly!")

st.markdown("---")

# Additional information
with st.expander("ℹ️ About Live Audio Features"):
    st.markdown("""
    ### What's New?
    
    **Real-time Visualization:**
    - See your voice as you speak
    - Instant feedback on volume levels
    - Frequency analysis in real-time
    
    **Quality Indicators:**
    - 🔴 Red: Too quiet (speak louder)
    - 🟡 Yellow: Good volume
    - 🟢 Green: Excellent volume
    
    **Post-Recording Analysis:**
    - Complete waveform view
    - Spectrogram analysis
    - Pitch contour tracking
    
    **Benefits:**
    - Better recording quality
    - Immediate feedback
    - Confidence building
    - Professional audio analysis
    
    For more information, see `LIVE_AUDIO_FEATURES.md`
    """)

with st.expander("🔧 Troubleshooting"):
    st.markdown("""
    ### Common Issues
    
    **No audio visualization:**
    - Check microphone permissions
    - Ensure microphone is connected
    - Refresh the page
    
    **Volume too low:**
    - Speak louder
    - Move closer to microphone
    - Check system microphone settings
    
    **Recording fails:**
    - Check browser console for errors
    - Verify all dependencies are installed
    - Try a shorter recording duration
    
    **Performance issues:**
    - Close other browser tabs
    - Use a more powerful device
    - Reduce recording duration
    """)

st.markdown("---")
st.markdown("**EchoLearn** - Live Audio Features v1.0.0")
