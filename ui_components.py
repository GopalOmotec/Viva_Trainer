# -*- coding: utf-8 -*-
"""
UI Components Module for EchoLearn
Handles Streamlit UI components and user interface logic
"""

import streamlit as st
import time
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any
from scoring import ScoringAnalytics
from live_audio_visualizer import live_audio_interface

class UIComponents:
    """Handles UI components and user interface logic"""
    
    @staticmethod
    def display_question_navigation(current_index: int, total_questions: int, qa_data: Dict) -> None:
        """Display question navigation and progress"""
        # Progress bar at top
        progress = (len(st.session_state.used_q_indices)) / total_questions if total_questions > 0 else 0
        
        st.markdown(f"""
        <div style="background: white; border-radius: 16px; padding: 1.5rem; margin-bottom: 1rem;
                    box-shadow: 0 2px 8px rgba(0,0,0,0.05); border: 1px solid #e2e8f0;">
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 1rem;">
                <span style="color: #64748b; font-size: 0.9rem;">Question {current_index + 1} of {total_questions}</span>
                <span style="background: linear-gradient(135deg, #6366f1, #8b5cf6); color: white; 
                            padding: 0.25rem 0.75rem; border-radius: 20px; font-size: 0.8rem; font-weight: 500;">
                    {qa_data.get('level', 'Basic')}
                </span>
            </div>
            <div style="background: #e2e8f0; border-radius: 10px; height: 8px; overflow: hidden;">
                <div style="background: linear-gradient(90deg, #6366f1, #8b5cf6); height: 100%; 
                            width: {progress * 100}%; border-radius: 10px; transition: width 0.3s ease;"></div>
            </div>
            <div style="text-align: center; margin-top: 0.5rem; color: #64748b; font-size: 0.85rem;">
                {len(st.session_state.used_q_indices)} answered • {total_questions - len(st.session_state.used_q_indices)} remaining
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # Navigation buttons
        col1, col2, col3 = st.columns([1, 2, 1])
        
        with col1:
            if st.button("← Previous", disabled=(current_index == 0), use_container_width=True):
                st.session_state.qa_index = current_index - 1
                st.rerun()
                
        with col3:
            if st.button("Next →", disabled=(current_index == total_questions - 1), use_container_width=True):
                st.session_state.qa_index = current_index + 1
                st.rerun()
    
    @staticmethod
    def display_mode_toggles() -> Dict[str, bool]:
        """Display mode toggles and return current settings"""
        
        st.markdown("""
        <div style="background: white; border-radius: 12px; padding: 1rem; margin-bottom: 1rem;
                    border: 1px solid #e2e8f0;">
            <div style="font-size: 0.85rem; color: #64748b; margin-bottom: 0.5rem;">⚙️ Session Settings</div>
        </div>
        """, unsafe_allow_html=True)
        
        col_adaptive, col_sm = st.columns(2)
        
        with col_adaptive:
            adaptive_mode = st.checkbox(
                "🎯 Adaptive Difficulty", 
                value=st.session_state.get('adaptive_mode', True),
                help="Automatically adjust question difficulty based on your performance"
            )
            if adaptive_mode != st.session_state.get('adaptive_mode', True):
                st.session_state.adaptive_mode = adaptive_mode
                st.rerun()
        
        with col_sm:
            selective_mutism_mode = st.checkbox(
                "💜 Speech Practice Mode", 
                value=st.session_state.get('selective_mutism_mode', False),
                help="Gentle, encouraging environment for speech practice"
            )
            if selective_mutism_mode != st.session_state.get('selective_mutism_mode', False):
                st.session_state.selective_mutism_mode = selective_mutism_mode
                if selective_mutism_mode:
                    st.session_state.confidence_level = 1
                    st.session_state.success_streak = 0
                st.rerun()
        
        # Info display
        if adaptive_mode and not selective_mutism_mode:
            current_difficulty = st.session_state.get('current_difficulty', 10)
            st.markdown(f"""
            <div style="background: linear-gradient(135deg, #eff6ff, #dbeafe); border-radius: 8px; 
                        padding: 0.5rem 1rem; display: inline-block; margin-top: 0.5rem;">
                <span style="color: #1e40af; font-size: 0.85rem;">
                    🎯 Difficulty Level: <strong>{current_difficulty}/20</strong>
                </span>
            </div>
            """, unsafe_allow_html=True)
        elif selective_mutism_mode:
            confidence_level = st.session_state.get('confidence_level', 1)
            success_streak = st.session_state.get('success_streak', 0)
            stars = "⭐" * min(confidence_level, 5)
            st.markdown(f"""
            <div style="background: linear-gradient(135deg, #faf5ff, #f3e8ff); border-radius: 8px; 
                        padding: 0.5rem 1rem; display: inline-flex; gap: 1rem; margin-top: 0.5rem;">
                <span style="color: #6d28d9; font-size: 0.85rem;">💜 Speech Mode</span>
                <span style="color: #7c3aed; font-size: 0.85rem;">{stars}</span>
                <span style="color: #8b5cf6; font-size: 0.85rem;">🔥 {success_streak} streak</span>
            </div>
            """, unsafe_allow_html=True)
        
        return {
            'adaptive_mode': adaptive_mode,
            'selective_mutism_mode': selective_mutism_mode
        }
    
    @staticmethod
    def display_question_info(qa_data: Dict, adaptive_mode: bool = False) -> None:
        """Display question information and current difficulty"""
        # Beautiful question card
        score = qa_data.get('score')
        score_html = ""
        if score is not None:
            if score >= 8:
                score_class = "score-high"
                score_bg = "linear-gradient(135deg, #dcfce7, #bbf7d0)"
                score_color = "#166534"
            elif score >= 5:
                score_class = "score-medium"
                score_bg = "linear-gradient(135deg, #fef9c3, #fef08a)"
                score_color = "#854d0e"
            else:
                score_class = "score-low"
                score_bg = "linear-gradient(135deg, #fee2e2, #fecaca)"
                score_color = "#991b1b"
            score_html = f"""
            <div style="background: {score_bg}; padding: 0.5rem 1rem; border-radius: 20px; 
                        display: inline-block; margin-top: 1rem;">
                <span style="color: {score_color}; font-weight: 600;">Score: {score}/10</span>
            </div>
            """
        
        st.markdown(f"""
        <div style="background: white; border-radius: 20px; padding: 2rem; margin: 1rem 0;
                    box-shadow: 0 4px 20px rgba(0,0,0,0.05); border: 1px solid #e2e8f0;">
            <div style="color: #64748b; font-size: 0.85rem; margin-bottom: 0.5rem;">📝 Question</div>
            <div style="color: #1e293b; font-size: 1.15rem; line-height: 1.6; font-weight: 500;">
                {qa_data['question']}
            </div>
            {score_html}
        </div>
        """, unsafe_allow_html=True)
    
    @staticmethod
    def display_tts_button(qa_data: Dict) -> None:
        """Display text-to-speech button"""
        col1, col2, col3 = st.columns([1, 1, 2])
        with col1:
            if st.button("🔊 Read Aloud", use_container_width=True):
                try:
                    import pyttsx3
                    engine = pyttsx3.init()
                    engine.say(qa_data["question"])
                    engine.runAndWait()
                except Exception as e:
                    st.warning(f"Text-to-speech unavailable: {e}")
    
    @staticmethod
    def display_audio_recording_interface(qa_data: Dict, current_index: int, selective_mutism_mode: bool = False) -> Optional[str]:
        """Display audio recording interface and return transcribed text"""
        if selective_mutism_mode:
            return UIComponents._display_selective_mutism_audio_interface(qa_data, current_index)
        else:
            return UIComponents._display_standard_audio_interface(qa_data, current_index)
    
    @staticmethod
    def _display_selective_mutism_audio_interface(qa_data: Dict, current_index: int) -> Optional[str]:
        """Display selective mutism audio interface with live visualization"""
        st.markdown("""
        <div style="background: linear-gradient(135deg, #faf5ff 0%, #f3e8ff 100%); 
                    border-radius: 20px; padding: 2rem; margin: 1rem 0;
                    border: 2px solid #c4b5fd;">
            <h3 style="color: #6d28d9; margin: 0 0 0.5rem 0;">🎙️ Speech Practice Zone</h3>
            <p style="color: #7c3aed; margin: 0;">
                💪 You're doing amazing! Every word you speak makes you stronger.
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        # Initialize states
        if f'sm_transcribed_text_{current_index}' not in st.session_state:
            st.session_state[f'sm_transcribed_text_{current_index}'] = None
        
        # Comfort settings
        col1, col2, col3 = st.columns([2, 1, 1])
        with col1:
            record_seconds = st.slider(
                "🕐 Choose your comfortable time:", 
                3, 15, 5,
                help="Start with shorter times if you prefer"
            )
        with col2:
            confidence_level = st.session_state.get('confidence_level', 1)
            stars = "⭐" * min(confidence_level, 5)
            st.markdown(f"""
            <div style="background: white; border-radius: 12px; padding: 1rem; text-align: center; margin-top: 1rem;">
                <div style="font-size: 0.8rem; color: #64748b;">Your Confidence</div>
                <div style="font-size: 1.2rem;">{stars}</div>
            </div>
            """, unsafe_allow_html=True)
        with col3:
            success_streak = st.session_state.get('success_streak', 0)
            st.markdown(f"""
            <div style="background: white; border-radius: 12px; padding: 1rem; text-align: center; margin-top: 1rem;">
                <div style="font-size: 0.8rem; color: #64748b;">Streak</div>
                <div style="font-size: 1.2rem;">🔥 {success_streak}</div>
            </div>
            """, unsafe_allow_html=True)
        
        # Encouraging message based on confidence
        if confidence_level >= 4:
            st.success("🌟 You're building incredible confidence! Keep going!")
        elif confidence_level >= 2:
            st.info("😊 You're making wonderful progress! Every try counts!")
        else:
            st.info("🌱 Take your time. Starting is the bravest part!")
        
        # Record button
        if st.button(
            "🎙️ I'm Ready to Speak!", 
            type="primary", 
            key=f"sm_record_{current_index}",
            use_container_width=True
        ):
            try:
                import sounddevice as sd
                import scipy.io.wavfile as wav
                import speech_recognition as sr
                import os
                
                st.markdown("""
                <div style="background: linear-gradient(135deg, #ddd6fe 0%, #c4b5fd 100%); 
                            border-radius: 16px; padding: 1.5rem; text-align: center; margin: 1rem 0;">
                    <div style="font-size: 2rem;">🎙️</div>
                    <strong style="color: #5b21b6;">Recording... You've got this!</strong>
                    <p style="color: #7c3aed; margin: 0.5rem 0 0 0;">Speak whenever you're ready</p>
                </div>
                """, unsafe_allow_html=True)
                
                # Record
                fs = 44100
                progress = st.progress(0)
                
                chunk_duration = 0.5
                num_chunks = int(record_seconds / chunk_duration)
                all_audio = []
                
                for i in range(num_chunks):
                    chunk = sd.rec(int(chunk_duration * fs), samplerate=fs, channels=1, dtype='int16')
                    sd.wait()
                    all_audio.append(chunk)
                    progress.progress((i + 1) / num_chunks)
                
                audio = np.vstack(all_audio)
                temp_file = f"temp_sm_{current_index}.wav"
                wav.write(temp_file, fs, audio)
                
                # Store audio
                st.session_state[f'audio_data_{current_index}'] = audio.flatten().astype(np.float32) / 32767.0
                st.session_state[f'audio_sample_rate_{current_index}'] = fs
                st.session_state['last_recording_duration'] = record_seconds
                
                # Transcribe
                st.markdown("""
                <div style="background: linear-gradient(135deg, #a5b4fc 0%, #818cf8 100%); 
                            border-radius: 12px; padding: 1rem; text-align: center; color: white;">
                    🔍 Listening to your wonderful voice...
                </div>
                """, unsafe_allow_html=True)
                
                recognizer = sr.Recognizer()
                with sr.AudioFile(temp_file) as source:
                    audio_file = recognizer.record(source)
                    text = recognizer.recognize_google(audio_file)
                    
                    st.session_state[f'sm_transcribed_text_{current_index}'] = text
                    
                    try:
                        os.remove(temp_file)
                    except:
                        pass
                    
                    st.rerun()
                    
            except sr.UnknownValueError:
                st.markdown("""
                <div style="background: linear-gradient(135deg, #fef3c7 0%, #fde68a 100%); 
                            border-radius: 12px; padding: 1.5rem; text-align: center;">
                    <div style="font-size: 1.5rem;">🤗</div>
                    <strong style="color: #92400e;">That's okay!</strong>
                    <p style="color: #a16207; margin: 0.5rem 0 0 0;">
                        The important thing is that you tried. Want to try again, or use text instead?
                    </p>
                </div>
                """, unsafe_allow_html=True)
                return None
            except Exception as e:
                st.warning("🤗 Technology can be tricky! You can try again or use text below.")
                return None
        
        # Show transcription if available
        if st.session_state.get(f'sm_transcribed_text_{current_index}'):
            text = st.session_state[f'sm_transcribed_text_{current_index}']
            
            st.markdown("""
            <div style="background: linear-gradient(135deg, #dcfce7 0%, #bbf7d0 100%); 
                        border-radius: 16px; padding: 1.5rem; text-align: center; margin: 1rem 0;">
                <div style="font-size: 2rem;">🎉</div>
                <strong style="color: #166534;">Amazing! You spoke beautifully!</strong>
                <p style="color: #15803d; margin: 0.5rem 0 0 0;">I heard every word. You should be so proud!</p>
            </div>
            """, unsafe_allow_html=True)
            
            edited_text = st.text_area(
                "📝 What you said (feel free to edit):",
                value=text,
                height=100,
                key=f"sm_edited_{current_index}"
            )
            
            col1, col2 = st.columns(2)
            with col1:
                if st.button("🔄 Try Again", key=f"sm_retry_{current_index}"):
                    st.session_state[f'sm_transcribed_text_{current_index}'] = None
                    st.rerun()
            
            return edited_text
        
        return None
    
    @staticmethod
    def _display_standard_audio_interface(qa_data: Dict, current_index: int) -> Optional[str]:
        """Display standard audio recording interface with live visualization"""
        
        st.markdown("""
        <div style="background: white; border-radius: 16px; padding: 1.5rem; border: 1px solid #e2e8f0; margin: 1rem 0;">
            <h4 style="color: #334155; margin-bottom: 1rem;">🎙️ Record Your Answer</h4>
        </div>
        """, unsafe_allow_html=True)
        
        # Initialize recording state
        if 'is_recording' not in st.session_state:
            st.session_state.is_recording = False
        if f'transcribed_text_{current_index}' not in st.session_state:
            st.session_state[f'transcribed_text_{current_index}'] = None
            
        col1, col2, col3 = st.columns([2, 1, 1])
        
        with col1:
            record_seconds = st.slider(
                "⏱️ Recording duration", 
                3, 30, 10,
                help="Select how long you want to record"
            )
        
        with col2:
            use_live = st.checkbox("🔴 Live waveform", value=False, help="Show real-time audio visualization (may be slower)")
        
        with col3:
            st.markdown(f"<div style='padding-top: 1.5rem; color: #64748b;'>📊 {record_seconds}s max</div>", unsafe_allow_html=True)
        
        # Recording button with better state management
        col_btn1, col_btn2 = st.columns(2)
        
        with col_btn1:
            record_clicked = st.button(
                "🎙️ Start Recording", 
                type="primary",
                key=f"record_btn_{current_index}",
                use_container_width=True
            )
        
        # Handle recording
        if record_clicked:
            try:
                import sounddevice as sd
                import scipy.io.wavfile as wav
                import speech_recognition as sr
                import os
                
                # Show recording status
                status_placeholder = st.empty()
                progress_placeholder = st.empty()
                
                status_placeholder.markdown("""
                <div style="background: linear-gradient(135deg, #fee2e2 0%, #fecaca 100%); 
                            border-radius: 12px; padding: 1rem; text-align: center;">
                    <span style="font-size: 1.5rem;">🔴</span>
                    <strong style="color: #991b1b;"> Recording in progress...</strong>
                </div>
                """, unsafe_allow_html=True)
                
                # Record audio
                fs = 44100
                temp_file = f"temp_audio_{current_index}.wav"  # Initialize temp_file early
                
                if use_live:
                    # Use live visualizer - use the new direct recording method
                    audio_data, sample_rate = live_audio_interface.record_with_visualization(record_seconds)
                    if audio_data is not None and len(audio_data) > 0:
                        # Convert float to int16 for wav file
                        audio_int = (audio_data * 32767).astype(np.int16)
                        wav.write(temp_file, sample_rate, audio_int)
                    else:
                        status_placeholder.empty()
                        progress_placeholder.empty()
                        st.warning("🔇 No audio recorded. Please try again.")
                        return None
                else:
                    # Simple recording with progress
                    progress_bar = progress_placeholder.progress(0)
                    
                    # Record in chunks to show progress
                    chunk_duration = 0.5  # seconds per chunk
                    num_chunks = int(record_seconds / chunk_duration)
                    all_audio = []
                    
                    for i in range(num_chunks):
                        chunk = sd.rec(int(chunk_duration * fs), samplerate=fs, channels=1, dtype='int16')
                        sd.wait()
                        all_audio.append(chunk)
                        progress_bar.progress((i + 1) / num_chunks)
                    
                    audio = np.vstack(all_audio)
                    wav.write(temp_file, fs, audio)
                    sample_rate = fs
                    audio_data = audio.flatten().astype(np.float32) / 32767.0
                
                status_placeholder.empty()
                progress_placeholder.empty()
                
                # Store audio data
                st.session_state[f'audio_data_{current_index}'] = audio_data
                st.session_state[f'audio_sample_rate_{current_index}'] = sample_rate
                st.session_state['last_recording_duration'] = record_seconds
                
                # Transcribe
                st.markdown("""
                <div style="background: linear-gradient(135deg, #dbeafe 0%, #bfdbfe 100%); 
                            border-radius: 12px; padding: 1rem; text-align: center;">
                    <span style="font-size: 1.2rem;">🔍</span>
                    <strong style="color: #1e40af;"> Transcribing your answer...</strong>
                </div>
                """, unsafe_allow_html=True)
                
                recognizer = sr.Recognizer()
                with sr.AudioFile(temp_file) as source:
                    audio_file = recognizer.record(source)
                    text = recognizer.recognize_google(audio_file)
                    
                    st.session_state[f'transcribed_text_{current_index}'] = text
                    
                    # Clean up temp file
                    try:
                        os.remove(temp_file)
                    except:
                        pass
                    
                    st.rerun()
                    
            except sr.UnknownValueError:
                st.warning("🔇 Couldn't understand the audio. Please try speaking more clearly.")
                return None
            except sr.RequestError as e:
                st.error(f"❌ Speech recognition service error: {e}")
                return None
            except Exception as e:
                st.error(f"❌ Recording error: {str(e)}")
                return None
        
        # Display transcription if available
        if st.session_state.get(f'transcribed_text_{current_index}'):
            text = st.session_state[f'transcribed_text_{current_index}']
            
            st.markdown("""
            <div style="background: linear-gradient(135deg, #dcfce7 0%, #bbf7d0 100%); 
                        border-radius: 12px; padding: 1rem; margin: 1rem 0;">
                <span style="font-size: 1.2rem;">✅</span>
                <strong style="color: #166534;"> Recording transcribed successfully!</strong>
            </div>
            """, unsafe_allow_html=True)
            
            # Show editable transcription
            edited_text = st.text_area(
                "📝 Your transcribed answer (you can edit if needed):",
                value=text,
                height=120,
                key=f"edited_transcription_{current_index}"
            )
            
            # Display audio analysis
            if f'audio_data_{current_index}' in st.session_state:
                with st.expander("📊 Audio Analysis", expanded=False):
                    UIComponents.display_audio_analysis(
                        st.session_state[f'audio_data_{current_index}'],
                        st.session_state[f'audio_sample_rate_{current_index}']
                    )
            
            # Clear and re-record option
            col1, col2 = st.columns(2)
            with col1:
                if st.button("🔄 Re-record", key=f"rerecord_{current_index}"):
                    st.session_state[f'transcribed_text_{current_index}'] = None
                    if f'audio_data_{current_index}' in st.session_state:
                        del st.session_state[f'audio_data_{current_index}']
                    st.rerun()
            
            return edited_text
        
        return None
    
    @staticmethod
    def display_text_input(qa_data: Dict, current_index: int, selective_mutism_mode: bool = False) -> str:
        """Display text input interface"""
        if selective_mutism_mode:
            st.markdown("---")
            st.markdown("### ✍️ **Alternative: Write Your Answer**")
            st.info("🌱 If speaking feels too hard right now, you can write your answer. This is also great practice!")
            
            return st.text_area(
                "Type your answer here:", 
                value=qa_data.get("user_answer", ""), 
                key=f"backup_answer_{current_index}",
                help="Writing is also a wonderful way to express your thoughts!"
            )
        else:
            return st.text_area("Edit Your Answer", value=qa_data.get("user_answer", ""), key=f"user_answer_{current_index}")
    
    @staticmethod
    def display_submit_button(mode: str = "standard") -> bool:
        """Display submit button with appropriate text"""
        if mode == "selective_mutism_text":
            return st.button("📝 Submit Written Answer", key="backup_submit")
        else:
            return st.button("✅ Submit Answer", type="primary")
    
    @staticmethod
    def display_evaluation_result(evaluation_result: Dict, mode: str = "standard") -> None:
        """Display evaluation result with appropriate messaging"""
        score = evaluation_result.get('score', 0)
        
        if mode == "selective_mutism":
            # Display encouraging messages for selective mutism
            encouragement = evaluation_result.get('encouragement', 'Great job!')
            feedback = evaluation_result.get('feedback', 'You\'re doing wonderfully!')
            
            st.markdown(f"""
            <div style="background: linear-gradient(135deg, #faf5ff 0%, #f3e8ff 100%); 
                        border-radius: 16px; padding: 1.5rem; margin: 1rem 0; text-align: center;
                        border: 2px solid #c4b5fd;">
                <div style="font-size: 2rem; margin-bottom: 0.5rem;">{"🎉" if score >= 6 else "💜"}</div>
                <div style="color: #6d28d9; font-size: 1.2rem; font-weight: 600;">{encouragement}</div>
                <div style="color: #7c3aed; margin-top: 0.5rem;">{feedback}</div>
                <div style="margin-top: 1rem;">
                    <span style="background: linear-gradient(135deg, #8b5cf6, #6366f1); color: white;
                                padding: 0.5rem 1.5rem; border-radius: 20px; font-weight: 600;">
                        Score: {score}/10
                    </span>
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            if score >= 6:
                st.balloons()
        else:
            # Determine score category
            if score >= 8:
                bg = "linear-gradient(135deg, #dcfce7 0%, #bbf7d0 100%)"
                border_color = "#86efac"
                text_color = "#166534"
                emoji = "🌟"
                message = "Excellent answer!"
            elif score >= 6:
                bg = "linear-gradient(135deg, #dbeafe 0%, #bfdbfe 100%)"
                border_color = "#93c5fd"
                text_color = "#1e40af"
                emoji = "👍"
                message = "Good job!"
            elif score >= 4:
                bg = "linear-gradient(135deg, #fef9c3 0%, #fef08a 100%)"
                border_color = "#fde047"
                text_color = "#854d0e"
                emoji = "📝"
                message = "Getting there!"
            else:
                bg = "linear-gradient(135deg, #fee2e2 0%, #fecaca 100%)"
                border_color = "#fca5a5"
                text_color = "#991b1b"
                emoji = "📚"
                message = "Keep practicing!"
            
            feedback = evaluation_result.get('feedback', '')
            suggestions = evaluation_result.get('suggestions', '')
            
            st.markdown(f"""
            <div style="background: {bg}; border-radius: 16px; padding: 1.5rem; margin: 1rem 0;
                        border: 2px solid {border_color};">
                <div style="display: flex; align-items: center; justify-content: space-between;">
                    <div>
                        <span style="font-size: 1.5rem;">{emoji}</span>
                        <span style="color: {text_color}; font-size: 1.1rem; font-weight: 600; margin-left: 0.5rem;">
                            {message}
                        </span>
                    </div>
                    <div style="background: white; padding: 0.5rem 1rem; border-radius: 12px;">
                        <span style="color: {text_color}; font-weight: 700; font-size: 1.2rem;">{score}/10</span>
                    </div>
                </div>
                {"<div style='margin-top: 1rem; color: " + text_color + "; font-size: 0.95rem;'><strong>💡 Feedback:</strong> " + feedback + "</div>" if feedback else ""}
                {"<div style='margin-top: 0.5rem; color: " + text_color + "; font-size: 0.9rem;'><strong>📚 Tip:</strong> " + suggestions + "</div>" if suggestions else ""}
            </div>
            """, unsafe_allow_html=True)
    
    @staticmethod
    def display_session_statistics(evaluations: List[Dict]) -> None:
        """Display session statistics"""
        if not evaluations:
            return
        
        stats = ScoringAnalytics.calculate_session_statistics(evaluations)
        
        st.markdown("""
        <div style="background: white; border-radius: 16px; padding: 1.5rem; margin: 1.5rem 0;
                    box-shadow: 0 2px 8px rgba(0,0,0,0.05); border: 1px solid #e2e8f0;">
            <div style="color: #64748b; font-size: 0.9rem; margin-bottom: 1rem;">📊 Session Progress</div>
        </div>
        """, unsafe_allow_html=True)
        
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("📋 Questions", stats['total_questions'])
        col2.metric("✅ Answered", stats['answered_questions'])
        col3.metric("🎯 Score", f"{stats['total_score']}/{stats['max_possible_score']}")
        if stats['answered_questions'] > 0:
            col4.metric("📈 Average", f"{stats['average_score']:.1f}/10")
        else:
            col4.metric("📈 Average", "—")
    
    @staticmethod
    def display_final_score_report(evaluations: List[Dict]) -> None:
        """Display comprehensive final score report"""
        st.subheader("🏆 Final Score Report")
        
        stats = ScoringAnalytics.calculate_session_statistics(evaluations)
        
        # Display main metrics
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric(
                label="🎯 Overall Score",
                value=f"{stats['total_score']}/{stats['max_possible_score']}",
                delta=f"{stats['percentage']:.1f}%"
            )
        
        with col2:
            st.metric(
                label="📊 Average per Question",
                value=f"{stats['average_score']:.1f}/10",
                delta=f"{(stats['average_score']/10)*100:.0f}%"
            )
        
        with col3:
            grade_messages = {
                "A+": "🏅 Outstanding!",
                "A": "⭐ Excellent!",
                "B": "😊 Good Job!",
                "C": "👍 Fair Performance",
                "D": "💪 Need Improvement",
                "F": "📚 Keep Studying!"
            }
            delta_message = grade_messages.get(stats['grade'], "No questions answered")
            st.metric(
                label="🏅 Final Grade",
                value=stats['grade'],
                delta=delta_message
            )
        
        # Difficulty distribution analysis
        if evaluations:
            UIComponents._display_difficulty_analysis(evaluations)
        
        # Selective Mutism Progress Insights (if applicable)
        if st.session_state.get('selective_mutism_mode', False):
            UIComponents._display_selective_mutism_progress()
        
        # Adaptive learning insights (if applicable)
        elif st.session_state.get('adaptive_mode', False):
            UIComponents._display_adaptive_learning_insights()
    
    @staticmethod
    def _display_difficulty_analysis(evaluations: List[Dict]) -> None:
        """Display difficulty distribution analysis"""
        st.subheader("📈 Performance by Difficulty")
        
        difficulty_stats = ScoringAnalytics.analyze_difficulty_performance(evaluations)
        
        for diff_range, stats in difficulty_stats.items():
            st.write(f"**{diff_range}:** {stats['count']} questions, {stats['average']:.1f}/10 avg ({stats['percentage']:.1f}%)")
            st.progress(stats['percentage'] / 100)
    
    @staticmethod
    def _display_selective_mutism_progress() -> None:
        """Display selective mutism progress insights"""
        st.subheader("🤝 Selective Mutism Progress")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            confidence_level = st.session_state.get('confidence_level', 1)
            confidence_stars = "⭐" * confidence_level
            st.metric(
                "Confidence Level",
                f"{confidence_stars} ({confidence_level}/5)",
                help="Your confidence has grown through successful participation"
            )
        
        with col2:
            success_streak = st.session_state.get('success_streak', 0)
            st.metric(
                "Success Streak",
                str(success_streak),
                help="Consecutive good answers (builds confidence)"
            )
        
        with col3:
            # Count milestones achieved
            milestones = st.session_state.get('sm_progress_milestones', [])
            milestones_achieved = len([m for m in milestones if m.get('type') == 'confidence_increase'])
            st.metric(
                "Confidence Milestones",
                str(milestones_achieved),
                help="Times you've leveled up in confidence"
            )
        
        # Encouragement based on progress
        confidence_level = st.session_state.get('confidence_level', 1)
        if confidence_level >= 4:
            st.success("🌟 Amazing! You've built tremendous confidence. You should be very proud of your progress!")
        elif confidence_level >= 3:
            st.success("🎉 Great job! Your confidence is growing strong. Keep up the excellent work!")
        elif confidence_level >= 2:
            st.info("😊 You're making good progress! Each question you answer builds your confidence.")
        else:
            st.info("🌱 You've taken the first step, and that's wonderful! Every answer helps you grow.")
    
    @staticmethod
    def _display_adaptive_learning_insights() -> None:
        """Display adaptive learning insights"""
        st.subheader("🧠 Adaptive Learning Insights")
        
        difficulty_path = st.session_state.get('difficulty_path', [])
        if not difficulty_path:
            return
        
        # Calculate learning trajectory
        initial_difficulty = difficulty_path[0]['difficulty']
        final_difficulty = st.session_state.get('current_difficulty', 10)
        difficulty_change = final_difficulty - initial_difficulty
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric(
                "Starting Difficulty",
                f"{initial_difficulty}/20",
                help="Difficulty level of first question"
            )
        
        with col2:
            st.metric(
                "Final Difficulty", 
                f"{final_difficulty}/20",
                delta=f"{difficulty_change:+d}"
            )
        
        with col3:
            correct_answers = sum(1 for step in difficulty_path if step.get('correct', False))
            accuracy = (correct_answers / len(difficulty_path)) * 100 if difficulty_path else 0
            st.metric(
                "Accuracy Rate",
                f"{accuracy:.1f}%",
                delta=f"{correct_answers}/{len(difficulty_path)}"
            )
        
        # Learning trajectory chart
        if len(difficulty_path) > 1:
            st.write("**📈 Learning Trajectory:**")
            
            difficulty_progression = [step['difficulty'] for step in difficulty_path]
            scores_progression = [step['score'] for step in difficulty_path]
            
            df = pd.DataFrame({
                'Question': range(1, len(difficulty_progression) + 1),
                'Difficulty Level': difficulty_progression,
                'Score': scores_progression
            })
            
            st.line_chart(df.set_index('Question'))
            
            # Performance insights
            if difficulty_change > 0:
                st.success(f"🚀 Great progress! You advanced {difficulty_change} difficulty levels.")
            elif difficulty_change == 0:
                st.info("🎯 You maintained a consistent difficulty level throughout the session.")
            else:
                st.info(f"📚 The system adapted to your learning pace, focusing on foundational concepts.")
    
    @staticmethod
    def display_adaptive_progress() -> None:
        """Display adaptive learning progress visualization"""
        difficulty_path = st.session_state.get('difficulty_path', [])
        if not difficulty_path:
            return
        
        # Show recent difficulty changes
        recent_steps = difficulty_path[-5:] if len(difficulty_path) > 5 else difficulty_path
        
        st.write("**Recent Progress:**")
        for i, step in enumerate(recent_steps, 1):
            status = "✅" if step.get('correct', False) else "❌"
            st.write(f"{status} Q{step.get('question_index', 0)+1}: Difficulty {step['difficulty']} → Score {step['score']}/10")
        
        # Show difficulty trend
        if len(difficulty_path) >= 3:
            recent_difficulties = [step['difficulty'] for step in difficulty_path[-3:]]
            if recent_difficulties[-1] > recent_difficulties[0]:
                st.success("📈 Trending upward in difficulty!")
            elif recent_difficulties[-1] < recent_difficulties[0]:
                st.info("📉 Focusing on strengthening fundamentals")
            else:
                st.info("🎯 Maintaining consistent challenge level")
    
    @staticmethod
    def display_report_download(evaluations: List[Dict], user_info: Dict) -> None:
        """Display report download functionality"""
        st.subheader("📄 Download Q&A + Scores")
        
        if st.button("📥 Generate Report"):
            # Generate report content
            report_content = UIComponents._generate_report_content(evaluations, user_info)
            
            st.download_button(
                label="Download as Text File",
                data=report_content,
                file_name=f"viva_evaluation_report_{user_info.get('username', 'user')}_{int(time.time())}.txt",
                mime="text/plain"
            )
    
    @staticmethod
    def _generate_report_content(evaluations: List[Dict], user_info: Dict) -> str:
        """Generate report content"""
        import io
        
        output = io.StringIO()
        output.write(f"Name: {user_info.get('full_name', user_info.get('username', 'N/A'))}\n")
        output.write(f"Subject: {user_info.get('subject', 'N/A')}\n")
        output.write(f"Book Title: {user_info.get('book_title', 'N/A')}\n\n")
        output.write("Structured Viva Questions, Answers, and Scores\n")
        output.write("=" * 70 + "\n\n")
        
        for i, eval_data in enumerate(evaluations, 1):
            output.write(f"[{i}] Difficulty: {eval_data.get('level', 'Unknown')}\n")
            output.write(f"Q: {eval_data.get('question', 'N/A')}\n")
            output.write(f"LLM Answer: {eval_data.get('answer', 'N/A')}\n")
            output.write(f"User Answer: {eval_data.get('user_answer', '[Not answered]')}\n")
            output.write(f"Score: {eval_data.get('score', '[Not evaluated]')} / 10\n")
            output.write("-" * 70 + "\n")
        
        return output.getvalue()
    
    @staticmethod
    def _get_difficulty_from_level(level: str) -> int:
        """Convert text levels to numeric difficulty for compatibility"""
        mapping = {
            'Basic': 3, 'Easy': 3,
            'Intermediate': 8, 'Moderate': 8, 
            'Advanced': 13, 'Difficult': 13,
            'Expert': 18
        }
        return mapping.get(level, 10)
    
    @staticmethod
    def display_audio_analysis(audio_data: np.ndarray, sample_rate: int = 44100) -> None:
        """Display FFT and audio analysis (integrated from Audio Training Lab)"""
        st.markdown("---")
        st.subheader("📊 Audio Analysis")
        
        try:
            import librosa
            import plotly.graph_objects as go
            import plotly.express as px
        except ImportError:
            st.warning("⚠️ Audio analysis requires librosa and plotly. Install with: pip install librosa plotly")
            return
        
        # Convert int16 to float32 if needed
        if audio_data.dtype == np.int16:
            audio_data = audio_data.astype(np.float32) / 32768.0
        
        # Basic info
        duration = len(audio_data) / sample_rate
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Duration", f"{duration:.2f}s")
        col2.metric("Sample Rate", f"{sample_rate} Hz")
        col3.metric("Samples", len(audio_data))
        col4.metric("Max Amplitude", f"{np.max(np.abs(audio_data)):.3f}")
        
        # Waveform
        st.markdown("#### 🌊 Waveform")
        fig_wave = UIComponents._create_waveform_plot(audio_data, sample_rate)
        st.plotly_chart(fig_wave, use_container_width=True)
        
        # Spectrogram
        st.markdown("#### 🎨 Spectrogram")
        fig_spec = UIComponents._create_spectrogram_plot(audio_data, sample_rate)
        st.plotly_chart(fig_spec, use_container_width=True)
        
        # Feature analysis
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### 📈 Spectral Features")
            features = UIComponents._extract_spectral_features(audio_data, sample_rate)
            
            st.metric("Spectral Centroid (Hz)", f"{features['spectral_centroid']:.1f}")
            st.metric("Spectral Rolloff (Hz)", f"{features['spectral_rolloff']:.1f}")
            st.metric("Zero Crossing Rate", f"{features['zcr']:.4f}")
            st.metric("RMS Energy", f"{features['rms']:.4f}")
        
        with col2:
            st.markdown("#### 🎵 MFCC Features")
            mfcc = librosa.feature.mfcc(y=audio_data, sr=sample_rate, n_mfcc=13)
            
            # MFCC heatmap
            fig_mfcc = px.imshow(
                mfcc,
                title="MFCC Coefficients",
                labels={'x': 'Time Frame', 'y': 'MFCC Coefficient', 'color': 'Value'},
                aspect='auto'
            )
            st.plotly_chart(fig_mfcc, use_container_width=True)
        
        # Frequency analysis (FFT)
        st.markdown("#### 🔊 Frequency Analysis (FFT)")
        fig_fft = UIComponents._create_fft_plot(audio_data, sample_rate)
        st.plotly_chart(fig_fft, use_container_width=True)
    
    @staticmethod
    def _create_waveform_plot(audio_data: np.ndarray, sample_rate: int):
        """Create waveform visualization"""
        try:
            import plotly.graph_objects as go
        except ImportError:
            return None
        
        time_axis = np.linspace(0, len(audio_data) / sample_rate, len(audio_data))
        
        # Downsample for display if too large
        if len(audio_data) > 50000:
            step = len(audio_data) // 50000
            time_axis = time_axis[::step]
            audio_data = audio_data[::step]
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=time_axis,
            y=audio_data,
            mode='lines',
            name='Waveform',
            line=dict(color='blue', width=1)
        ))
        
        fig.update_layout(
            title="Audio Waveform",
            xaxis_title="Time (seconds)",
            yaxis_title="Amplitude",
            height=300
        )
        
        return fig
    
    @staticmethod
    def _create_spectrogram_plot(audio_data: np.ndarray, sample_rate: int):
        """Create spectrogram visualization"""
        try:
            import librosa
            import plotly.graph_objects as go
            
            # Compute spectrogram
            D = librosa.stft(audio_data)
            S_db = librosa.amplitude_to_db(np.abs(D), ref=np.max)
            
            # Create time and frequency axes
            times = librosa.frames_to_time(np.arange(S_db.shape[1]), sr=sample_rate)
            freqs = librosa.fft_frequencies(sr=sample_rate)
            
            fig = go.Figure(data=go.Heatmap(
                z=S_db,
                x=times,
                y=freqs,
                colorscale='Viridis',
                colorbar=dict(title="dB")
            ))
            
            fig.update_layout(
                title="Spectrogram",
                xaxis_title="Time (seconds)",
                yaxis_title="Frequency (Hz)",
                height=400
            )
            
            return fig
        except Exception as e:
            st.error(f"Error creating spectrogram: {e}")
            try:
                import plotly.graph_objects as go
                return go.Figure()
            except ImportError:
                return None
    
    @staticmethod
    def _create_fft_plot(audio_data: np.ndarray, sample_rate: int):
        """Create FFT frequency analysis plot"""
        try:
            import plotly.graph_objects as go
        except ImportError:
            return None
        
        # Compute FFT
        fft = np.fft.fft(audio_data)
        freqs = np.fft.fftfreq(len(audio_data), 1/sample_rate)
        magnitude = np.abs(fft)
        
        # Only plot positive frequencies
        positive_freqs = freqs[:len(freqs)//2]
        positive_magnitude = magnitude[:len(magnitude)//2]
        
        # Downsample for display if too large
        if len(positive_freqs) > 10000:
            step = len(positive_freqs) // 10000
            positive_freqs = positive_freqs[::step]
            positive_magnitude = positive_magnitude[::step]
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=positive_freqs,
            y=positive_magnitude,
            mode='lines',
            name='Frequency Spectrum',
            line=dict(color='red')
        ))
        
        fig.update_layout(
            title="Frequency Spectrum (FFT)",
            xaxis_title="Frequency (Hz)",
            yaxis_title="Magnitude",
            height=300
        )
        
        return fig
    
    @staticmethod
    def _extract_spectral_features(audio_data: np.ndarray, sample_rate: int) -> Dict[str, float]:
        """Extract spectral features from audio"""
        try:
            import librosa
            
            # Spectral centroid
            spectral_centroid = librosa.feature.spectral_centroid(y=audio_data, sr=sample_rate)[0]
            
            # Spectral rolloff
            spectral_rolloff = librosa.feature.spectral_rolloff(y=audio_data, sr=sample_rate)[0]
            
            # Zero crossing rate
            zcr = librosa.feature.zero_crossing_rate(audio_data)[0]
            
            # RMS energy
            rms = librosa.feature.rms(y=audio_data)[0]
            
            return {
                'spectral_centroid': np.mean(spectral_centroid),
                'spectral_rolloff': np.mean(spectral_rolloff),
                'zcr': np.mean(zcr),
                'rms': np.mean(rms)
            }
        except Exception as e:
            st.warning(f"Error extracting features: {e}")
            return {
                'spectral_centroid': 0.0,
                'spectral_rolloff': 0.0,
                'zcr': 0.0,
                'rms': 0.0
            }
