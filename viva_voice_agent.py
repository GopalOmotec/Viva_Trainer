import os
import time
import tempfile
import json
import base64
import threading
import queue
import wave
from typing import Dict, List, Optional
from collections import deque
import re

import numpy as np
import streamlit as st
import sounddevice as sd
import scipy.io.wavfile as wav
import websocket
from streamlit_autorefresh import st_autorefresh
from openai import OpenAI
import streamlit.components.v1 as components

from mock_viva import InterviewerPersona, INTERVIEWER_PROFILES
from database import db_manager
from auth import auth_manager
from live_audio_visualizer import live_audio_interface

DEFAULT_CHAT_MODEL = os.getenv("OPENAI_VIVA_MENTOR_MODEL", "gpt-4o")
DEFAULT_TRANSCRIBE_MODEL = os.getenv("OPENAI_AUDIO_TRANSCRIBE_MODEL", "gpt-4o-mini-transcribe")
DEFAULT_TTS_MODEL = os.getenv("OPENAI_AUDIO_TTS_MODEL", "gpt-4o-mini-tts")
DEFAULT_TTS_VOICE = os.getenv("OPENAI_AUDIO_TTS_VOICE", "alloy")
DEFAULT_REALTIME_MODEL = os.getenv("OPENAI_REALTIME_MODEL", "gpt-realtime")
REALTIME_VOICES = ["marin", "cedar", "alloy", "verse", "coral", "sage", "echo", "ash", "ballad", "shimmer"]


def _get_openai_client() -> OpenAI:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        st.error("⚠️ Please set OPENAI_API_KEY in your .env file")
        st.stop()
    return OpenAI(api_key=api_key)


def _get_openai_api_key() -> str:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        st.error("⚠️ Please set OPENAI_API_KEY in your .env file")
        st.stop()
    return api_key


def _initialize_state() -> None:
    defaults = {
        "voice_viva_mode": "Standard",
        "voice_viva_sm_mode": False,
        "voice_viva_active": False,
        "voice_viva_messages": [],
        "voice_viva_topic": "",
        "voice_viva_persona": None,
        "voice_viva_transcript": None,
        "voice_viva_last_reply": None,
        "voice_viva_last_question": "",
        "voice_viva_audio_path": None,
        "voice_viva_voice": DEFAULT_TTS_VOICE,
        "voice_viva_max_questions": 5,
        "voice_viva_questions_asked": 0,
        "voice_viva_complete": False,
        "voice_viva_question_list": [],
        "voice_viva_realtime_active": False,
        "voice_viva_realtime_controller": None,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def _build_system_prompt(profile, topic: str, question_list: Optional[List[str]] = None,
                         max_questions: Optional[int] = None,
                         selective_mutism_mode: bool = False) -> str:
    prompt = (
        f"You are {profile.name}, {profile.title}.\n"
        f"Style: {profile.style}. Follow-up style: {profile.follow_up_style}.\n"
        f"Characteristics: {', '.join(profile.characteristics)}.\n\n"
        "You are running a demo viva voce session as a voice mentor.\n"
        f"Topic focus: {topic}.\n"
        "Rules:\n"
        "- Ask one question at a time.\n"
        "- After each student answer, give brief feedback (1-2 sentences) then ask the next question.\n"
        "- Keep responses under 120 words.\n"
        "- If the user asks to end, provide a short summary and say 'Session complete.'\n"
    )
    if selective_mutism_mode:
        prompt += (
            "\nSelective mutism support:\n"
            "- Use gentle, positive, and encouraging language.\n"
            "- Praise any attempt to answer, even partial.\n"
            "- Offer a small hint or prompt to help the student respond.\n"
            "- Keep tone calm and supportive.\n"
        )
    if question_list:
        questions = "\n".join([f"{idx + 1}. {q}" for idx, q in enumerate(question_list)])
        total = max_questions or len(question_list)
        prompt += (
            "\nUse the exact questions below in order. Do not invent new questions.\n"
            f"You must ask exactly {total} questions total.\n"
            f"Questions:\n{questions}\n"
        )
    return prompt


def _truncate_messages(messages: List[Dict], max_messages: int = 18) -> List[Dict]:
    if len(messages) <= max_messages:
        return messages
    system = messages[0]
    recent = messages[-(max_messages - 1):]
    return [system, *recent]


def _generate_response(client: OpenAI, messages: List[Dict], model: str) -> str:
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=0.6,
    )
    content = response.choices[0].message.content
    return content.strip() if content else ""


def _generate_response_streaming(
    client: OpenAI, messages: List[Dict], model: str, placeholder
) -> str:
    stream = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=0.6,
        stream=True,
    )
    collected = []
    for chunk in stream:
        delta = chunk.choices[0].delta
        if delta and delta.content:
            collected.append(delta.content)
            placeholder.markdown("".join(collected))
    return "".join(collected).strip()


def _record_audio(duration_seconds: int, sample_rate: int) -> np.ndarray:
    audio = sd.rec(
        int(duration_seconds * sample_rate),
        samplerate=sample_rate,
        channels=1,
        dtype="int16",
    )
    progress = st.progress(0)
    for i in range(duration_seconds):
        time.sleep(1)
        progress.progress((i + 1) / duration_seconds)
    sd.wait()
    return audio.flatten()


def _transcribe_audio(client: OpenAI, audio_data: np.ndarray, sample_rate: int, model: str) -> str:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        wav.write(tmp.name, sample_rate, audio_data)
        tmp_path = tmp.name
    try:
        with open(tmp_path, "rb") as audio_file:
            result = client.audio.transcriptions.create(
                model=model,
                file=audio_file,
            )
        return result.text.strip() if hasattr(result, "text") else str(result).strip()
    finally:
        try:
            os.remove(tmp_path)
        except OSError:
            pass


def _synthesize_speech(client: OpenAI, text: str, model: str, voice: str) -> str:
    if not text:
        return ""
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tmp:
        audio = client.audio.speech.create(
            model=model,
            voice=voice,
            input=text,
        )
        audio.write_to_file(tmp.name)
        return tmp.name


def _autoplay_audio(path: str, mime: str) -> None:
    if not path or not os.path.exists(path):
        return
    with open(path, "rb") as audio_file:
        data = audio_file.read()
    b64 = base64.b64encode(data).decode("ascii")
    html = f"""
    <audio autoplay>
      <source src="data:{mime};base64,{b64}" type="{mime}">
    </audio>
    """
    components.html(html, height=0, width=0)


def _extract_feedback(text: str, max_sentences: int = 2) -> str:
    if not text:
        return ""
    parts = re.split(r"(?<=[.!?])\s+", text.strip())
    feedback = " ".join(parts[:max_sentences]).strip()
    return feedback or text.strip()


def _extract_last_question(text: str) -> str:
    if not text:
        return ""
    questions = re.findall(r"[^.?!]*\?", text)
    return questions[-1].strip() if questions else ""


def _configure_question_source(current_user: Optional[Dict], key_prefix: str = "voice") -> (List[str], int):
    source = st.selectbox(
        "Question Source",
        [
            "AI Generated (Auto)",
            "PDF Questions (Current Upload)",
            "Question Bank (Database)",
            "Previous PDF Sessions (Database)",
        ],
        key=f"{key_prefix}_qs_source",
    )

    if source == "AI Generated (Auto)":
        max_questions = st.slider(
            "Number of Questions", 3, 12, 5, key=f"{key_prefix}_qs_auto_count"
        )
        return [], max_questions

    question_list: List[str] = []

    if source == "PDF Questions (Current Upload)":
        pdf_questions = st.session_state.get("all_qas", [])
        question_list = [q.get("question") for q in pdf_questions if q.get("question")]
        if not question_list:
            st.warning("No PDF-generated questions found. Upload a PDF first.")
            return [], 0

    elif source == "Question Bank (Database)":
        subjects = db_manager.get_subjects()
        if not subjects:
            st.warning("No subjects found in the question bank.")
            return [], 0
        selected_subject = st.selectbox(
            "Subject", options=[s["name"] for s in subjects], key=f"{key_prefix}_qs_subject"
        )
        subject_id = next((s["id"] for s in subjects if s["name"] == selected_subject), None)

        topics = db_manager.get_topics_by_subject(subject_id) if subject_id else []
        topic_options = ["All Topics"] + [t["name"] for t in topics]
        selected_topic = st.selectbox("Topic", options=topic_options, key=f"{key_prefix}_qs_topic")
        topic_id = None
        if selected_topic != "All Topics":
            topic_id = next((t["id"] for t in topics if t["name"] == selected_topic), None)

        grade = st.text_input("Grade (optional)", value="", key=f"{key_prefix}_qs_grade")
        col1, col2 = st.columns(2)
        with col1:
            diff_min = st.slider("Min Difficulty", 1.0, 100.0, 1.0, 1.0, key=f"{key_prefix}_qs_dmin")
        with col2:
            diff_max = st.slider("Max Difficulty", 1.0, 100.0, 100.0, 1.0, key=f"{key_prefix}_qs_dmax")

        max_questions = st.slider(
            "Number of Questions", 3, 12, 5, key=f"{key_prefix}_qs_db_count"
        )
        questions = db_manager.get_predefined_questions(
            subject_id=subject_id,
            topic_id=topic_id,
            grade=grade if grade else None,
            difficulty_min=diff_min,
            difficulty_max=diff_max,
            limit=max_questions,
        )
        question_list = [q.get("question") for q in questions if q.get("question")]
        if not question_list:
            st.warning("No questions matched those filters.")
            return [], 0

        return question_list, len(question_list)

    elif source == "Previous PDF Sessions (Database)":
        if not current_user:
            st.warning("Please log in to access saved sessions.")
            return [], 0
        conversations = db_manager.get_user_conversations(current_user["id"])
        if not conversations:
            st.warning("No saved PDF sessions found.")
            return [], 0
        selected_conv = st.selectbox(
            "Choose Session",
            options=conversations,
            format_func=lambda c: f"{c['subject']} - {c['book_title']} ({c['created_at']})",
            key=f"{key_prefix}_qs_conv",
        )
        conv_questions = db_manager.get_conversation_questions(selected_conv["id"])
        question_list = [q.get("question") for q in conv_questions if q.get("question")]
        if not question_list:
            st.warning("No questions found for that session.")
            return [], 0

    max_questions = st.slider(
        "Number of Questions",
        1,
        min(12, len(question_list)),
        min(5, len(question_list)),
        key=f"{key_prefix}_qs_list_count",
    )
    return question_list[:max_questions], max_questions


def _float32_to_pcm16_bytes(float32_array: np.ndarray) -> bytes:
    clipped = np.clip(float32_array, -1.0, 1.0)
    pcm16 = (clipped * 32767).astype(np.int16)
    return pcm16.tobytes()


def _write_pcm16_wav(pcm_bytes: bytes, sample_rate: int) -> str:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        with wave.open(tmp.name, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(sample_rate)
            wf.writeframes(pcm_bytes)
        return tmp.name


class RealtimeVivaController:
    def __init__(
        self,
        api_key: str,
        topic: str,
        persona: InterviewerPersona,
        max_questions: int,
        voice: str,
        question_list: Optional[List[str]] = None,
        selective_mutism_mode: bool = False,
        sample_rate: int = 24000,
    ):
        self.api_key = api_key
        self.topic = topic
        self.persona = persona
        self.max_questions = max_questions
        self.voice = voice
        self.question_list = question_list or []
        self.selective_mutism_mode = selective_mutism_mode
        self.sample_rate = sample_rate

        self.ws_app: Optional[websocket.WebSocketApp] = None
        self.ws_thread: Optional[threading.Thread] = None
        self.sender_thread: Optional[threading.Thread] = None
        self.audio_stream: Optional[sd.InputStream] = None

        self.audio_queue: "queue.Queue[str]" = queue.Queue(maxsize=200)
        self.stop_event = threading.Event()

        self.status = "Idle"
        self.latest_text = ""
        self.latest_transcript = ""
        self.latest_audio_path: Optional[str] = None

        self._current_output_audio = bytearray()
        self._current_output_text = ""
        self.completed = False
        self.waveform_buffer: "deque[float]" = deque(maxlen=self.sample_rate * 2)
        self.waveform_lock = threading.Lock()

    def start(self) -> None:
        url = f"wss://api.openai.com/v1/realtime?model={DEFAULT_REALTIME_MODEL}"
        headers = [f"Authorization: Bearer {self.api_key}"]

        self.ws_app = websocket.WebSocketApp(
            url,
            header=headers,
            on_open=self._on_open,
            on_message=self._on_message,
            on_error=self._on_error,
            on_close=self._on_close,
        )

        self.ws_thread = threading.Thread(target=self.ws_app.run_forever, daemon=True)
        self.ws_thread.start()

        self.sender_thread = threading.Thread(target=self._sender_loop, daemon=True)
        self.sender_thread.start()

    def stop(self) -> None:
        self.stop_event.set()
        if self.audio_stream:
            try:
                self.audio_stream.stop()
                self.audio_stream.close()
            except Exception:
                pass
            self.audio_stream = None
        if self.ws_app:
            try:
                self.ws_app.close()
            except Exception:
                pass
        if self.latest_audio_path:
            try:
                os.remove(self.latest_audio_path)
            except OSError:
                pass
            self.latest_audio_path = None
        self.status = "Stopped"

    def _sender_loop(self) -> None:
        while not self.stop_event.is_set():
            try:
                chunk = self.audio_queue.get(timeout=0.2)
            except queue.Empty:
                continue
            if self.ws_app:
                event = {"type": "input_audio_buffer.append", "audio": chunk}
                try:
                    self.ws_app.send(json.dumps(event))
                except Exception:
                    self.stop_event.set()
                    break

    def _start_audio_stream(self) -> None:
        if self.audio_stream:
            return

        def callback(indata, frames, time_info, status):
            if self.stop_event.is_set():
                return
            with self.waveform_lock:
                self.waveform_buffer.extend(indata[:, 0].tolist())
            audio_bytes = _float32_to_pcm16_bytes(indata[:, 0])
            b64 = base64.b64encode(audio_bytes).decode("ascii")
            try:
                self.audio_queue.put_nowait(b64)
            except queue.Full:
                pass

        self.audio_stream = sd.InputStream(
            samplerate=self.sample_rate,
            channels=1,
            dtype="float32",
            callback=callback,
        )
        self.audio_stream.start()

    def get_waveform_snapshot(self, max_points: int = 1200) -> np.ndarray:
        with self.waveform_lock:
            data = np.array(self.waveform_buffer, dtype=np.float32)
        if data.size == 0:
            return data
        if data.size > max_points:
            step = max(1, data.size // max_points)
            data = data[::step]
        return data

    def _on_open(self, ws):
        profile = INTERVIEWER_PROFILES[self.persona]
        system_prompt = _build_system_prompt(
            profile,
            self.topic,
            question_list=self.question_list,
            max_questions=self.max_questions,
            selective_mutism_mode=self.selective_mutism_mode,
        )
        system_prompt += "\nAfter the last answer, provide a brief summary and say 'Session complete.'"

        session_update = {
            "type": "session.update",
            "session": {
                "type": "realtime",
                "output_modalities": ["audio", "text"],
                "audio": {
                    "input": {
                        "format": {"type": "audio/pcm", "rate": self.sample_rate},
                        "turn_detection": {"type": "semantic_vad"},
                    },
                    "output": {"format": {"type": "audio/pcm"}, "voice": self.voice},
                },
                "instructions": system_prompt,
            },
        }
        ws.send(json.dumps(session_update))

        if self.question_list:
            kickoff = (
                "Greet the student and ask exactly this question: "
                f"{self.question_list[0]}"
            )
        else:
            kickoff = (
                "Greet the student and ask a topic-specific question about "
                f"{self.topic}. Ask question 1 of {self.max_questions}."
            )
        ws.send(
            json.dumps(
                {
                    "type": "conversation.item.create",
                    "item": {
                        "type": "message",
                        "role": "user",
                        "content": [{"type": "input_text", "text": kickoff}],
                    },
                }
            )
        )
        ws.send(json.dumps({"type": "response.create", "response": {"output_modalities": ["audio", "text"]}}))
        self.status = "Listening"
        self._start_audio_stream()

    def _on_message(self, ws, message):
        try:
            data = json.loads(message)
        except json.JSONDecodeError:
            return

        event_type = data.get("type")

        if event_type in ("response.output_audio.delta", "response.audio.delta"):
            delta = data.get("delta") or data.get("audio")
            if delta:
                self._current_output_audio.extend(base64.b64decode(delta))
            return

        if event_type in ("response.output_audio_transcript.delta", "response.output_text.delta"):
            delta = data.get("delta")
            if delta:
                self._current_output_text += delta
                self.latest_text = self._current_output_text
            return

        if event_type in ("response.output_audio_transcript.done", "response.output_text.done"):
            text = data.get("text")
            if text:
                self.latest_text = text
            return

        if event_type == "response.done":
            if self._current_output_audio:
                if self.latest_audio_path:
                    try:
                        os.remove(self.latest_audio_path)
                    except OSError:
                        pass
                self.latest_audio_path = _write_pcm16_wav(
                    bytes(self._current_output_audio), self.sample_rate
                )
                self._current_output_audio = bytearray()

            response = data.get("response", {})
            output_items = response.get("output", [])
            for item in output_items:
                if item.get("type") == "message":
                    content = item.get("content", [])
                    for part in content:
                        if part.get("type") in ("output_text", "output_audio_transcript"):
                            text = part.get("text", "")
                            if text:
                                self.latest_text = text
                                if "session complete" in text.lower():
                                    self.completed = True
                                    self.status = "Completed"
                                    self.stop_event.set()
            return

        if event_type == "input_audio_buffer.speech_started":
            self.status = "User speaking"
            return

        if event_type == "input_audio_buffer.speech_stopped":
            self.status = "Processing"
            return

    def _on_error(self, ws, error):
        self.status = f"Error: {error}"
        self.stop_event.set()

    def _on_close(self, ws, close_status_code, close_msg):
        if not self.stop_event.is_set():
            self.status = "Closed"
        self.stop_event.set()


def display_realtime_viva_agent() -> None:
    st.markdown("### Realtime Hands-Free Setup")

    if not st.session_state.voice_viva_realtime_active:
        col1, col2 = st.columns([2, 1])

        with col1:
            topic = st.text_input(
                "Viva Topic",
                placeholder="e.g., Data Structures, Networking, Machine Learning",
                value="General Viva Practice",
                key="rt_topic",
            )
            persona = st.selectbox(
                "Mentor Persona",
                options=list(INTERVIEWER_PROFILES.keys()),
                format_func=lambda p: f"{INTERVIEWER_PROFILES[p].emoji} {INTERVIEWER_PROFILES[p].name}",
                key="rt_persona",
            )
            current_user = auth_manager.get_current_user()
            question_list, max_questions = _configure_question_source(
                current_user, key_prefix="rt"
            )
            st.session_state.voice_viva_sm_mode = st.checkbox(
                "Selective Mutism Support (extra encouragement)", value=False, key="rt_sm"
            )
        with col2:
            default_voice = DEFAULT_TTS_VOICE if DEFAULT_TTS_VOICE in REALTIME_VOICES else REALTIME_VOICES[0]
            default_index = REALTIME_VOICES.index(default_voice)
            voice = st.selectbox(
                "Realtime Voice",
                options=REALTIME_VOICES,
                index=default_index,
                help="Recommended: marin or cedar for best quality",
                key="rt_voice",
            )
            st.caption(f"Realtime model: {DEFAULT_REALTIME_MODEL}")

        if st.button("🚀 Start Realtime Viva", type="primary", use_container_width=True):
            api_key = _get_openai_api_key()
            if max_questions == 0:
                st.warning("Please select a valid question source first.")
                return
            controller = RealtimeVivaController(
                api_key=api_key,
                topic=topic,
                persona=persona,
                max_questions=max_questions,
                voice=voice,
                question_list=question_list,
                selective_mutism_mode=st.session_state.voice_viva_sm_mode,
            )
            controller.start()
            st.session_state.voice_viva_realtime_controller = controller
            st.session_state.voice_viva_realtime_active = True
            st.rerun()
        return

    st_autorefresh(interval=1000, key="realtime_refresh")

    controller: Optional[RealtimeVivaController] = st.session_state.voice_viva_realtime_controller
    if not controller:
        st.warning("Realtime controller not available.")
        st.session_state.voice_viva_realtime_active = False
        return

    st.markdown("### Realtime Session")
    st.markdown(f"**Status:** {controller.status}")

    waveform = controller.get_waveform_snapshot()
    if waveform.size > 0:
        st.line_chart(waveform, height=200)

    col1, col2 = st.columns([3, 1])
    with col1:
        st.markdown("#### Mentor Response (Live)")
        if controller.latest_text:
            st.markdown(controller.latest_text)
        else:
            st.caption("Waiting for the mentor response...")
    with col2:
        if st.button("🛑 Stop Realtime Session", use_container_width=True):
            controller.stop()
            st.session_state.voice_viva_realtime_controller = None
            st.session_state.voice_viva_realtime_active = False
            st.rerun()

    if controller.latest_audio_path:
        st.audio(controller.latest_audio_path, format="audio/wav")
        _autoplay_audio(controller.latest_audio_path, "audio/wav")

    if controller.completed:
        st.success("Session complete. Start a new session to continue practicing.")
        controller.stop()
        st.session_state.voice_viva_realtime_controller = None
        st.session_state.voice_viva_realtime_active = False


def display_viva_voice_agent() -> None:
    _initialize_state()

    st.markdown(
        """
        <div style="background: linear-gradient(135deg, #e0f2fe 0%, #bae6fd 100%); 
                    border-radius: 16px; padding: 1.5rem; margin-bottom: 1rem;
                    border-left: 4px solid #0ea5e9;">
            <h2 style="color: #0369a1; margin: 0;">🎙️ Viva Voice Mentor (Demo)</h2>
            <p style="color: #64748b; margin: 0.5rem 0 0 0;">
                Interactive, voice-driven viva practice powered by OpenAI audio models
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    mode = st.radio(
        "Voice Mode",
        ["Standard (Push-to-answer)", "Realtime Hands-Free"],
        horizontal=True,
    )
    st.session_state.voice_viva_mode = mode

    if mode == "Realtime Hands-Free":
        display_realtime_viva_agent()
        return

    client = _get_openai_client()

    if not st.session_state.voice_viva_active:
        st.markdown("### Setup")
        col1, col2 = st.columns([2, 1])

        with col1:
            topic = st.text_input(
                "Viva Topic",
                placeholder="e.g., Data Structures, Networking, Machine Learning",
                value="General Viva Practice",
            )
            persona = st.selectbox(
                "Mentor Persona",
                options=list(INTERVIEWER_PROFILES.keys()),
                format_func=lambda p: f"{INTERVIEWER_PROFILES[p].emoji} {INTERVIEWER_PROFILES[p].name}",
            )
            current_user = auth_manager.get_current_user()
            question_list, max_questions = _configure_question_source(
                current_user, key_prefix="voice"
            )
            st.session_state.voice_viva_sm_mode = st.checkbox(
                "Selective Mutism Support", value=False, key="voice_sm"
            )
        with col2:
            voice = st.text_input(
                "TTS Voice",
                value=st.session_state.voice_viva_voice or DEFAULT_TTS_VOICE,
                help="OpenAI voice name, e.g., alloy, verse, coral, sage",
            )
            st.caption(f"Chat model: {DEFAULT_CHAT_MODEL}")

        if st.button("🚀 Start Viva Voice Demo", type="primary", use_container_width=True):
            if max_questions == 0:
                st.warning("Please select a valid question source first.")
                return
            profile = INTERVIEWER_PROFILES[persona]
            system_prompt = _build_system_prompt(
                profile,
                topic,
                question_list=question_list,
                max_questions=max_questions,
                selective_mutism_mode=st.session_state.voice_viva_sm_mode,
            )
            st.session_state.voice_viva_messages = [{"role": "system", "content": system_prompt}]
            st.session_state.voice_viva_active = True
            st.session_state.voice_viva_topic = topic
            st.session_state.voice_viva_persona = persona.value
            st.session_state.voice_viva_voice = voice or DEFAULT_TTS_VOICE
            st.session_state.voice_viva_max_questions = max_questions
            st.session_state.voice_viva_questions_asked = 0
            st.session_state.voice_viva_complete = False
            st.session_state.voice_viva_question_list = question_list
            st.session_state.voice_viva_last_question = ""

            if question_list:
                kickoff = (
                    "Greet the student and ask exactly this question: "
                    f"{question_list[0]}"
                )
            else:
                kickoff = (
                    "Greet the student and ask a topic-specific question about "
                    f"{topic}. Ask question 1 of {st.session_state.voice_viva_max_questions}."
                )
            prompt_messages = _truncate_messages(
                st.session_state.voice_viva_messages + [{"role": "user", "content": kickoff}]
            )
            streaming_placeholder = st.empty()
            reply = _generate_response_streaming(
                client, prompt_messages, DEFAULT_CHAT_MODEL, streaming_placeholder
            )
            st.session_state.voice_viva_messages.append({"role": "assistant", "content": reply})
            st.session_state.voice_viva_last_reply = reply
            st.session_state.voice_viva_last_question = _extract_last_question(reply)
            st.session_state.voice_viva_audio_path = _synthesize_speech(
                client, reply, DEFAULT_TTS_MODEL, st.session_state.voice_viva_voice
            )
            st.session_state.voice_viva_questions_asked = 1
            st.rerun()

        return

    profile = INTERVIEWER_PROFILES[InterviewerPersona(st.session_state.voice_viva_persona)]
    st.markdown(f"### {profile.emoji} {profile.name} — {profile.title}")
    st.caption(
        f"Topic: {st.session_state.voice_viva_topic} | "
        f"Question {st.session_state.voice_viva_questions_asked}/"
        f"{st.session_state.voice_viva_max_questions}"
    )

    col1, col2 = st.columns([3, 1])
    with col1:
        st.markdown("#### Conversation")
    with col2:
        if st.button("🛑 End Session", use_container_width=True):
            if st.session_state.voice_viva_audio_path:
                try:
                    os.remove(st.session_state.voice_viva_audio_path)
                except OSError:
                    pass
            st.session_state.voice_viva_active = False
            st.session_state.voice_viva_messages = []
            st.session_state.voice_viva_transcript = None
            st.session_state.voice_viva_last_reply = None
            st.session_state.voice_viva_last_question = ""
            st.session_state.voice_viva_audio_path = None
            st.session_state.voice_viva_complete = False
            st.session_state.voice_viva_question_list = []
            st.rerun()

    for turn in st.session_state.voice_viva_messages:
        if turn["role"] == "assistant":
            st.markdown(f"**{profile.emoji} {profile.name}:** {turn['content']}")
        elif turn["role"] == "user":
            st.markdown(f"**🎓 You:** {turn['content']}")

    if st.session_state.voice_viva_audio_path:
        st.audio(st.session_state.voice_viva_audio_path, format="audio/mp3")
        _autoplay_audio(st.session_state.voice_viva_audio_path, "audio/mp3")

    st.markdown("---")
    st.markdown("#### Your Voice Answer")
    if st.session_state.voice_viva_complete:
        st.success("Session complete. Start a new session to continue practicing.")
    else:
        st.info("Waiting for your response. Record when ready.")
        duration = st.slider("Recording duration (seconds)", 3, 30, 8)
        use_live_waveform = st.checkbox("Show live waveform", value=True)
        sample_rate = 16000

        if st.button("🎙️ Record Answer", type="primary"):
            try:
                if use_live_waveform:
                    audio_data, sample_rate = live_audio_interface.record_with_visualization(duration)
                else:
                    audio_data = _record_audio(duration, sample_rate)
                if audio_data is None or len(audio_data) == 0:
                    st.warning("No audio captured. Please try again.")
                    return
                transcript = _transcribe_audio(
                    client, audio_data, sample_rate, DEFAULT_TRANSCRIBE_MODEL
                )
                st.session_state.voice_viva_transcript = transcript
                st.rerun()
            except Exception as e:
                st.error(f"Recording or transcription failed: {e}")

    if st.session_state.voice_viva_transcript:
        edited = st.text_area(
            "Transcribed text (edit if needed)",
            value=st.session_state.voice_viva_transcript,
            height=120,
        )
        if st.button("📤 Send Answer", type="primary"):
            if edited.strip():
                st.session_state.voice_viva_messages.append(
                    {"role": "user", "content": edited.strip()}
                )
                next_index = st.session_state.voice_viva_questions_asked + 1
                question_list = st.session_state.get("voice_viva_question_list") or []
                if next_index > st.session_state.voice_viva_max_questions:
                    followup = (
                        "Provide a brief summary of the student's performance and end with "
                        "'Session complete.'"
                    )
                else:
                    if question_list and next_index <= len(question_list):
                        followup = (
                            "Provide brief feedback (1-2 sentences) only. Do not ask any question."
                        )
                    else:
                        last_q = st.session_state.voice_viva_last_question
                        followup = (
                            "Provide brief feedback (1-2 sentences) then ask a NEW "
                            f"topic-specific question about {st.session_state.voice_viva_topic}. "
                            f"Do not repeat previous questions. Last question: {last_q}. "
                            f"This is question {next_index} of {st.session_state.voice_viva_max_questions}."
                        )
                if st.session_state.voice_viva_sm_mode:
                    followup = (
                        "Be gentle and encouraging. Praise any attempt, offer a small hint. "
                        + followup
                    )
                prompt_messages = _truncate_messages(
                    st.session_state.voice_viva_messages + [{"role": "user", "content": followup}]
                )
                streaming_placeholder = st.empty()
                reply = _generate_response_streaming(
                    client, prompt_messages, DEFAULT_CHAT_MODEL, streaming_placeholder
                )
                if question_list and next_index <= len(question_list):
                    feedback_only = _extract_feedback(reply, max_sentences=2)
                    reply = f"{feedback_only}\n\nNext question: {question_list[next_index - 1]}"
                    st.session_state.voice_viva_last_question = question_list[next_index - 1]
                else:
                    st.session_state.voice_viva_last_question = _extract_last_question(reply)
                st.session_state.voice_viva_messages.append(
                    {"role": "assistant", "content": reply}
                )
                st.session_state.voice_viva_last_reply = reply
                if st.session_state.voice_viva_audio_path:
                    try:
                        os.remove(st.session_state.voice_viva_audio_path)
                    except OSError:
                        pass
                st.session_state.voice_viva_audio_path = _synthesize_speech(
                    client, reply, DEFAULT_TTS_MODEL, st.session_state.voice_viva_voice
                )
                if next_index <= st.session_state.voice_viva_max_questions:
                    st.session_state.voice_viva_questions_asked = next_index
                else:
                    st.session_state.voice_viva_complete = True
                st.session_state.voice_viva_transcript = None
                st.rerun()
            else:
                st.warning("Please provide a response before sending.")
