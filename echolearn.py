# -*- coding: utf-8 -*-
"""
EchoLearn - Viva Question Evaluator (Refactored)
Main application file using modular architecture
"""

from pathlib import Path
import sys

APP_ROOT = Path(__file__).resolve().parent
if str(APP_ROOT) not in sys.path:
    sys.path.insert(0, str(APP_ROOT))

import streamlit as st
import fitz  # PyMuPDF
from langchain_community.llms import OpenAI
from dotenv import load_dotenv
import os
import time
from auth import auth_manager
from database import db_manager

# Import our new modules
from scoring import AnswerEvaluator, ScoringAnalytics
from ui_components import UIComponents
from question_manager import QuestionManager
from adaptive_learning import AdaptiveLearningEngine
from selective_mutism_support import SelectiveMutismSupport
from audio_lab import audio_lab
from viva_voice_agent import display_viva_voice_agent

# Import NEW feature modules
from voice_analysis import VoiceAnalyzer, VoiceCoach, display_voice_analysis_ui
from mock_viva import MockVivaInterviewer, MockVivaUI, InterviewerPersona, INTERVIEWER_PROFILES
from smart_analytics import SmartAnalytics, AnalyticsDashboard

# ------------------ Load API & Init Model ------------------
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")

if not openai_api_key:
    st.error("⚠️ Please set OPENAI_API_KEY in your .env file")
    st.stop()

llm = OpenAI(openai_api_key=openai_api_key, temperature=0)

# Initialize modules
scoring_evaluator = AnswerEvaluator(llm)
question_manager = QuestionManager(llm)
adaptive_engine = AdaptiveLearningEngine()
selective_mutism_support = SelectiveMutismSupport()

# Initialize NEW feature modules
voice_analyzer = VoiceAnalyzer()
voice_coach = VoiceCoach()
mock_viva_interviewer = MockVivaInterviewer(llm)
mock_viva_ui = MockVivaUI(llm)
smart_analytics = SmartAnalytics(db_manager)
analytics_dashboard = AnalyticsDashboard(smart_analytics)

# ------------------ Authentication Check ------------------
auth_manager.require_authentication()

# Show user profile in sidebar
auth_manager.show_user_profile_sidebar()

# Get current user
current_user = auth_manager.get_current_user()

# ==================== MODERN LIGHT THEME ====================
st.markdown("""
<style>
    /* Clean, modern light background */
    .stApp {
        background: linear-gradient(180deg, #ffffff 0%, #f0f4ff 100%);
    }
    
    /* Main content area */
    .main .block-container {
        max-width: 1000px;
        padding: 2rem 2rem 4rem 2rem;
    }
    
    /* Headers styling */
    h1 {
        color: #1e3a5f !important;
        font-weight: 700 !important;
        letter-spacing: -0.5px;
    }
    
    h2, h3 {
        color: #334155 !important;
        font-weight: 600 !important;
    }
    
    /* Primary buttons */
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 12px;
        padding: 0.6rem 1.5rem;
        font-weight: 600;
        font-size: 0.95rem;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.3);
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(102, 126, 234, 0.4);
    }
    
    /* Secondary (white/outline) buttons for specific elements */
    .stButton > button[kind="secondary"] {
        background: white;
        color: #667eea;
        border: 2px solid #667eea;
    }
    
    /* Selectbox styling */
    .stSelectbox > div > div {
        background: white;
        border: 2px solid #e2e8f0;
        border-radius: 12px;
        transition: border-color 0.2s ease;
    }
    
    .stSelectbox > div > div:focus-within {
        border-color: #667eea;
        box-shadow: 0 0 0 3px rgba(102, 126, 234, 0.1);
    }
    
    /* Text input styling */
    .stTextInput > div > div > input {
        background: white;
        border: 2px solid #e2e8f0;
        border-radius: 12px;
        padding: 0.75rem 1rem;
        font-size: 1rem;
        transition: all 0.2s ease;
    }
    
    .stTextInput > div > div > input:focus {
        border-color: #667eea;
        box-shadow: 0 0 0 3px rgba(102, 126, 234, 0.1);
    }
    
    /* Text area styling */
    .stTextArea > div > div > textarea {
        background: white;
        border: 2px solid #e2e8f0;
        border-radius: 12px;
        padding: 0.75rem 1rem;
        font-size: 1rem;
    }
    
    .stTextArea > div > div > textarea:focus {
        border-color: #667eea;
        box-shadow: 0 0 0 3px rgba(102, 126, 234, 0.1);
    }
    
    /* Slider styling */
    .stSlider > div > div > div {
        background: #e2e8f0;
    }
    
    .stSlider > div > div > div > div {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    }
    
    /* File uploader */
    [data-testid="stFileUploader"] {
        background: white;
        border: 2px dashed #cbd5e1;
        border-radius: 16px;
        padding: 1.5rem;
        transition: all 0.3s ease;
    }
    
    [data-testid="stFileUploader"]:hover {
        border-color: #667eea;
        background: #f8faff;
    }
    
    /* Info/success/warning boxes */
    .stAlert {
        border-radius: 12px;
        border: none;
    }
    
    /* Expanders */
    .streamlit-expanderHeader {
        background: white;
        border-radius: 12px;
        font-weight: 600;
    }
    
    /* Progress bar */
    .stProgress > div > div {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 10px;
    }
    
    /* Metrics */
    [data-testid="metric-container"] {
        background: white;
        border-radius: 12px;
        padding: 1rem;
        box-shadow: 0 2px 10px rgba(0,0,0,0.05);
    }
    
    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background: #f1f5f9;
        padding: 0.5rem;
        border-radius: 12px;
    }
    
    .stTabs [data-baseweb="tab"] {
        border-radius: 8px;
        padding: 0.5rem 1rem;
        font-weight: 500;
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
    }
    
    /* Dividers */
    hr {
        border: none;
        height: 1px;
        background: linear-gradient(90deg, transparent, #e2e8f0, transparent);
        margin: 1.5rem 0;
    }
    
    /* Radio buttons */
    .stRadio > div {
        display: flex;
        flex-direction: row;
        gap: 1rem;
        flex-wrap: wrap;
    }
    
    .stRadio > div > label {
        background: white;
        padding: 0.75rem 1.25rem;
        border-radius: 10px;
        border: 2px solid #e2e8f0;
        cursor: pointer;
        transition: all 0.2s ease;
    }
    
    .stRadio > div > label:hover {
        border-color: #667eea;
        background: #f8faff;
    }
    
    /* Hide Streamlit elements */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    
    /* Custom card styling for markdown content */
    .custom-card {
        background: white;
        border-radius: 16px;
        padding: 1.5rem;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.05);
        border: 1px solid #e2e8f0;
        margin: 1rem 0;
    }
    
    /* Question display card */
    .question-display {
        background: linear-gradient(135deg, #ffffff 0%, #f8faff 100%);
        border-radius: 16px;
        padding: 1.5rem;
        border-left: 4px solid #667eea;
        margin: 1rem 0;
        box-shadow: 0 2px 10px rgba(102, 126, 234, 0.1);
    }
    
    /* Success message styling */
    .success-badge {
        background: linear-gradient(135deg, #10b981 0%, #059669 100%);
        color: white;
        padding: 0.5rem 1rem;
        border-radius: 20px;
        display: inline-block;
        font-weight: 600;
    }
    
    /* Recording button animation */
    @keyframes pulse {
        0% { box-shadow: 0 0 0 0 rgba(239, 68, 68, 0.4); }
        70% { box-shadow: 0 0 0 15px rgba(239, 68, 68, 0); }
        100% { box-shadow: 0 0 0 0 rgba(239, 68, 68, 0); }
    }
    
    .recording-active {
        animation: pulse 2s infinite;
    }
</style>
""", unsafe_allow_html=True)

# Modern header with gradient text
st.markdown("""
<div style="text-align: center; padding: 1rem 0 1.5rem 0;">
    <h1 style="
        font-size: 3rem; 
        font-weight: 800; 
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        margin: 0;
    ">🎓 EchoLearn</h1>
    <p style="color: #64748b; font-size: 1.1rem; margin: 0.5rem 0 0 0;">
        AI-Powered Viva Practice Platform
    </p>
</div>
""", unsafe_allow_html=True)

# ------------------ Session State Initialization ------------------
def initialize_session_state():
    """Initialize all session state variables"""
    default_states = {
        'pdf_text_dict': {},
        'qa_dict': {},
        'all_qas': [],
        'qa_index': 0,
        'used_q_indices': [],
        'current_conversation_id': None,
        'resume_session': False,
        'question_mode': "PDF Upload",
        'current_predefined_session_id': None,
        'resume_predefined_session': False,
        'adaptive_mode': True,
        'current_difficulty': 10,
        'last_answer_correct': None,
        'consecutive_wrong_same_level': 0,
        'difficulty_path': [],
        'session_complete': False,
        'selective_mutism_mode': False,
        'confidence_level': 1,
        'success_streak': 0,
        'sm_progress_milestones': []
    }
    
    for key, default_value in default_states.items():
        if key not in st.session_state:
            st.session_state[key] = default_value

initialize_session_state()

# ------------------ Resume Session Logic ------------------
def handle_resume_sessions():
    """Handle resuming PDF or predefined question sessions"""
    if st.session_state.resume_session and st.session_state.current_conversation_id:
        # Load PDF-based conversation data
        conversations = db_manager.get_user_conversations(current_user['id'])
        current_conv = next((c for c in conversations if c['id'] == st.session_state.current_conversation_id), None)
        
        if current_conv:
            st.info(f"🔄 Resuming PDF session: {current_conv['subject']} - {current_conv['book_title']}")
            
            # Load conversation details
            name = current_conv['name']
            grade = current_conv['grade']
            subject = current_conv['subject']
            book_title = current_conv['book_title']
            
            # Load questions and answers
            questions = db_manager.get_conversation_questions(st.session_state.current_conversation_id)
            st.session_state.all_qas = questions
            
            # Set current question index to first unanswered question
            answered_indices = [i for i, q in enumerate(questions) if q['score'] is not None]
            st.session_state.used_q_indices = answered_indices
            
            # Find next unanswered question
            next_unanswered = next((i for i, q in enumerate(questions) if q['score'] is None), 0)
            st.session_state.qa_index = next_unanswered
            
            st.session_state.resume_session = False
            st.session_state.question_mode = "PDF Upload"
            
            return name, grade, subject, book_title
    
    elif st.session_state.resume_predefined_session and st.session_state.current_predefined_session_id:
        # Load predefined question session data
        session_info, questions = db_manager.get_predefined_session_questions(st.session_state.current_predefined_session_id)
        
        if session_info:
            st.info(f"🔄 Resuming predefined session: {session_info['subject_name']} - {session_info.get('topic_name', 'All Topics')}")
            
            # Load session details
            name = session_info['name']
            grade = session_info['grade']
            subject = session_info['subject_name']
            book_title = f"Predefined Questions - {session_info['subject_name']}"
            
            # Convert predefined questions to the format expected by the UI
            st.session_state.all_qas = questions
            
            # Set current question index to first unanswered question
            answered_indices = [i for i, q in enumerate(questions) if q['score'] is not None]
            st.session_state.used_q_indices = answered_indices
            
            # Find next unanswered question
            next_unanswered = next((i for i, q in enumerate(questions) if q['score'] is None), 0)
            st.session_state.qa_index = next_unanswered
            
            st.session_state.resume_predefined_session = False
            st.session_state.question_mode = "Predefined Questions"
            
            return name, grade, subject, book_title
    
    return None, None, None, None

# ------------------ Main Application Logic ------------------
def main():
    """Main application logic"""
    # Get current user within the function
    current_user = auth_manager.get_current_user()
    
    # Handle resume sessions
    name, grade, subject, book_title = handle_resume_sessions()
    
    if name is None:
        # ------------------ Show User Dashboard ------------------
        auth_manager.show_user_dashboard()
        
        # ------------------ Question Mode Selection ------------------
        st.markdown("### 🎯 Choose Your Learning Mode")
        
        # Mode descriptions for better UX
        mode_info = {
            "📄 PDF Upload": ("Upload & Learn", "Upload your PDFs and get AI-generated questions"),
            "📋 Question Bank": ("Practice Questions", "Practice with our curated question bank"),
            "🎤 Mock Viva": ("Interview Prep", "Practice with AI interviewers"),
            "🎙️ Viva Voice": ("Voice Mentor", "Interactive voice-driven viva demo"),
            "📊 Analytics": ("Your Progress", "View detailed performance insights"),
            "🎧 Audio Lab": ("Voice Training", "Improve your speaking skills")
        }
        
        mode_mapping = {
            "📄 PDF Upload": "PDF Upload",
            "📋 Question Bank": "Predefined Questions", 
            "🎤 Mock Viva": "🎤 Mock Viva Interview",
            "🎙️ Viva Voice": "🎙️ Viva Voice Mentor",
            "📊 Analytics": "📊 Analytics Dashboard",
            "🎧 Audio Lab": "Audio Training Lab"
        }
        
        # Find current mode key
        current_mode_key = next((k for k, v in mode_mapping.items() if v == st.session_state.question_mode), "📄 PDF Upload")
        
        # Create mode selection with radio buttons
        selected_mode = st.radio(
            "Select Mode:",
            list(mode_info.keys()),
            index=list(mode_info.keys()).index(current_mode_key),
            horizontal=True,
            label_visibility="collapsed"
        )
        
        # Show selected mode description
        title, desc = mode_info[selected_mode]
        st.caption(f"**{title}** — {desc}")
        
        st.session_state.question_mode = mode_mapping[selected_mode]
        question_mode = st.session_state.question_mode
        
        st.markdown("---")
        
        # ------------------ Input Fields ------------------
        # Handle case where current_user might be None
        default_name = ""
        if current_user:
            default_name = current_user.get('full_name', current_user.get('username', ''))
        name = st.text_input("Name : ", value=default_name)
        
        if question_mode == "PDF Upload":
            grade = st.text_input("Grade : ")
            subject = st.text_input("Subject : ")
            book_title = st.text_input("Book Title : ")
        elif question_mode == "Audio Training Lab":
            # Audio Lab doesn't need grade/subject input
            grade = "N/A"
            subject = "Audio Training"
            book_title = "Audio Training Lab"
        elif question_mode == "🎤 Mock Viva Interview":
            # Mock Viva Interview mode
            grade = "N/A"
            subject = "Mock Viva"
            book_title = "Mock Viva Interview"
        elif question_mode == "🎙️ Viva Voice Mentor":
            grade = "N/A"
            subject = "Viva Voice Mentor"
            book_title = "Viva Voice Mentor"
        elif question_mode == "📊 Analytics Dashboard":
            # Analytics Dashboard mode - no input needed
            grade = "N/A"
            subject = "Analytics"
            book_title = "Analytics Dashboard"
        else:
            # Predefined Questions Mode - Initialize all variables first
            subject_id = None
            topic_id = None
            difficulty_min = 1.0
            difficulty_max = 100.0
            subject = ""
            grade = ""
            book_title = ""
            
            subjects = db_manager.get_subjects()
            
            if subjects:
                selected_subject = st.selectbox(
                    "Subject:",
                    options=[s['name'] for s in subjects],
                    help="Select the subject for your practice session"
                )
                
                subject_id = next((s['id'] for s in subjects if s['name'] == selected_subject), None)
                
                if subject_id:
                    # Get available grades for this subject
                    grades = db_manager.get_grades_by_subject(subject_id)
                    if grades:
                        grade = st.selectbox("Grade:", grades)
                    else:
                        grade = st.text_input("Grade:", value="11")
                    
                    # Get topics for this subject
                    topics = db_manager.get_topics_by_subject(subject_id)
                    topic_options = ["All Topics"] + [t['name'] for t in topics]
                    selected_topic = st.selectbox("Topic:", topic_options)
                    
                    topic_id = None
                    if selected_topic != "All Topics":
                        topic_id = next((t['id'] for t in topics if t['name'] == selected_topic), None)
                    
                    # Difficulty range
                    col1, col2 = st.columns(2)
                    with col1:
                        difficulty_min = st.slider("Minimum Difficulty:", 1.0, 100.0, 1.0, 1.0)
                    with col2:
                        difficulty_max = st.slider("Maximum Difficulty:", 1.0, 100.0, 100.0, 1.0)
                    
                    # Preview available questions
                    preview_questions = db_manager.get_predefined_questions(
                        subject_id=subject_id,
                        topic_id=topic_id,
                        grade=grade,
                        difficulty_min=difficulty_min,
                        difficulty_max=difficulty_max
                    )
                    
                    st.info(f"📊 {len(preview_questions)} questions available with your current filters")
                    
                    subject = selected_subject
                    book_title = f"Predefined Questions - {selected_subject}"
            else:
                st.error("No subjects found in the question bank. Please contact administrator.")
    
    # ------------------ PDF Upload (only for PDF mode) ------------------
    if st.session_state.question_mode == "PDF Upload":
        handle_pdf_upload(name, grade, subject, book_title, current_user)
    
    # ------------------ Predefined Questions Mode ------------------
    elif st.session_state.question_mode == "Predefined Questions":
        # Pass all required variables to the function
        predefined_vars = {
            'subject_id': locals().get('subject_id'),
            'topic_id': locals().get('topic_id'),
            'difficulty_min': locals().get('difficulty_min', 1.0),
            'difficulty_max': locals().get('difficulty_max', 100.0),
            'current_user': current_user
        }
        handle_predefined_questions(name, grade, subject, book_title, predefined_vars)
    
    # ------------------ Mock Viva Interview Mode (NEW) ------------------
    elif st.session_state.question_mode == "🎤 Mock Viva Interview":
        handle_mock_viva_interview(current_user)

    # ------------------ Viva Voice Mentor Mode (NEW) ------------------
    elif st.session_state.question_mode == "🎙️ Viva Voice Mentor":
        handle_viva_voice_mentor()
    
    # ------------------ Analytics Dashboard Mode (NEW) ------------------
    elif st.session_state.question_mode == "📊 Analytics Dashboard":
        handle_analytics_dashboard(current_user)
    
    # ------------------ Audio Training Lab Mode ------------------
    elif st.session_state.question_mode == "Audio Training Lab":
        handle_audio_training_lab()
    
    # ------------------ Viva Questions Interface ------------------
    # Only render the viva Q&A interface for modes that actually use it
    viva_modes = ("PDF Upload", "Predefined Questions")
    if st.session_state.all_qas and st.session_state.question_mode in viva_modes:
        handle_viva_interface(name, grade, subject, book_title)

def handle_pdf_upload(name, grade, subject, book_title, current_user):
    """Handle PDF upload and question generation"""
    st.markdown("""
    <div style="background: linear-gradient(135deg, #faf5ff 0%, #f3e8ff 100%); 
                border-radius: 16px; padding: 1.5rem; margin-bottom: 1rem;
                border-left: 4px solid #a855f7;">
        <h2 style="color: #7c3aed; margin: 0;">📄 Upload Your PDF</h2>
        <p style="color: #64748b; margin: 0.5rem 0 0 0;">Upload a study material and get AI-generated questions</p>
    </div>
    """, unsafe_allow_html=True)
    
    book_pdf_file = st.file_uploader("Choose a PDF file", type="pdf", label_visibility="collapsed")

    if book_pdf_file is not None:
        doc = fitz.open(stream=book_pdf_file.read(), filetype="pdf")
        st.session_state.pdf_text_dict.clear()

        for i, page in enumerate(doc):
            text = page.get_text().strip()
            if text:
                st.session_state.pdf_text_dict[i + 1] = text

        st.success("✅ PDF uploaded and text extracted.")
        
        # Create new conversation in database
        if not st.session_state.current_conversation_id and name and grade and subject and book_title:
            try:
                pdf_content = "\n\n".join(st.session_state.pdf_text_dict.values())
                conversation_id = db_manager.create_conversation(
                    user_id=current_user['id'],
                    name=name,
                    grade=grade,
                    subject=subject,
                    book_title=book_title,
                    pdf_content=pdf_content
                )
                st.session_state.current_conversation_id = conversation_id
                st.success(f"📚 Study session created and saved!")
            except Exception as e:
                st.error(f"Error creating study session: {str(e)}")

    # ------------------ Page Viewer ------------------
    if st.session_state.pdf_text_dict:
        selected_page = st.selectbox("View a Page:", list(st.session_state.pdf_text_dict.keys()))
        st.text_area("Extracted Text", st.session_state.pdf_text_dict[selected_page], height=300)

    # ------------------ Question Generation ------------------
    if st.button("🔍 Generate Viva Questions"):
        if st.session_state.pdf_text_dict:
            full_text = "\n\n".join(st.session_state.pdf_text_dict.values())
            
            # Use the new question manager
            questions, validation_errors = question_manager.generate_and_validate_questions(full_text, 20)
            
            if validation_errors:
                st.warning("Some questions had validation issues:")
                for error in validation_errors:
                    st.write(f"• {error}")
            
            if questions:
                st.session_state.all_qas = questions
                st.session_state.qa_index = 0
                st.session_state.used_q_indices = []
                
                # Save questions to database
                if st.session_state.current_conversation_id:
                    success = db_manager.save_questions(st.session_state.current_conversation_id, questions)
                    if success:
                        st.success("✅ Viva questions generated and saved to database.")
                    else:
                        st.warning("✅ Viva questions generated but couldn't save to database.")
                else:
                    st.success("✅ Viva questions generated.")
            else:
                st.error("Failed to generate valid questions. Please try again.")

def handle_predefined_questions(name, grade, subject, book_title, predefined_vars):
    """Handle predefined questions mode"""
    st.markdown("""
    <div style="background: linear-gradient(135deg, #f0f9ff 0%, #e0f2fe 100%); 
                border-radius: 16px; padding: 1.5rem; margin-bottom: 1rem;
                border-left: 4px solid #0ea5e9;">
        <h2 style="color: #0369a1; margin: 0;">📋 Question Bank</h2>
        <p style="color: #64748b; margin: 0.5rem 0 0 0;">Practice with curated questions from our database</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Extract variables from the passed dictionary
    subject_id = predefined_vars.get('subject_id')
    topic_id = predefined_vars.get('topic_id')
    difficulty_min = predefined_vars.get('difficulty_min', 1.0)
    difficulty_max = predefined_vars.get('difficulty_max', 100.0)
    current_user = predefined_vars.get('current_user')
    
    # Start predefined question session button
    st.markdown("")  # Add spacing
    if st.button("🚀 Start Question Session", type="primary", use_container_width=True):
        # Detailed validation with specific error messages
        validation_errors = []
        
        if not name or name.strip() == "":
            validation_errors.append("Name is required")
        if not grade or grade.strip() == "":
            validation_errors.append("Grade is required")
        if not subject or subject.strip() == "":
            validation_errors.append("Subject is required")
        if not subject_id:
            validation_errors.append("Please select a valid subject")
        if not current_user:
            validation_errors.append("User authentication required")
        
        if validation_errors:
            st.error("❌ **Please fix the following issues:**")
            for error in validation_errors:
                st.error(f"   • {error}")
        else:
            try:
                session_id = db_manager.create_predefined_question_session(
                    user_id=current_user['id'],
                    name=name,
                    grade=grade,
                    subject_id=subject_id,
                    topic_id=topic_id,
                    difficulty_min=difficulty_min,
                    difficulty_max=difficulty_max
                )
                
                st.session_state.current_predefined_session_id = session_id
                
                # Load questions for the session
                session_info, questions = db_manager.get_predefined_session_questions(session_id)
                st.session_state.all_qas = questions
                st.session_state.qa_index = 0
                st.session_state.used_q_indices = []
                
                st.success(f"📚 Question session started with {len(questions)} questions!")
                st.rerun()
                
            except Exception as e:
                st.error(f"Error creating question session: {str(e)}")

def handle_audio_training_lab():
    """Handle the Audio Training Lab interface"""
    audio_lab.display_audio_lab_interface()


def handle_viva_voice_mentor():
    """Handle the Viva Voice Mentor demo interface"""
    display_viva_voice_agent()


# =============================================
# NEW: Mock Viva Interview Handler
# =============================================
def handle_mock_viva_interview(current_user):
    """Handle the Mock Viva Interview interface"""
    st.markdown("""
    <div style="background: linear-gradient(135deg, #fef3c7 0%, #fde68a 100%); 
                border-radius: 16px; padding: 1.5rem; margin-bottom: 1rem;
                border-left: 4px solid #f59e0b;">
        <h2 style="color: #b45309; margin: 0;">🎤 Mock Viva Interview</h2>
        <p style="color: #64748b; margin: 0.5rem 0 0 0;">Practice with AI-powered interviewers who adapt to your performance</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Initialize session state for mock viva
    if 'mock_viva_session' not in st.session_state:
        st.session_state.mock_viva_session = None
    if 'mock_viva_current_question' not in st.session_state:
        st.session_state.mock_viva_current_question = None
    if 'mock_viva_waiting_for_answer' not in st.session_state:
        st.session_state.mock_viva_waiting_for_answer = False
    
    # If no active session, show setup
    if st.session_state.mock_viva_session is None:
        st.markdown("---")
        st.markdown("### 👤 Choose Your Interviewer")
        
        # Display interviewer options
        cols = st.columns(len(INTERVIEWER_PROFILES))
        selected_persona = None
        
        for i, (persona, profile) in enumerate(INTERVIEWER_PROFILES.items()):
            with cols[i]:
                st.markdown(f"#### {profile.emoji}")
                st.markdown(f"**{profile.name}**")
                st.caption(profile.title)
                
                with st.expander("Details"):
                    st.write(f"*{profile.style}*")
                    for char in profile.characteristics:
                        st.write(f"• {char}")
                
                if st.button(f"Select", key=f"select_{persona.value}"):
                    selected_persona = persona
        
        if selected_persona:
            st.session_state.selected_interviewer = selected_persona
        
        # Topic and question source selection
        st.markdown("---")
        st.markdown("### 📚 Select Questions")
        
        question_source = st.radio(
            "Question Source:",
            ["Use Predefined Questions", "Use PDF-Generated Questions"],
            help="Choose where to get questions from"
        )
        
        questions_for_viva = []
        topic_name = "General"
        
        if question_source == "Use Predefined Questions":
            subjects = db_manager.get_subjects()
            if subjects:
                selected_subject = st.selectbox("Subject:", [s['name'] for s in subjects])
                subject_id = next((s['id'] for s in subjects if s['name'] == selected_subject), None)
                
                if subject_id:
                    topics = db_manager.get_topics_by_subject(subject_id)
                    topic_options = ["All Topics"] + [t['name'] for t in topics]
                    selected_topic = st.selectbox("Topic:", topic_options)
                    
                    topic_id = None
                    if selected_topic != "All Topics":
                        topic_id = next((t['id'] for t in topics if t['name'] == selected_topic), None)
                        topic_name = selected_topic
                    else:
                        topic_name = selected_subject
                    
                    num_questions = st.slider("Number of Questions:", 3, 15, 5)
                    
                    questions_for_viva = db_manager.get_predefined_questions(
                        subject_id=subject_id,
                        topic_id=topic_id,
                        limit=num_questions
                    )
                    
                    st.info(f"📊 {len(questions_for_viva)} questions ready for your mock viva!")
        else:
            if st.session_state.all_qas:
                questions_for_viva = st.session_state.all_qas[:10]
                topic_name = "PDF Content"
                st.success(f"✅ Using {len(questions_for_viva)} questions from your uploaded PDF")
            else:
                st.warning("⚠️ No PDF questions available. Please upload a PDF first in 'PDF Upload' mode.")
        
        # Start session button
        st.markdown("---")
        if st.button("🚀 Start Mock Viva Interview", type="primary", disabled=len(questions_for_viva) == 0):
            if 'selected_interviewer' in st.session_state:
                persona = st.session_state.selected_interviewer
                profile = INTERVIEWER_PROFILES[persona]
                
                # Create session
                session = mock_viva_interviewer.start_session(
                    persona=persona,
                    topic=topic_name,
                    questions=questions_for_viva
                )
                
                st.session_state.mock_viva_session = session
                
                # Save to database
                db_manager.create_mock_viva_session(
                    user_id=current_user['id'],
                    interviewer_persona=persona.value,
                    interviewer_name=profile.name,
                    topic=topic_name,
                    total_questions=len(questions_for_viva)
                )
                
                st.rerun()
            else:
                st.error("Please select an interviewer first!")
    
    else:
        # Active session - show interview interface
        session = st.session_state.mock_viva_session
        profile = session.profile
        
        # Session header
        col1, col2, col3 = st.columns([2, 1, 1])
        with col1:
            st.markdown(f"### {profile.emoji} Interview with {profile.name}")
            st.caption(profile.title)
        with col2:
            progress = (session.current_question_index + 1) / len(session.questions)
            st.progress(progress)
            st.caption(f"Q{session.current_question_index + 1}/{len(session.questions)}")
        with col3:
            st.metric("Score", f"{session.total_score}/{session.max_score}")
        
        st.markdown("---")
        
        # Conversation display
        st.markdown("#### 💬 Conversation")
        for turn in session.conversation_history[-6:]:  # Show last 6 turns
            if turn.speaker == 'interviewer':
                st.markdown(f"**{profile.emoji} {profile.name}:** {turn.message}")
            else:
                st.markdown(f"**🎓 You:** {turn.message}")
        
        st.markdown("---")
        
        # If waiting for answer
        if not st.session_state.mock_viva_waiting_for_answer:
            # Ask next question
            if session.current_question_index < len(session.questions):
                question_text = mock_viva_interviewer.ask_question()
                st.session_state.mock_viva_current_question = question_text
                st.session_state.mock_viva_waiting_for_answer = True
                st.rerun()
        else:
            # Show current question and get answer
            st.markdown(f"**📝 Current Question:** {st.session_state.mock_viva_current_question}")
            
            # Timer for quick-fire mode
            if profile.time_pressure:
                st.warning("⏱️ Quick-fire mode: Answer within 30 seconds!")
            
            # Answer input
            col1, col2 = st.columns([3, 1])
            with col1:
                user_answer = st.text_area("Your Answer:", height=100, key="mock_viva_answer")
            with col2:
                if profile.encouragement_level >= 0.5:
                    if st.button("💡 Get Hint"):
                        hint = mock_viva_interviewer.get_hint()
                        if hint:
                            st.info(f"💡 {hint}")
            
            col1, col2, col3 = st.columns(3)
            with col1:
                if st.button("📤 Submit Answer", type="primary"):
                    if user_answer.strip():
                        # Evaluate answer
                        result = mock_viva_interviewer.evaluate_answer(user_answer)
                        
                        # Display feedback
                        evaluation = result['evaluation']
                        st.markdown(f"**Score:** {evaluation['score']}/10")
                        st.markdown(f"**Feedback:** {evaluation['feedback']}")
                        
                        # Voice analysis on the answer text
                        voice_result = voice_analyzer.analyze_speech(user_answer, audio_duration_seconds=30)
                        if voice_result.filler_word_count > 0:
                            st.caption(f"💬 Filler words detected: {voice_result.filler_word_count}")
                        
                        # Handle follow-up
                        if result.get('should_follow_up') and result.get('follow_up'):
                            st.info(f"**{profile.emoji} Follow-up:** {result['follow_up']}")
                        
                        # Move to next question
                        if mock_viva_interviewer.next_question():
                            st.session_state.mock_viva_waiting_for_answer = False
                            st.rerun()
                        else:
                            # Session complete
                            summary = mock_viva_interviewer.end_session()
                            st.session_state.mock_viva_summary = summary
                            st.session_state.mock_viva_session = None
                            st.session_state.mock_viva_waiting_for_answer = False
                            st.rerun()
                    else:
                        st.warning("Please provide an answer!")
            
            with col2:
                if st.button("⏭️ Skip Question"):
                    if mock_viva_interviewer.next_question():
                        st.session_state.mock_viva_waiting_for_answer = False
                        st.rerun()
            
            with col3:
                if st.button("🛑 End Interview"):
                    summary = mock_viva_interviewer.end_session()
                    st.session_state.mock_viva_summary = summary
                    st.session_state.mock_viva_session = None
                    st.session_state.mock_viva_waiting_for_answer = False
                    st.rerun()
    
    # Show summary if available
    if 'mock_viva_summary' in st.session_state and st.session_state.mock_viva_summary:
        summary = st.session_state.mock_viva_summary
        st.markdown("---")
        st.markdown("## 🎉 Interview Complete!")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Final Score", f"{summary['total_score']}/{summary['max_score']}")
        with col2:
            st.metric("Average", f"{summary['average_score']:.1f}/10")
        with col3:
            st.metric("Duration", summary['duration_formatted'])
        
        st.markdown(f"### {summary['performance_emoji']} {summary['performance_rating']}")
        
        if st.button("🔄 Start New Interview"):
            st.session_state.mock_viva_summary = None
            st.rerun()


# =============================================
# NEW: Analytics Dashboard Handler
# =============================================
def handle_analytics_dashboard(current_user):
    """Handle the Smart Analytics Dashboard"""
    st.markdown("""
    <div style="background: linear-gradient(135deg, #ecfdf5 0%, #d1fae5 100%); 
                border-radius: 16px; padding: 1.5rem; margin-bottom: 1rem;
                border-left: 4px solid #10b981;">
        <h2 style="color: #047857; margin: 0;">📊 Analytics Dashboard</h2>
        <p style="color: #64748b; margin: 0.5rem 0 0 0;">Track your progress and identify areas for improvement</p>
    </div>
    """, unsafe_allow_html=True)
    
    if not current_user:
        st.error("Please log in to view your analytics.")
        return
    
    # Get user data
    performance_data = db_manager.get_user_performance_data(current_user['id'])
    activity_data = db_manager.get_user_activity_data(current_user['id'])
    
    if not performance_data:
        st.info("📝 No data yet! Complete some questions to see your analytics.")
        st.markdown("""
        ### Get Started:
        1. 📚 Upload a PDF and answer questions
        2. 📋 Practice with predefined questions
        3. 🎤 Try a mock viva interview
        
        Your analytics will appear here once you have some practice data!
        """)
        return
    
    # Display the full analytics dashboard
    analytics_dashboard.display_full_dashboard(
        user_id=current_user['id'],
        performance_data=performance_data,
        session_data=activity_data
    )
    
    # Additional voice analytics section
    st.markdown("---")
    st.markdown("### 🎙️ Voice Analysis History")
    
    voice_data = db_manager.get_user_voice_analytics(current_user['id'])
    
    if voice_data:
        import pandas as pd
        import plotly.express as px
        
        df = pd.DataFrame(voice_data)
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Confidence score over time
            fig = px.line(df, y='confidence_score', title='Confidence Score Progress',
                         labels={'confidence_score': 'Confidence', 'index': 'Session'})
            fig.update_layout(height=300)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Filler word percentage over time
            fig = px.line(df, y='filler_word_percentage', title='Filler Word % Progress',
                         labels={'filler_word_percentage': 'Filler %', 'index': 'Session'})
            fig.update_layout(height=300)
            st.plotly_chart(fig, use_container_width=True)
        
        # Speaking pace
        avg_wpm = df['words_per_minute'].mean()
        st.metric("Average Speaking Pace", f"{avg_wpm:.0f} WPM", 
                  delta="Optimal" if 120 <= avg_wpm <= 160 else "Adjust pace")
    else:
        st.info("🎙️ Complete some audio recordings to see your voice analysis progress!")
    
    # Mock Viva History
    st.markdown("---")
    st.markdown("### 🎤 Mock Viva History")
    
    viva_history = db_manager.get_user_mock_viva_history(current_user['id'])
    
    if viva_history:
        for session in viva_history[:5]:
            with st.expander(f"📅 {session['started_at']} - {session['interviewer_name']} ({session['topic']})"):
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Score", f"{session['total_score']}/{session['max_possible_score']}")
                with col2:
                    st.metric("Questions", f"{session['questions_answered']}/{session['total_questions']}")
                with col3:
                    st.metric("Performance", session['performance_rating'] or "N/A")
    else:
        st.info("🎤 Complete a mock viva interview to see your history here!")


def handle_viva_interface(name, grade, subject, book_title):
    """Handle the main viva questions interface"""
    current_user = auth_manager.get_current_user()

    st.subheader("🧠 Viva Questions")
    
    # Display mode toggles
    mode_settings = UIComponents.display_mode_toggles()
    adaptive_mode = mode_settings['adaptive_mode']
    selective_mutism_mode = mode_settings['selective_mutism_mode']
    
    current = st.session_state.qa_index
    qa = st.session_state.all_qas[current]
    total_questions = len(st.session_state.all_qas)
    
    # Display question navigation
    UIComponents.display_question_navigation(current, total_questions, qa)
    
    # Display question info
    UIComponents.display_question_info(qa, adaptive_mode)
    
    # Display TTS button
    UIComponents.display_tts_button(qa)
    
    # Handle audio recording
    transcribed_text = UIComponents.display_audio_recording_interface(qa, current, selective_mutism_mode)
    
    if transcribed_text:
        handle_audio_answer(qa, current, transcribed_text, selective_mutism_mode, adaptive_mode, subject, current_user)
    
    # Handle text input
    if not selective_mutism_mode:
        handle_text_input(qa, current, adaptive_mode, subject, current_user)
    else:
        handle_selective_mutism_text_input(qa, current, subject, current_user)
    
    # Display session statistics
    UIComponents.display_session_statistics(st.session_state.all_qas)
    
    # Display report download
    user_info = {'name': name, 'subject': subject, 'book_title': book_title}
    UIComponents.display_report_download(st.session_state.all_qas, user_info)
    
    # Add option to start new session
    if st.button("🆕 Start New Session"):
        clear_session_state()
        st.rerun()

def handle_audio_answer(qa, current, transcribed_text, selective_mutism_mode, adaptive_mode, subject, current_user):
    """Handle audio answer processing"""
    st.session_state.all_qas[current]["user_answer"] = transcribed_text
    
    # NEW: Perform voice analysis on the transcribed text
    voice_result = voice_analyzer.analyze_speech(
        transcript=transcribed_text,
        audio_duration_seconds=st.session_state.get('last_recording_duration', 10)
    )
    
    # Evaluate answer using appropriate method
    if selective_mutism_mode:
        evaluation = scoring_evaluator.evaluate_answer_selective_mutism(
            qa["question"], qa["answer"], transcribed_text, st.session_state.confidence_level
        )
        score = evaluation['score']
        
        # Update confidence and show encouragement
        confidence_update = selective_mutism_support.update_confidence_level(
            score >= 6, 'speech'
        )
        
        UIComponents.display_evaluation_result(evaluation, 'selective_mutism')
        
        # Special celebration for speech training
        if score >= 6:
            st.balloons()
            st.success("🎙️ **You did it! You spoke up and that's incredible!** Your voice matters!")
        else:
            st.info("🎙️ **You were so brave to speak! Every time you practice, you get stronger!**")
        
        # Update session state
        st.session_state.confidence_level = selective_mutism_support.state.confidence_level
        st.session_state.success_streak = selective_mutism_support.state.success_streak
        st.session_state.sm_progress_milestones = selective_mutism_support.state.progress_milestones
        
    else:
        evaluation = scoring_evaluator.evaluate_answer_standard(
            qa["question"], qa["answer"], transcribed_text
        )
        score = evaluation['score']
        UIComponents.display_evaluation_result(evaluation, 'standard')
    
    st.session_state.all_qas[current]["score"] = score
    
    # NEW: Display voice analysis results
    with st.expander("🎙️ Voice Analysis Feedback", expanded=True):
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Speaking Pace", f"{voice_result.words_per_minute:.0f} WPM", voice_result.pace_rating)
        with col2:
            st.metric("Filler Words", f"{voice_result.filler_word_count}", f"{voice_result.filler_word_percentage:.1f}%")
        with col3:
            st.metric("Confidence", f"{voice_result.confidence_score:.0f}/100", voice_result.confidence_rating)
        
        # Show improvement tips
        if voice_result.areas_for_improvement:
            st.warning("**Tips for Improvement:**")
            for tip in voice_result.areas_for_improvement:
                st.write(f"• {tip}")
        
        if voice_result.strengths:
            st.success("**Strengths:** " + " • ".join(voice_result.strengths))
    
    # NEW: Save voice analysis to database
    try:
        db_manager.save_voice_analysis(
            user_id=current_user['id'],
            transcript=transcribed_text,
            analysis_result={
                'filler_word_count': voice_result.filler_word_count,
                'filler_words_found': voice_result.filler_words_found,
                'filler_word_percentage': voice_result.filler_word_percentage,
                'words_per_minute': voice_result.words_per_minute,
                'total_words': voice_result.total_words,
                'speaking_duration_seconds': voice_result.speaking_duration_seconds,
                'confidence_score': voice_result.confidence_score,
                'overall_score': voice_result.overall_score,
                'strengths': voice_result.strengths,
                'areas_for_improvement': voice_result.areas_for_improvement
            }
        )
    except Exception as e:
        pass  # Silently fail if voice analysis save fails
    
    # Save to database
    save_answer_to_database(current, transcribed_text, score, 'audio', subject, current_user)
    
    # NEW: Log study activity
    try:
        db_manager.log_study_activity(
            user_id=current_user['id'],
            activity_type='question_answered',
            subject=subject,
            questions_attempted=1,
            questions_correct=1 if score >= 6 else 0,
            total_score=score,
            max_score=10
        )
    except Exception as e:
        pass  # Silently fail
    
    # Add to used indices
    if current not in st.session_state.used_q_indices:
        st.session_state.used_q_indices.append(current)
    
    # Handle next question logic
    handle_next_question_logic(score, adaptive_mode, selective_mutism_mode, current, qa)

def handle_text_input(qa, current, adaptive_mode, subject, current_user):
    """Handle regular text input"""
    manual_answer = UIComponents.display_text_input(qa, current, False)
    
    if UIComponents.display_submit_button("standard"):
        if manual_answer.strip():
            st.session_state.all_qas[current]["user_answer"] = manual_answer
            
            # Evaluate answer
            evaluation = scoring_evaluator.evaluate_answer_standard(
                qa["question"], qa["answer"], manual_answer
            )
            score = evaluation['score']
            st.session_state.all_qas[current]["score"] = score
            
            # Display result
            UIComponents.display_evaluation_result(evaluation, 'standard')
            
            # Save to database
            save_answer_to_database(current, manual_answer, score, 'text', subject, current_user)
            
            # Add to used indices
            if current not in st.session_state.used_q_indices:
                st.session_state.used_q_indices.append(current)
            
            # Handle next question logic
            handle_next_question_logic(score, adaptive_mode, False, current, qa)
        else:
            st.warning("Please provide an answer.")

def handle_selective_mutism_text_input(qa, current, subject, current_user):
    """Handle selective mutism text input"""
    backup_answer = UIComponents.display_text_input(qa, current, True)
    
    if UIComponents.display_submit_button("selective_mutism_text"):
        if backup_answer.strip():
            # Evaluate answer
            evaluation = scoring_evaluator.evaluate_answer_selective_mutism(
                qa["question"], qa["answer"], backup_answer, st.session_state.confidence_level
            )
            score = evaluation['score']
            
            st.session_state.all_qas[current]["user_answer"] = backup_answer
            st.session_state.all_qas[current]["score"] = score
            
            # Update confidence and show encouragement
            confidence_update = selective_mutism_support.update_confidence_level(
                score >= 6, 'text'
            )
            
            UIComponents.display_evaluation_result(evaluation, 'selective_mutism')
            st.info("💪 **Great job expressing yourself in writing! You're building communication skills!**")
            
            # Update session state
            st.session_state.confidence_level = selective_mutism_support.state.confidence_level
            st.session_state.success_streak = selective_mutism_support.state.success_streak
            st.session_state.sm_progress_milestones = selective_mutism_support.state.progress_milestones
            
            # Save to database
            save_answer_to_database(current, backup_answer, score, 'selective_mutism_text', subject, current_user)
            
            # Add to used indices
            if current not in st.session_state.used_q_indices:
                st.session_state.used_q_indices.append(current)
            
            # Handle next question logic
            handle_next_question_logic(score, False, True, current, qa)
        else:
            st.warning("💖 Please write something! Even a few words show you're trying.")

def handle_next_question_logic(score, adaptive_mode, selective_mutism_mode, current, qa):
    """Handle logic for moving to next question"""
    time.sleep(1)  # Short delay to allow user to see the message
    
    # Check if session is complete
    if len(st.session_state.used_q_indices) >= len(st.session_state.all_qas):
        mark_session_complete()
        st.session_state.session_complete = True
        UIComponents.display_final_score_report(st.session_state.all_qas)
        st.success(f"🎉 All questions completed! Total Score: {sum(q.get('score', 0) for q in st.session_state.all_qas if q.get('score') is not None)}/{len(st.session_state.all_qas) * 10}")
    else:
        if adaptive_mode and not selective_mutism_mode:
            # Run adaptive selection
            current_index_before_adaptive = st.session_state.qa_index
            
            # Update adaptive engine
            question_difficulty = qa.get('difficulty', 10)
            recommendations = adaptive_engine.update_state(score, current, question_difficulty)
            
            # Find next question
            next_question_index = adaptive_engine.find_next_question(
                st.session_state.all_qas, st.session_state.used_q_indices
            )
            
            if next_question_index is not None:
                st.session_state.qa_index = next_question_index
                st.session_state.current_difficulty = recommendations['target_difficulty']
                st.info(f"🎯 Adaptive system selected question {st.session_state.qa_index + 1} (Difficulty: {st.session_state.current_difficulty})")
                st.rerun()
            else:
                st.warning("⚠️ No more suitable questions found at current difficulty level.")
        else:
            # Manual mode or selective mutism mode - just proceed to next unanswered question
            next_unanswered = next((i for i, q in enumerate(st.session_state.all_qas) 
                                  if i not in st.session_state.used_q_indices), None)
            if next_unanswered is not None:
                st.session_state.qa_index = next_unanswered
                st.rerun()

def save_answer_to_database(current, answer_text, score, method, subject, current_user):
    """Save answer to database"""
    try:
        if st.session_state.current_conversation_id:
            # PDF-generated questions
            questions = db_manager.get_conversation_questions(st.session_state.current_conversation_id)
            if current < len(questions):
                question_id = questions[current]['id']
                db_manager.save_user_answer(question_id, answer_text, score, answer_method=method)
                if current_user:
                    db_manager.update_user_progress(current_user['id'], subject)
        elif st.session_state.current_predefined_session_id:
            # Predefined questions
            qa = st.session_state.all_qas[current]
            question_id = qa.get('id')
            if question_id:
                db_manager.save_predefined_question_answer(
                    st.session_state.current_predefined_session_id,
                    question_id,
                    answer_text,
                    score,
                    answer_method=method
                )
                if current_user:
                    db_manager.update_user_progress(current_user['id'], subject)
    except Exception as e:
        st.error(f"Error saving answer: {str(e)}")

def mark_session_complete():
    """Mark session as completed in database"""
    try:
        import sqlite3
        if st.session_state.current_conversation_id:
            with sqlite3.connect(db_manager.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    UPDATE conversations 
                    SET status = 'completed', completed_at = CURRENT_TIMESTAMP
                    WHERE id = ?
                """, (st.session_state.current_conversation_id,))
                conn.commit()
        elif st.session_state.current_predefined_session_id:
            with sqlite3.connect(db_manager.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    UPDATE predefined_question_sessions 
                    SET status = 'completed', completed_at = CURRENT_TIMESTAMP
                    WHERE id = ?
                """, (st.session_state.current_predefined_session_id,))
                conn.commit()
    except Exception as e:
        st.error(f"Error marking session complete: {str(e)}")

def clear_session_state():
    """Clear session state for new session"""
    keys_to_clear = [
        'current_conversation_id', 'current_predefined_session_id', 
        'pdf_text_dict', 'qa_dict', 'all_qas', 'qa_index', 'used_q_indices', 
        'resume_session', 'resume_predefined_session', 'question_mode',
        'adaptive_mode', 'current_difficulty', 'last_answer_correct',
        'consecutive_wrong_same_level', 'difficulty_path', 'session_complete'
    ]
    for key in keys_to_clear:
        if key in st.session_state:
            del st.session_state[key]
    
    # Reset modules
    adaptive_engine.reset_state()
    selective_mutism_support.reset_state()

# ------------------ Run Main Application ------------------
if __name__ == "__main__":
    main()
