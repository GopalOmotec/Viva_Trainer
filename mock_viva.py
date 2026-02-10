# -*- coding: utf-8 -*-
"""
Mock Viva Interview Module for EchoLearn
AI-powered interactive viva examination with follow-up questions and different interviewer personas
"""

import streamlit as st
import time
import random
import logging
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class InterviewerPersona(Enum):
    """Different interviewer personality types"""
    FRIENDLY = "friendly"
    TOUGH = "tough"
    INDUSTRY = "industry"
    ACADEMIC = "academic"
    QUICK_FIRE = "quick_fire"


@dataclass
class InterviewerProfile:
    """Profile for an interviewer persona"""
    name: str
    title: str
    style: str
    emoji: str
    greeting: str
    follow_up_style: str
    encouragement_level: float  # 0-1, how encouraging
    difficulty_multiplier: float  # 1.0 = normal, 1.5 = harder
    time_pressure: bool
    characteristics: List[str]


# Predefined interviewer profiles
INTERVIEWER_PROFILES = {
    InterviewerPersona.FRIENDLY: InterviewerProfile(
        name="Dr. Sarah Chen",
        title="Supportive Academic Mentor",
        style="Warm, encouraging, and patient",
        emoji="😊",
        greeting="Hello! I'm so glad you're here today. Don't worry about being perfect - just share what you know and we'll explore the topic together!",
        follow_up_style="gentle_probing",
        encouragement_level=0.9,
        difficulty_multiplier=0.9,
        time_pressure=False,
        characteristics=[
            "Gives hints when you struggle",
            "Celebrates partial answers",
            "Uses encouraging language",
            "Allows time to think"
        ]
    ),
    InterviewerPersona.TOUGH: InterviewerProfile(
        name="Prof. Richard Blackwell",
        title="Rigorous Examiner",
        style="Direct, challenging, expects precision",
        emoji="🧐",
        greeting="Good day. I expect clear, precise answers backed by solid understanding. Let's see how well you really know this material.",
        follow_up_style="challenging",
        encouragement_level=0.3,
        difficulty_multiplier=1.3,
        time_pressure=True,
        characteristics=[
            "Asks deep follow-up questions",
            "Challenges assumptions",
            "Expects technical precision",
            "Limited hints provided"
        ]
    ),
    InterviewerPersona.INDUSTRY: InterviewerProfile(
        name="Alex Morgan",
        title="Tech Industry Veteran",
        style="Practical, real-world focused",
        emoji="💼",
        greeting="Hey! I'm more interested in how you'd apply this knowledge in the real world. Theory is great, but let's talk practical applications.",
        follow_up_style="practical",
        encouragement_level=0.6,
        difficulty_multiplier=1.1,
        time_pressure=False,
        characteristics=[
            "Focuses on practical applications",
            "Asks 'how would you use this?'",
            "Values problem-solving approach",
            "Interested in real-world examples"
        ]
    ),
    InterviewerPersona.ACADEMIC: InterviewerProfile(
        name="Dr. Emily Watson",
        title="Research-Focused Academic",
        style="Theoretical, deep conceptual understanding",
        emoji="📚",
        greeting="Welcome. I'm particularly interested in your conceptual understanding and ability to connect different ideas. Let's dive deep into the theory.",
        follow_up_style="theoretical",
        encouragement_level=0.5,
        difficulty_multiplier=1.2,
        time_pressure=False,
        characteristics=[
            "Asks about underlying principles",
            "Explores edge cases",
            "Values conceptual connections",
            "Encourages critical thinking"
        ]
    ),
    InterviewerPersona.QUICK_FIRE: InterviewerProfile(
        name="Quiz Master",
        title="Rapid Assessment Mode",
        style="Fast-paced, time-pressured",
        emoji="⚡",
        greeting="Ready for a challenge? Quick answers only - trust your first instinct! 30 seconds per question. Let's go!",
        follow_up_style="rapid",
        encouragement_level=0.4,
        difficulty_multiplier=1.0,
        time_pressure=True,
        characteristics=[
            "30-second time limit per question",
            "No follow-ups, just next question",
            "Tests recall speed",
            "Builds exam pressure tolerance"
        ]
    )
}


@dataclass
class ConversationTurn:
    """Represents a single turn in the conversation"""
    speaker: str  # 'interviewer' or 'student'
    message: str
    timestamp: float
    turn_type: str  # 'question', 'answer', 'follow_up', 'hint', 'feedback'
    score: Optional[int] = None
    time_taken: Optional[float] = None


@dataclass
class MockVivaSession:
    """Represents a complete mock viva session"""
    session_id: str
    persona: InterviewerPersona
    profile: InterviewerProfile
    topic: str
    questions: List[Dict]
    conversation_history: List[ConversationTurn] = field(default_factory=list)
    current_question_index: int = 0
    total_score: int = 0
    max_score: int = 0
    follow_up_count: int = 0
    hints_used: int = 0
    started_at: float = field(default_factory=time.time)
    ended_at: Optional[float] = None
    is_complete: bool = False


class MockVivaInterviewer:
    """AI-powered mock viva interviewer"""
    
    def __init__(self, llm):
        self.llm = llm
        self.current_session: Optional[MockVivaSession] = None
    
    def start_session(self, persona: InterviewerPersona, topic: str, 
                      questions: List[Dict], session_id: str = None) -> MockVivaSession:
        """Start a new mock viva session"""
        profile = INTERVIEWER_PROFILES[persona]
        
        self.current_session = MockVivaSession(
            session_id=session_id or f"viva_{int(time.time())}",
            persona=persona,
            profile=profile,
            topic=topic,
            questions=questions
        )
        
        # Add greeting to conversation
        self.current_session.conversation_history.append(ConversationTurn(
            speaker='interviewer',
            message=profile.greeting,
            timestamp=time.time(),
            turn_type='greeting'
        ))
        
        return self.current_session
    
    def get_current_question(self) -> Optional[Dict]:
        """Get the current question"""
        if not self.current_session:
            return None
        
        idx = self.current_session.current_question_index
        if idx < len(self.current_session.questions):
            return self.current_session.questions[idx]
        return None
    
    def ask_question(self) -> str:
        """Interviewer asks the current question"""
        if not self.current_session:
            return "Session not started"
        
        question = self.get_current_question()
        if not question:
            return "No more questions"
        
        profile = self.current_session.profile
        
        # Format question based on interviewer style
        question_text = question.get('question', '')
        
        styled_question = self._style_question(question_text, profile)
        
        self.current_session.conversation_history.append(ConversationTurn(
            speaker='interviewer',
            message=styled_question,
            timestamp=time.time(),
            turn_type='question'
        ))
        
        return styled_question
    
    def _style_question(self, question: str, profile: InterviewerProfile) -> str:
        """Style the question based on interviewer persona"""
        prefixes = {
            "friendly": [
                "Let's explore this together: ",
                "I'd love to hear your thoughts on: ",
                "Here's an interesting one: ",
            ],
            "tough": [
                "Explain precisely: ",
                "Define and elaborate: ",
                "Demonstrate your understanding of: ",
            ],
            "industry": [
                "In a real-world scenario: ",
                "How would you apply: ",
                "Practically speaking: ",
            ],
            "academic": [
                "From a theoretical standpoint: ",
                "Considering the underlying principles: ",
                "Conceptually speaking: ",
            ],
            "quick_fire": [
                "Quick! ",
                "Fast answer: ",
                "",
            ]
        }
        
        prefix = random.choice(prefixes.get(profile.style.split(',')[0].lower().replace(' ', '_'), [""]))
        if prefix and not prefix.endswith(' '):
            prefix = ""
        
        return f"{prefix}{question}"
    
    def evaluate_answer(self, student_answer: str, time_taken: float = None) -> Dict:
        """Evaluate student's answer and decide on follow-up"""
        if not self.current_session:
            return {'error': 'No active session'}
        
        question = self.get_current_question()
        if not question:
            return {'error': 'No current question'}
        
        profile = self.current_session.profile
        correct_answer = question.get('answer', '')
        
        # Record student's answer
        self.current_session.conversation_history.append(ConversationTurn(
            speaker='student',
            message=student_answer,
            timestamp=time.time(),
            turn_type='answer',
            time_taken=time_taken
        ))
        
        # Evaluate the answer using LLM
        evaluation = self._evaluate_with_llm(
            question['question'], 
            correct_answer, 
            student_answer,
            profile
        )
        
        # Update scores
        score = evaluation.get('score', 5)
        self.current_session.total_score += score
        self.current_session.max_score += 10
        
        # Decide on follow-up based on persona and answer quality
        follow_up = self._generate_follow_up(evaluation, profile)
        
        return {
            'evaluation': evaluation,
            'follow_up': follow_up,
            'score': score,
            'should_follow_up': follow_up is not None and profile.follow_up_style != 'rapid'
        }
    
    def _evaluate_with_llm(self, question: str, correct_answer: str, 
                          student_answer: str, profile: InterviewerProfile) -> Dict:
        """Use LLM to evaluate the answer"""
        try:
            eval_prompt = f"""
You are {profile.name}, a {profile.title} with a {profile.style} approach.

Question: {question}
Correct Answer: {correct_answer}
Student's Answer: {student_answer}

Evaluate the student's answer considering:
1. Accuracy (how correct is the answer?)
2. Completeness (did they cover key points?)
3. Understanding (do they show genuine comprehension?)

Your evaluation style: {profile.follow_up_style}
Difficulty adjustment: {profile.difficulty_multiplier}x
Encouragement level: {profile.encouragement_level * 100}%

Provide your evaluation in this format:
SCORE: [0-10]
ACCURACY: [brief assessment]
COMPLETENESS: [brief assessment]
KEY_GAPS: [what was missed, if anything]
FEEDBACK: [feedback in character as {profile.name}]
FOLLOW_UP_NEEDED: [yes/no - does this need deeper exploration?]
FOLLOW_UP_TOPIC: [if yes, what aspect to probe deeper]
"""
            result = self.llm.invoke(eval_prompt)
            response = result.strip() if isinstance(result, str) else result.content.strip()
            
            # Parse response
            import re
            score_match = re.search(r'SCORE:\s*(\d+)', response, re.IGNORECASE)
            accuracy_match = re.search(r'ACCURACY:\s*(.+?)(?=COMPLETENESS:|$)', response, re.IGNORECASE | re.DOTALL)
            completeness_match = re.search(r'COMPLETENESS:\s*(.+?)(?=KEY_GAPS:|$)', response, re.IGNORECASE | re.DOTALL)
            gaps_match = re.search(r'KEY_GAPS:\s*(.+?)(?=FEEDBACK:|$)', response, re.IGNORECASE | re.DOTALL)
            feedback_match = re.search(r'FEEDBACK:\s*(.+?)(?=FOLLOW_UP_NEEDED:|$)', response, re.IGNORECASE | re.DOTALL)
            follow_up_match = re.search(r'FOLLOW_UP_NEEDED:\s*(.+?)(?=FOLLOW_UP_TOPIC:|$)', response, re.IGNORECASE | re.DOTALL)
            topic_match = re.search(r'FOLLOW_UP_TOPIC:\s*(.+?)$', response, re.IGNORECASE | re.DOTALL)
            
            return {
                'score': int(score_match.group(1)) if score_match else 5,
                'accuracy': accuracy_match.group(1).strip() if accuracy_match else "N/A",
                'completeness': completeness_match.group(1).strip() if completeness_match else "N/A",
                'key_gaps': gaps_match.group(1).strip() if gaps_match else "None identified",
                'feedback': feedback_match.group(1).strip() if feedback_match else "Good attempt.",
                'follow_up_needed': 'yes' in follow_up_match.group(1).lower() if follow_up_match else False,
                'follow_up_topic': topic_match.group(1).strip() if topic_match else None
            }
            
        except Exception as e:
            logger.error(f"LLM evaluation error: {e}")
            return {
                'score': 5,
                'accuracy': "Evaluation error",
                'completeness': "Unknown",
                'key_gaps': "Unknown",
                'feedback': "There was an error evaluating your answer. Please try again.",
                'follow_up_needed': False,
                'follow_up_topic': None
            }
    
    def _generate_follow_up(self, evaluation: Dict, profile: InterviewerProfile) -> Optional[str]:
        """Generate a follow-up question based on evaluation and persona"""
        if not evaluation.get('follow_up_needed'):
            return None
        
        score = evaluation.get('score', 5)
        gaps = evaluation.get('key_gaps', '')
        topic = evaluation.get('follow_up_topic', '')
        
        try:
            follow_up_prompt = f"""
You are {profile.name}, a {profile.title}.
The student scored {score}/10 on the previous question.
Key gaps identified: {gaps}
Topic to probe: {topic}

Your follow-up style: {profile.follow_up_style}

Generate ONE follow-up question in character. The question should:
- Probe deeper into the topic
- Match your interviewer personality
- Be appropriate for the gap identified

Just provide the follow-up question, nothing else.
"""
            result = self.llm.invoke(follow_up_prompt)
            follow_up = result.strip() if isinstance(result, str) else result.content.strip()
            
            self.current_session.follow_up_count += 1
            self.current_session.conversation_history.append(ConversationTurn(
                speaker='interviewer',
                message=follow_up,
                timestamp=time.time(),
                turn_type='follow_up'
            ))
            
            return follow_up
            
        except Exception as e:
            logger.error(f"Follow-up generation error: {e}")
            return None
    
    def get_hint(self) -> Optional[str]:
        """Get a hint for the current question (if persona allows)"""
        if not self.current_session:
            return None
        
        profile = self.current_session.profile
        if profile.encouragement_level < 0.5:
            return "No hints available with this examiner."
        
        question = self.get_current_question()
        if not question:
            return None
        
        try:
            hint_prompt = f"""
Question: {question.get('question', '')}
Answer: {question.get('answer', '')}

As {profile.name}, provide a helpful hint that guides the student toward the answer without giving it away.
Keep it brief (1-2 sentences). Be encouraging.
"""
            result = self.llm.invoke(hint_prompt)
            hint = result.strip() if isinstance(result, str) else result.content.strip()
            
            self.current_session.hints_used += 1
            self.current_session.conversation_history.append(ConversationTurn(
                speaker='interviewer',
                message=f"💡 Hint: {hint}",
                timestamp=time.time(),
                turn_type='hint'
            ))
            
            return hint
            
        except Exception as e:
            logger.error(f"Hint generation error: {e}")
            return "Think about the key concepts we've discussed."
    
    def next_question(self) -> bool:
        """Move to the next question"""
        if not self.current_session:
            return False
        
        self.current_session.current_question_index += 1
        
        if self.current_session.current_question_index >= len(self.current_session.questions):
            self.end_session()
            return False
        
        return True
    
    def end_session(self) -> Dict:
        """End the current session and get summary"""
        if not self.current_session:
            return {}
        
        self.current_session.ended_at = time.time()
        self.current_session.is_complete = True
        
        # Generate session summary
        summary = self._generate_session_summary()
        
        return summary
    
    def _generate_session_summary(self) -> Dict:
        """Generate a comprehensive session summary"""
        session = self.current_session
        profile = session.profile
        
        duration = (session.ended_at or time.time()) - session.started_at
        avg_score = session.total_score / max(1, session.current_question_index + 1)
        
        # Performance rating
        if avg_score >= 8:
            performance_rating = "Excellent"
            performance_emoji = "🌟"
        elif avg_score >= 6:
            performance_rating = "Good"
            performance_emoji = "👍"
        elif avg_score >= 4:
            performance_rating = "Fair"
            performance_emoji = "📈"
        else:
            performance_rating = "Needs Improvement"
            performance_emoji = "💪"
        
        return {
            'session_id': session.session_id,
            'interviewer': profile.name,
            'interviewer_style': profile.title,
            'topic': session.topic,
            'questions_answered': session.current_question_index + 1,
            'total_questions': len(session.questions),
            'total_score': session.total_score,
            'max_score': session.max_score,
            'average_score': avg_score,
            'follow_ups': session.follow_up_count,
            'hints_used': session.hints_used,
            'duration_seconds': duration,
            'duration_formatted': f"{int(duration // 60)}m {int(duration % 60)}s",
            'performance_rating': performance_rating,
            'performance_emoji': performance_emoji,
            'conversation_turns': len(session.conversation_history)
        }


class MockVivaUI:
    """Streamlit UI for Mock Viva sessions"""
    
    def __init__(self, llm):
        self.interviewer = MockVivaInterviewer(llm)
    
    def display_persona_selector(self) -> InterviewerPersona:
        """Display interviewer persona selection UI"""
        st.markdown("### 👤 Choose Your Interviewer")
        
        cols = st.columns(len(INTERVIEWER_PROFILES))
        
        selected_persona = None
        
        for i, (persona, profile) in enumerate(INTERVIEWER_PROFILES.items()):
            with cols[i]:
                st.markdown(f"#### {profile.emoji} {profile.name}")
                st.caption(profile.title)
                st.markdown(f"*{profile.style}*")
                
                with st.expander("Details"):
                    for char in profile.characteristics:
                        st.write(f"• {char}")
                
                if st.button(f"Select", key=f"persona_{persona.value}"):
                    selected_persona = persona
        
        return selected_persona
    
    def display_session_ui(self, session: MockVivaSession, questions: List[Dict]):
        """Display the main mock viva session UI"""
        profile = session.profile
        
        # Header
        st.markdown(f"## {profile.emoji} Mock Viva with {profile.name}")
        st.caption(f"*{profile.title}* | Topic: {session.topic}")
        
        # Progress bar
        progress = (session.current_question_index + 1) / len(session.questions)
        st.progress(progress)
        st.caption(f"Question {session.current_question_index + 1} of {len(session.questions)}")
        
        # Score display
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Current Score", f"{session.total_score}/{session.max_score}")
        with col2:
            st.metric("Follow-ups", session.follow_up_count)
        with col3:
            st.metric("Hints Used", session.hints_used)
        
        # Conversation display
        st.markdown("---")
        self._display_conversation(session)
        
        return session
    
    def _display_conversation(self, session: MockVivaSession):
        """Display the conversation history"""
        profile = session.profile
        
        for turn in session.conversation_history:
            if turn.speaker == 'interviewer':
                st.markdown(f"**{profile.emoji} {profile.name}:** {turn.message}")
            else:
                st.markdown(f"**🎓 You:** {turn.message}")
    
    def display_session_summary(self, summary: Dict):
        """Display the session summary"""
        st.markdown("## 🎉 Mock Viva Complete!")
        st.balloons()
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown(f"### {summary['performance_emoji']} {summary['performance_rating']}")
            st.metric("Final Score", f"{summary['total_score']}/{summary['max_score']}")
            st.metric("Average Score", f"{summary['average_score']:.1f}/10")
        
        with col2:
            st.metric("Questions Completed", f"{summary['questions_answered']}/{summary['total_questions']}")
            st.metric("Duration", summary['duration_formatted'])
            st.metric("Follow-up Questions", summary['follow_ups'])
        
        # Detailed feedback
        with st.expander("📊 Detailed Analysis"):
            st.write(f"**Interviewer:** {summary['interviewer']} ({summary['interviewer_style']})")
            st.write(f"**Topic:** {summary['topic']}")
            st.write(f"**Hints Used:** {summary['hints_used']}")
            st.write(f"**Conversation Turns:** {summary['conversation_turns']}")


# Helper functions for integration
def create_mock_viva_session(llm, persona: InterviewerPersona, topic: str, 
                             questions: List[Dict]) -> MockVivaSession:
    """Create and start a new mock viva session"""
    interviewer = MockVivaInterviewer(llm)
    return interviewer.start_session(persona, topic, questions)


def get_available_personas() -> Dict[InterviewerPersona, InterviewerProfile]:
    """Get all available interviewer personas"""
    return INTERVIEWER_PROFILES


# Create module-level instance
mock_viva_ui = None  # Will be initialized with LLM when needed
