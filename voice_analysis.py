# -*- coding: utf-8 -*-
"""
Voice Analysis Module for EchoLearn
Advanced AI-powered analysis of speech patterns, filler words, pace, and confidence
"""

import re
import logging
import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from collections import Counter
import time

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Common filler words across languages
FILLER_WORDS = {
    'english': [
        'um', 'uh', 'er', 'ah', 'like', 'you know', 'basically', 'actually',
        'literally', 'honestly', 'right', 'so', 'well', 'i mean', 'kind of',
        'sort of', 'you see', 'anyway', 'whatever', 'stuff', 'things',
        'obviously', 'apparently', 'essentially', 'just', 'really'
    ]
}

# Technical terms that are commonly mispronounced (for pronunciation feedback)
TECHNICAL_TERMS = {
    'algorithm': 'AL-guh-rith-um',
    'asynchronous': 'ay-SING-kruh-nuhs',
    'cache': 'KASH',
    'concatenate': 'kon-KAT-uh-nayt',
    'deprecated': 'DEP-ruh-kay-tid',
    'epitome': 'ih-PIT-uh-mee',
    'facade': 'fuh-SAHD',
    'genre': 'ZHAHN-ruh',
    'hierarchy': 'HY-uh-rahr-kee',
    'hyperbole': 'hy-PUR-buh-lee',
    'mnemonic': 'neh-MON-ik',
    'paradigm': 'PAR-uh-dym',
    'queue': 'KYOO',
    'recursion': 'ri-KUR-zhun',
    'segue': 'SEG-way',
    'schema': 'SKEE-muh',
    'sudo': 'SOO-doo',
    'nginx': 'engine-X',
    'linux': 'LIN-uks',
    'sql': 'S-Q-L or sequel',
    'gui': 'GOO-ee',
    'api': 'A-P-I',
    'json': 'JAY-son',
    'ajax': 'AY-jaks',
    'oauth': 'OH-auth'
}


@dataclass
class VoiceAnalysisResult:
    """Data class for voice analysis results"""
    # Filler word analysis
    filler_word_count: int = 0
    filler_words_found: Dict[str, int] = field(default_factory=dict)
    filler_word_percentage: float = 0.0
    filler_word_rating: str = "Unknown"
    
    # Speaking pace analysis
    words_per_minute: float = 0.0
    total_words: int = 0
    speaking_duration_seconds: float = 0.0
    pace_rating: str = "Unknown"
    pace_feedback: str = ""
    
    # Confidence indicators
    confidence_score: float = 0.0  # 0-100
    confidence_indicators: Dict[str, any] = field(default_factory=dict)
    confidence_rating: str = "Unknown"
    
    # Pronunciation
    technical_terms_found: List[str] = field(default_factory=list)
    pronunciation_tips: List[Dict] = field(default_factory=list)
    
    # Overall feedback
    overall_score: float = 0.0
    strengths: List[str] = field(default_factory=list)
    areas_for_improvement: List[str] = field(default_factory=list)
    detailed_feedback: str = ""


class VoiceAnalyzer:
    """Advanced voice and speech pattern analyzer"""
    
    def __init__(self, language: str = 'english'):
        self.language = language
        self.filler_words = FILLER_WORDS.get(language, FILLER_WORDS['english'])
        self.technical_terms = TECHNICAL_TERMS
        
        # Optimal speaking pace ranges (words per minute)
        self.pace_ranges = {
            'too_slow': (0, 100),
            'slightly_slow': (100, 120),
            'optimal': (120, 160),
            'slightly_fast': (160, 180),
            'too_fast': (180, float('inf'))
        }
    
    def analyze_speech(self, transcript: str, audio_duration_seconds: float = None,
                       audio_data: np.ndarray = None, sample_rate: int = 44100) -> VoiceAnalysisResult:
        """
        Comprehensive speech analysis
        
        Args:
            transcript: The transcribed text from speech
            audio_duration_seconds: Duration of the audio recording
            audio_data: Raw audio data (optional, for advanced analysis)
            sample_rate: Audio sample rate
        
        Returns:
            VoiceAnalysisResult with all analysis metrics
        """
        result = VoiceAnalysisResult()
        
        if not transcript or not transcript.strip():
            result.detailed_feedback = "No speech detected. Please try speaking into the microphone."
            return result
        
        # Clean and prepare transcript
        clean_transcript = self._clean_transcript(transcript)
        words = clean_transcript.split()
        result.total_words = len(words)
        
        # 1. Filler Word Analysis
        self._analyze_filler_words(transcript, result)
        
        # 2. Speaking Pace Analysis
        if audio_duration_seconds:
            self._analyze_speaking_pace(result.total_words, audio_duration_seconds, result)
        
        # 3. Confidence Analysis
        self._analyze_confidence(transcript, audio_data, sample_rate, result)
        
        # 4. Technical Term & Pronunciation Analysis
        self._analyze_pronunciation(transcript, result)
        
        # 5. Calculate Overall Score & Generate Feedback
        self._calculate_overall_score(result)
        self._generate_detailed_feedback(result)
        
        return result
    
    def _clean_transcript(self, transcript: str) -> str:
        """Clean transcript for word counting"""
        # Remove punctuation but keep words
        clean = re.sub(r'[^\w\s]', '', transcript.lower())
        return ' '.join(clean.split())
    
    def _analyze_filler_words(self, transcript: str, result: VoiceAnalysisResult) -> None:
        """Analyze filler word usage"""
        transcript_lower = transcript.lower()
        words = transcript_lower.split()
        total_words = len(words)
        
        filler_counts = Counter()
        
        # Count single-word fillers
        for word in words:
            clean_word = re.sub(r'[^\w]', '', word)
            if clean_word in self.filler_words:
                filler_counts[clean_word] += 1
        
        # Count multi-word fillers (like "you know", "kind of")
        for filler in self.filler_words:
            if ' ' in filler:
                count = transcript_lower.count(filler)
                if count > 0:
                    filler_counts[filler] += count
        
        result.filler_words_found = dict(filler_counts)
        result.filler_word_count = sum(filler_counts.values())
        
        if total_words > 0:
            result.filler_word_percentage = (result.filler_word_count / total_words) * 100
        
        # Rate filler word usage
        if result.filler_word_percentage < 2:
            result.filler_word_rating = "Excellent"
        elif result.filler_word_percentage < 5:
            result.filler_word_rating = "Good"
        elif result.filler_word_percentage < 10:
            result.filler_word_rating = "Fair"
        else:
            result.filler_word_rating = "Needs Work"
    
    def _analyze_speaking_pace(self, word_count: int, duration_seconds: float, 
                               result: VoiceAnalysisResult) -> None:
        """Analyze speaking pace"""
        result.speaking_duration_seconds = duration_seconds
        
        if duration_seconds > 0:
            result.words_per_minute = (word_count / duration_seconds) * 60
        
        wpm = result.words_per_minute
        
        # Determine pace rating and feedback
        if wpm < self.pace_ranges['too_slow'][1]:
            result.pace_rating = "Too Slow"
            result.pace_feedback = "Try to speak a bit faster. Your current pace might lose the examiner's attention."
        elif wpm < self.pace_ranges['slightly_slow'][1]:
            result.pace_rating = "Slightly Slow"
            result.pace_feedback = "Good pace, but you could be slightly more energetic."
        elif wpm < self.pace_ranges['optimal'][1]:
            result.pace_rating = "Optimal"
            result.pace_feedback = "Excellent! Your speaking pace is ideal for a viva examination."
        elif wpm < self.pace_ranges['slightly_fast'][1]:
            result.pace_rating = "Slightly Fast"
            result.pace_feedback = "Good energy! Try slowing down slightly for complex explanations."
        else:
            result.pace_rating = "Too Fast"
            result.pace_feedback = "Slow down! Speaking too fast can make you seem nervous and may confuse the examiner."
    
    def _analyze_confidence(self, transcript: str, audio_data: np.ndarray,
                           sample_rate: int, result: VoiceAnalysisResult) -> None:
        """Analyze confidence indicators from speech"""
        confidence_score = 70  # Base confidence score
        indicators = {}
        
        # 1. Hedging language analysis (decreases confidence score)
        hedging_phrases = [
            'i think', 'maybe', 'perhaps', 'possibly', 'might be', 'could be',
            'i guess', 'not sure', 'i believe', 'sort of', 'kind of', 
            'probably', 'it seems', 'in my opinion'
        ]
        
        transcript_lower = transcript.lower()
        hedging_count = sum(1 for phrase in hedging_phrases if phrase in transcript_lower)
        indicators['hedging_phrases'] = hedging_count
        confidence_score -= hedging_count * 3
        
        # 2. Assertive language analysis (increases confidence score)
        assertive_phrases = [
            'definitely', 'certainly', 'clearly', 'obviously', 'specifically',
            'the answer is', 'this means', 'this is because', 'therefore',
            'as a result', 'in conclusion', 'to summarize'
        ]
        
        assertive_count = sum(1 for phrase in assertive_phrases if phrase in transcript_lower)
        indicators['assertive_phrases'] = assertive_count
        confidence_score += assertive_count * 4
        
        # 3. Sentence structure analysis
        sentences = re.split(r'[.!?]+', transcript)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        # Check for complete sentences (increases confidence)
        complete_sentences = sum(1 for s in sentences if len(s.split()) >= 5)
        indicators['complete_sentences'] = complete_sentences
        confidence_score += complete_sentences * 2
        
        # 4. Question marks at end of statements (decreases confidence - uptalk)
        uptalk_count = transcript.count('?')
        indicators['uptalk_instances'] = uptalk_count
        confidence_score -= uptalk_count * 2
        
        # 5. Filler word impact on confidence
        if result.filler_word_percentage > 10:
            confidence_score -= 10
        elif result.filler_word_percentage > 5:
            confidence_score -= 5
        
        # 6. Audio-based confidence analysis (if audio data provided)
        if audio_data is not None and len(audio_data) > 0:
            audio_confidence = self._analyze_audio_confidence(audio_data, sample_rate)
            indicators.update(audio_confidence['indicators'])
            confidence_score += audio_confidence['score_adjustment']
        
        # Clamp confidence score between 0 and 100
        confidence_score = max(0, min(100, confidence_score))
        result.confidence_score = confidence_score
        result.confidence_indicators = indicators
        
        # Rate confidence level
        if confidence_score >= 80:
            result.confidence_rating = "Very Confident"
        elif confidence_score >= 65:
            result.confidence_rating = "Confident"
        elif confidence_score >= 50:
            result.confidence_rating = "Moderately Confident"
        elif confidence_score >= 35:
            result.confidence_rating = "Somewhat Uncertain"
        else:
            result.confidence_rating = "Needs Confidence Building"
    
    def _analyze_audio_confidence(self, audio_data: np.ndarray, sample_rate: int) -> Dict:
        """Analyze audio characteristics for confidence indicators"""
        indicators = {}
        score_adjustment = 0
        
        try:
            # Volume consistency analysis
            chunk_size = sample_rate // 4  # 250ms chunks
            chunks = [audio_data[i:i+chunk_size] for i in range(0, len(audio_data), chunk_size) if len(audio_data[i:i+chunk_size]) == chunk_size]
            
            if chunks:
                # Calculate RMS for each chunk
                chunk_volumes = [np.sqrt(np.mean(chunk**2)) for chunk in chunks]
                volume_std = np.std(chunk_volumes)
                volume_mean = np.mean(chunk_volumes)
                
                # Volume consistency (lower std = more consistent = more confident)
                if volume_mean > 0:
                    volume_cv = volume_std / volume_mean  # Coefficient of variation
                    indicators['volume_consistency'] = 1 - min(volume_cv, 1)
                    
                    if volume_cv < 0.3:
                        score_adjustment += 5  # Very consistent volume
                    elif volume_cv > 0.6:
                        score_adjustment -= 5  # Very inconsistent volume
                
                # Volume level (too quiet = less confident)
                avg_db = 20 * np.log10(max(volume_mean, 1e-10))
                indicators['average_volume_db'] = avg_db
                
                if avg_db < -40:
                    score_adjustment -= 5  # Speaking too quietly
                    indicators['volume_warning'] = "Speaking too quietly"
                elif avg_db > -20:
                    score_adjustment += 3  # Good projection
                    indicators['volume_note'] = "Good voice projection"
        
        except Exception as e:
            logger.warning(f"Audio confidence analysis error: {e}")
        
        return {'indicators': indicators, 'score_adjustment': score_adjustment}
    
    def _analyze_pronunciation(self, transcript: str, result: VoiceAnalysisResult) -> None:
        """Identify technical terms and provide pronunciation guidance"""
        transcript_lower = transcript.lower()
        words = set(re.findall(r'\b\w+\b', transcript_lower))
        
        found_terms = []
        pronunciation_tips = []
        
        for term, pronunciation in self.technical_terms.items():
            if term in words:
                found_terms.append(term)
                pronunciation_tips.append({
                    'term': term,
                    'pronunciation': pronunciation,
                    'tip': f"'{term}' is pronounced as: {pronunciation}"
                })
        
        result.technical_terms_found = found_terms
        result.pronunciation_tips = pronunciation_tips
    
    def _calculate_overall_score(self, result: VoiceAnalysisResult) -> None:
        """Calculate overall speech quality score"""
        scores = []
        
        # Filler word score (0-100)
        if result.filler_word_percentage < 2:
            scores.append(100)
        elif result.filler_word_percentage < 5:
            scores.append(80)
        elif result.filler_word_percentage < 10:
            scores.append(60)
        else:
            scores.append(max(0, 100 - result.filler_word_percentage * 5))
        
        # Pace score (0-100)
        pace_scores = {
            "Optimal": 100,
            "Slightly Slow": 80,
            "Slightly Fast": 80,
            "Too Slow": 50,
            "Too Fast": 50,
            "Unknown": 70
        }
        scores.append(pace_scores.get(result.pace_rating, 70))
        
        # Confidence score already 0-100
        scores.append(result.confidence_score)
        
        # Calculate weighted average
        result.overall_score = sum(scores) / len(scores)
        
        # Determine strengths and areas for improvement
        if result.filler_word_rating in ["Excellent", "Good"]:
            result.strengths.append("Low filler word usage")
        else:
            result.areas_for_improvement.append("Reduce filler words (um, uh, like)")
        
        if result.pace_rating == "Optimal":
            result.strengths.append("Excellent speaking pace")
        elif result.pace_rating in ["Too Slow", "Too Fast"]:
            result.areas_for_improvement.append(result.pace_feedback)
        
        if result.confidence_score >= 70:
            result.strengths.append("Good confidence indicators")
        else:
            result.areas_for_improvement.append("Work on sounding more confident (avoid hedging language)")
    
    def _generate_detailed_feedback(self, result: VoiceAnalysisResult) -> None:
        """Generate comprehensive feedback summary"""
        feedback_parts = []
        
        # Overall rating
        if result.overall_score >= 85:
            feedback_parts.append("🌟 **Excellent Performance!** Your speaking skills are impressive.")
        elif result.overall_score >= 70:
            feedback_parts.append("👍 **Good Performance!** You're on the right track with some areas to polish.")
        elif result.overall_score >= 55:
            feedback_parts.append("📈 **Decent Performance.** With practice, you can improve significantly.")
        else:
            feedback_parts.append("💪 **Keep Practicing!** Focus on the improvement areas below.")
        
        # Specific feedback
        feedback_parts.append(f"\n\n**Speaking Pace:** {result.pace_rating} ({result.words_per_minute:.0f} words/minute)")
        feedback_parts.append(f"**Filler Words:** {result.filler_word_count} found ({result.filler_word_percentage:.1f}% of speech)")
        feedback_parts.append(f"**Confidence Level:** {result.confidence_rating} ({result.confidence_score:.0f}/100)")
        
        # Top filler words to avoid
        if result.filler_words_found:
            top_fillers = sorted(result.filler_words_found.items(), key=lambda x: x[1], reverse=True)[:3]
            filler_str = ", ".join([f"'{w}' ({c}x)" for w, c in top_fillers])
            feedback_parts.append(f"\n**Most Used Fillers:** {filler_str}")
        
        # Pronunciation tips
        if result.pronunciation_tips:
            feedback_parts.append("\n**📚 Pronunciation Tips:**")
            for tip in result.pronunciation_tips[:3]:
                feedback_parts.append(f"  • {tip['tip']}")
        
        result.detailed_feedback = "\n".join(feedback_parts)


class VoiceCoach:
    """Interactive voice coaching with real-time tips"""
    
    def __init__(self):
        self.analyzer = VoiceAnalyzer()
        self.session_history = []
    
    def get_real_time_tip(self, current_wpm: float, recent_filler_count: int) -> Optional[str]:
        """Get a real-time coaching tip based on current performance"""
        tips = []
        
        if current_wpm > 180:
            tips.append("🐢 Slow down a bit - you're speaking quite fast!")
        elif current_wpm < 100:
            tips.append("🚀 Try to pick up the pace slightly")
        
        if recent_filler_count > 3:
            tips.append("💡 Pause instead of using 'um' or 'uh'")
        
        return tips[0] if tips else None
    
    def get_improvement_exercises(self, result: VoiceAnalysisResult) -> List[Dict]:
        """Get personalized improvement exercises"""
        exercises = []
        
        if result.filler_word_percentage > 5:
            exercises.append({
                'title': "Filler Word Elimination",
                'description': "Practice pausing silently instead of using filler words",
                'exercise': "Record yourself answering a question. Every time you want to say 'um' or 'like', take a breath instead. Silence is powerful!",
                'duration': "5 minutes daily"
            })
        
        if result.pace_rating in ["Too Fast", "Too Slow"]:
            exercises.append({
                'title': "Pace Control Practice",
                'description': "Develop awareness of your natural speaking speed",
                'exercise': f"Set a metronome to 120 BPM and practice speaking one word per beat. Current: {result.words_per_minute:.0f} WPM, Target: 130-150 WPM",
                'duration': "5 minutes daily"
            })
        
        if result.confidence_score < 70:
            exercises.append({
                'title': "Confidence Building",
                'description': "Replace uncertain language with assertive statements",
                'exercise': "Instead of 'I think it might be...', practice saying 'The answer is...' or 'This occurs because...'. Stand tall while speaking!",
                'duration': "Practice during every answer"
            })
        
        return exercises
    
    def track_session(self, result: VoiceAnalysisResult) -> Dict:
        """Track session for progress monitoring"""
        session_data = {
            'timestamp': time.time(),
            'overall_score': result.overall_score,
            'filler_percentage': result.filler_word_percentage,
            'wpm': result.words_per_minute,
            'confidence': result.confidence_score
        }
        self.session_history.append(session_data)
        
        # Calculate improvement if there's history
        improvement = {}
        if len(self.session_history) >= 2:
            prev = self.session_history[-2]
            improvement = {
                'overall_change': session_data['overall_score'] - prev['overall_score'],
                'filler_change': prev['filler_percentage'] - session_data['filler_percentage'],  # Lower is better
                'confidence_change': session_data['confidence'] - prev['confidence']
            }
        
        return {'current': session_data, 'improvement': improvement}


# Streamlit integration helper
def display_voice_analysis_ui(transcript: str, audio_duration: float = None, 
                              audio_data: np.ndarray = None) -> VoiceAnalysisResult:
    """Display voice analysis results in Streamlit UI"""
    import streamlit as st
    
    analyzer = VoiceAnalyzer()
    result = analyzer.analyze_speech(transcript, audio_duration, audio_data)
    
    st.markdown("### 🎙️ Voice Analysis Results")
    
    # Overall score gauge
    col1, col2, col3 = st.columns(3)
    
    with col1:
        score_color = "🟢" if result.overall_score >= 70 else "🟡" if result.overall_score >= 50 else "🔴"
        st.metric("Overall Score", f"{result.overall_score:.0f}/100", delta=None)
    
    with col2:
        st.metric("Speaking Pace", f"{result.words_per_minute:.0f} WPM", 
                  delta=result.pace_rating)
    
    with col3:
        st.metric("Confidence", f"{result.confidence_score:.0f}/100",
                  delta=result.confidence_rating)
    
    # Detailed breakdown
    with st.expander("📊 Detailed Analysis", expanded=True):
        st.markdown(result.detailed_feedback)
        
        if result.strengths:
            st.success("**Strengths:** " + " • ".join(result.strengths))
        
        if result.areas_for_improvement:
            st.warning("**Areas to Improve:** " + " • ".join(result.areas_for_improvement))
    
    # Filler words breakdown
    if result.filler_words_found:
        with st.expander("🔤 Filler Words Breakdown"):
            for word, count in sorted(result.filler_words_found.items(), key=lambda x: x[1], reverse=True):
                st.write(f"• **'{word}'**: {count} times")
    
    return result


# Create global instance for import
voice_analyzer = VoiceAnalyzer()
voice_coach = VoiceCoach()
