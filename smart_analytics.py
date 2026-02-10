# -*- coding: utf-8 -*-
"""
Smart Analytics Dashboard Module for EchoLearn
Comprehensive learning analytics with weak topic detection, heatmaps, and predictive scoring
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from collections import defaultdict
import logging
import json

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class TopicPerformance:
    """Performance data for a specific topic"""
    topic_name: str
    total_questions: int = 0
    correct_answers: int = 0
    total_score: int = 0
    max_score: int = 0
    average_score: float = 0.0
    accuracy_rate: float = 0.0
    difficulty_distribution: Dict[str, int] = field(default_factory=dict)
    time_spent_seconds: float = 0.0
    last_practiced: Optional[datetime] = None
    trend: str = "stable"  # improving, declining, stable


@dataclass
class LearningPattern:
    """Detected learning patterns"""
    best_time_of_day: str = "Unknown"
    best_day_of_week: str = "Unknown"
    average_session_duration: float = 0.0
    peak_performance_hour: int = 0
    consistency_score: float = 0.0
    study_streak: int = 0
    longest_streak: int = 0


@dataclass
class PredictedPerformance:
    """Predicted exam performance"""
    predicted_score: float = 0.0
    confidence_interval: Tuple[float, float] = (0.0, 0.0)
    readiness_level: str = "Unknown"
    weak_topics: List[str] = field(default_factory=list)
    strong_topics: List[str] = field(default_factory=list)
    recommended_study_time: float = 0.0
    prediction_confidence: float = 0.0


class SmartAnalytics:
    """Advanced learning analytics engine"""
    
    def __init__(self, db_manager=None):
        self.db_manager = db_manager
        self.performance_cache = {}
    
    def analyze_topic_performance(self, user_id: int, performance_data: List[Dict]) -> Dict[str, TopicPerformance]:
        """Analyze performance across different topics"""
        topic_stats = defaultdict(lambda: TopicPerformance(topic_name=""))
        
        for record in performance_data:
            topic = record.get('topic') or record.get('subject', 'General')
            stats = topic_stats[topic]
            stats.topic_name = topic
            stats.total_questions += 1
            
            score = record.get('score', 0)
            stats.total_score += score
            stats.max_score += 10
            
            if score >= 6:
                stats.correct_answers += 1
            
            # Track difficulty distribution
            difficulty = record.get('difficulty', record.get('level', 'Medium'))
            if difficulty not in stats.difficulty_distribution:
                stats.difficulty_distribution[difficulty] = 0
            stats.difficulty_distribution[difficulty] += 1
            
            # Track time spent
            time_taken = record.get('time_taken', 30)  # Default 30 seconds
            stats.time_spent_seconds += time_taken
            
            # Track last practiced
            timestamp = record.get('timestamp') or record.get('answered_at')
            if timestamp:
                if isinstance(timestamp, str):
                    timestamp = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
                if stats.last_practiced is None or timestamp > stats.last_practiced:
                    stats.last_practiced = timestamp
        
        # Calculate derived metrics
        for topic, stats in topic_stats.items():
            if stats.total_questions > 0:
                stats.average_score = stats.total_score / stats.total_questions
                stats.accuracy_rate = (stats.correct_answers / stats.total_questions) * 100
            
            # Calculate trend based on recent performance
            stats.trend = self._calculate_topic_trend(performance_data, topic)
        
        return dict(topic_stats)
    
    def _calculate_topic_trend(self, performance_data: List[Dict], topic: str) -> str:
        """Calculate performance trend for a topic"""
        topic_records = [r for r in performance_data if (r.get('topic') or r.get('subject', 'General')) == topic]
        
        if len(topic_records) < 4:
            return "insufficient_data"
        
        # Sort by timestamp
        topic_records.sort(key=lambda x: x.get('timestamp', x.get('answered_at', '')))
        
        # Compare first half vs second half
        mid = len(topic_records) // 2
        first_half_avg = sum(r.get('score', 0) for r in topic_records[:mid]) / mid
        second_half_avg = sum(r.get('score', 0) for r in topic_records[mid:]) / (len(topic_records) - mid)
        
        if second_half_avg > first_half_avg + 1:
            return "improving"
        elif second_half_avg < first_half_avg - 1:
            return "declining"
        else:
            return "stable"
    
    def detect_weak_topics(self, topic_performance: Dict[str, TopicPerformance], 
                          threshold: float = 60.0) -> List[Dict]:
        """Detect topics where user is struggling"""
        weak_topics = []
        
        for topic, stats in topic_performance.items():
            if stats.total_questions >= 3 and stats.accuracy_rate < threshold:
                severity = "critical" if stats.accuracy_rate < 40 else "moderate"
                
                weak_topics.append({
                    'topic': topic,
                    'accuracy_rate': stats.accuracy_rate,
                    'average_score': stats.average_score,
                    'questions_attempted': stats.total_questions,
                    'severity': severity,
                    'trend': stats.trend,
                    'recommendation': self._generate_topic_recommendation(stats)
                })
        
        # Sort by severity (lowest accuracy first)
        weak_topics.sort(key=lambda x: x['accuracy_rate'])
        
        return weak_topics
    
    def _generate_topic_recommendation(self, stats: TopicPerformance) -> str:
        """Generate personalized recommendation for a topic"""
        if stats.accuracy_rate < 30:
            return f"🚨 Review fundamental concepts of {stats.topic_name}. Consider starting from basics."
        elif stats.accuracy_rate < 50:
            return f"📚 Spend extra time on {stats.topic_name}. Focus on understanding core principles."
        elif stats.accuracy_rate < 60:
            return f"💡 You're close! Practice more {stats.topic_name} questions, especially at higher difficulties."
        else:
            return f"👍 Keep practicing {stats.topic_name} to maintain your skills."
    
    def analyze_learning_patterns(self, session_data: List[Dict]) -> LearningPattern:
        """Analyze when and how the user learns best"""
        pattern = LearningPattern()
        
        if not session_data:
            return pattern
        
        # Extract timestamps and scores
        hour_performance = defaultdict(list)
        day_performance = defaultdict(list)
        session_durations = []
        study_dates = set()
        
        for session in session_data:
            timestamp = session.get('timestamp') or session.get('created_at')
            if not timestamp:
                continue
            
            if isinstance(timestamp, str):
                try:
                    timestamp = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
                except:
                    continue
            
            hour = timestamp.hour
            day = timestamp.strftime('%A')
            score = session.get('score', session.get('average_score', 0))
            
            hour_performance[hour].append(score)
            day_performance[day].append(score)
            study_dates.add(timestamp.date())
            
            if 'duration' in session:
                session_durations.append(session['duration'])
        
        # Find best time of day
        if hour_performance:
            hour_averages = {h: sum(scores)/len(scores) for h, scores in hour_performance.items()}
            best_hour = max(hour_averages, key=hour_averages.get)
            pattern.peak_performance_hour = best_hour
            
            if best_hour < 12:
                pattern.best_time_of_day = "Morning"
            elif best_hour < 17:
                pattern.best_time_of_day = "Afternoon"
            else:
                pattern.best_time_of_day = "Evening"
        
        # Find best day of week
        if day_performance:
            day_averages = {d: sum(scores)/len(scores) for d, scores in day_performance.items()}
            pattern.best_day_of_week = max(day_averages, key=day_averages.get)
        
        # Calculate average session duration
        if session_durations:
            pattern.average_session_duration = sum(session_durations) / len(session_durations)
        
        # Calculate study streak
        if study_dates:
            sorted_dates = sorted(study_dates)
            current_streak = 1
            longest_streak = 1
            
            for i in range(1, len(sorted_dates)):
                if (sorted_dates[i] - sorted_dates[i-1]).days == 1:
                    current_streak += 1
                    longest_streak = max(longest_streak, current_streak)
                else:
                    current_streak = 1
            
            # Check if streak continues to today
            today = datetime.now().date()
            if sorted_dates[-1] >= today - timedelta(days=1):
                pattern.study_streak = current_streak
            else:
                pattern.study_streak = 0
            
            pattern.longest_streak = longest_streak
        
        # Calculate consistency score (how regularly they study)
        if study_dates:
            date_range = (max(study_dates) - min(study_dates)).days + 1
            if date_range > 0:
                pattern.consistency_score = (len(study_dates) / date_range) * 100
        
        return pattern
    
    def predict_exam_performance(self, topic_performance: Dict[str, TopicPerformance],
                                recent_scores: List[float], target_topics: List[str] = None) -> PredictedPerformance:
        """Predict likely exam performance using ML-inspired heuristics"""
        prediction = PredictedPerformance()
        
        if not topic_performance and not recent_scores:
            prediction.readiness_level = "Insufficient Data"
            return prediction
        
        # Calculate base predicted score from overall performance
        all_scores = recent_scores if recent_scores else []
        for stats in topic_performance.values():
            if stats.average_score > 0:
                all_scores.extend([stats.average_score] * stats.total_questions)
        
        if all_scores:
            # Weighted average favoring recent performance
            weights = np.linspace(0.5, 1.0, len(all_scores))
            prediction.predicted_score = np.average(all_scores, weights=weights)
            
            # Calculate confidence interval
            std_dev = np.std(all_scores)
            prediction.confidence_interval = (
                max(0, prediction.predicted_score - 1.96 * std_dev / np.sqrt(len(all_scores))),
                min(10, prediction.predicted_score + 1.96 * std_dev / np.sqrt(len(all_scores)))
            )
            
            # Prediction confidence based on sample size
            prediction.prediction_confidence = min(95, 50 + len(all_scores) * 2)
        
        # Identify weak and strong topics
        for topic, stats in topic_performance.items():
            if stats.accuracy_rate < 50:
                prediction.weak_topics.append(topic)
            elif stats.accuracy_rate >= 80:
                prediction.strong_topics.append(topic)
        
        # Determine readiness level
        if prediction.predicted_score >= 8:
            prediction.readiness_level = "Excellent - Ready for Exam! 🌟"
        elif prediction.predicted_score >= 6.5:
            prediction.readiness_level = "Good - Almost Ready 👍"
        elif prediction.predicted_score >= 5:
            prediction.readiness_level = "Moderate - More Practice Needed 📚"
        else:
            prediction.readiness_level = "Needs Work - Focus on Weak Areas 💪"
        
        # Recommended study time based on gaps
        hours_needed = len(prediction.weak_topics) * 2 + (10 - prediction.predicted_score) * 0.5
        prediction.recommended_study_time = max(0, hours_needed)
        
        return prediction
    
    def generate_activity_heatmap_data(self, session_data: List[Dict]) -> pd.DataFrame:
        """Generate data for GitHub-style activity heatmap"""
        # Create a DataFrame with date and activity count
        activity_counts = defaultdict(int)
        activity_scores = defaultdict(list)
        
        for session in session_data:
            timestamp = session.get('timestamp') or session.get('created_at') or session.get('answered_at')
            if not timestamp:
                continue
            
            if isinstance(timestamp, str):
                try:
                    timestamp = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
                except:
                    continue
            
            date_str = timestamp.strftime('%Y-%m-%d')
            activity_counts[date_str] += 1
            
            score = session.get('score', session.get('average_score'))
            if score is not None:
                activity_scores[date_str].append(score)
        
        # Create DataFrame for last 365 days
        end_date = datetime.now()
        start_date = end_date - timedelta(days=365)
        
        dates = []
        counts = []
        avg_scores = []
        
        current_date = start_date
        while current_date <= end_date:
            date_str = current_date.strftime('%Y-%m-%d')
            dates.append(current_date)
            counts.append(activity_counts.get(date_str, 0))
            
            if date_str in activity_scores:
                avg_scores.append(sum(activity_scores[date_str]) / len(activity_scores[date_str]))
            else:
                avg_scores.append(0)
            
            current_date += timedelta(days=1)
        
        return pd.DataFrame({
            'date': dates,
            'activity_count': counts,
            'average_score': avg_scores,
            'day_of_week': [d.strftime('%A') for d in dates],
            'week_number': [d.isocalendar()[1] for d in dates]
        })


class AnalyticsDashboard:
    """Streamlit dashboard for smart analytics"""
    
    def __init__(self, analytics_engine: SmartAnalytics = None):
        self.analytics = analytics_engine or SmartAnalytics()
    
    def display_full_dashboard(self, user_id: int, performance_data: List[Dict], 
                               session_data: List[Dict]):
        """Display the complete analytics dashboard"""
        st.markdown("# 📊 Smart Analytics Dashboard")
        st.markdown("---")
        
        # Quick stats row
        self._display_quick_stats(performance_data, session_data)
        
        # Tabs for different analytics views
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "🎯 Topic Analysis", 
            "📈 Performance Trends", 
            "🔥 Activity Heatmap",
            "🔮 Exam Predictor",
            "💡 Insights & Tips"
        ])
        
        with tab1:
            self._display_topic_analysis(user_id, performance_data)
        
        with tab2:
            self._display_performance_trends(performance_data)
        
        with tab3:
            self._display_activity_heatmap(session_data)
        
        with tab4:
            self._display_exam_predictor(user_id, performance_data)
        
        with tab5:
            self._display_insights_and_tips(user_id, performance_data, session_data)
    
    def _display_quick_stats(self, performance_data: List[Dict], session_data: List[Dict]):
        """Display quick overview statistics"""
        col1, col2, col3, col4, col5 = st.columns(5)
        
        total_questions = len(performance_data)
        correct_answers = sum(1 for p in performance_data if p.get('score', 0) >= 6)
        avg_score = sum(p.get('score', 0) for p in performance_data) / max(1, total_questions)
        
        # Calculate streak
        pattern = self.analytics.analyze_learning_patterns(session_data)
        
        with col1:
            st.metric("Total Questions", total_questions)
        
        with col2:
            accuracy = (correct_answers / max(1, total_questions)) * 100
            st.metric("Accuracy Rate", f"{accuracy:.1f}%")
        
        with col3:
            st.metric("Average Score", f"{avg_score:.1f}/10")
        
        with col4:
            streak_emoji = "🔥" if pattern.study_streak > 0 else "❄️"
            st.metric(f"{streak_emoji} Study Streak", f"{pattern.study_streak} days")
        
        with col5:
            st.metric("Best Day", pattern.best_day_of_week[:3] if pattern.best_day_of_week != "Unknown" else "N/A")
    
    def _display_topic_analysis(self, user_id: int, performance_data: List[Dict]):
        """Display topic-wise performance analysis"""
        st.markdown("### 📚 Topic Performance Breakdown")
        
        topic_perf = self.analytics.analyze_topic_performance(user_id, performance_data)
        
        if not topic_perf:
            st.info("No topic data available yet. Complete some questions to see your topic analysis!")
            return
        
        # Create topic comparison chart
        topics = list(topic_perf.keys())
        accuracies = [topic_perf[t].accuracy_rate for t in topics]
        avg_scores = [topic_perf[t].average_score for t in topics]
        question_counts = [topic_perf[t].total_questions for t in topics]
        
        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=("Accuracy by Topic", "Questions Attempted"),
            specs=[[{"type": "bar"}, {"type": "pie"}]]
        )
        
        # Accuracy bar chart
        colors = ['#ff6b6b' if acc < 50 else '#ffd93d' if acc < 70 else '#6bcb77' for acc in accuracies]
        fig.add_trace(
            go.Bar(x=topics, y=accuracies, marker_color=colors, name="Accuracy %"),
            row=1, col=1
        )
        
        # Questions pie chart
        fig.add_trace(
            go.Pie(labels=topics, values=question_counts, hole=0.4),
            row=1, col=2
        )
        
        fig.update_layout(height=400, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
        
        # Weak topics alert
        weak_topics = self.analytics.detect_weak_topics(topic_perf)
        if weak_topics:
            st.markdown("### ⚠️ Topics Needing Attention")
            for weak in weak_topics[:3]:
                severity_color = "🔴" if weak['severity'] == 'critical' else "🟡"
                trend_icon = "📈" if weak['trend'] == 'improving' else "📉" if weak['trend'] == 'declining' else "➡️"
                
                with st.expander(f"{severity_color} {weak['topic']} ({weak['accuracy_rate']:.1f}% accuracy) {trend_icon}"):
                    st.write(f"**Average Score:** {weak['average_score']:.1f}/10")
                    st.write(f"**Questions Attempted:** {weak['questions_attempted']}")
                    st.write(f"**Trend:** {weak['trend'].capitalize()}")
                    st.info(weak['recommendation'])
    
    def _display_performance_trends(self, performance_data: List[Dict]):
        """Display performance trends over time"""
        st.markdown("### 📈 Performance Trends")
        
        if len(performance_data) < 5:
            st.info("Need at least 5 answered questions to show trends. Keep practicing!")
            return
        
        # Sort by timestamp
        sorted_data = sorted(performance_data, key=lambda x: x.get('timestamp', x.get('answered_at', '')))
        
        # Create rolling average
        scores = [d.get('score', 0) for d in sorted_data]
        timestamps = list(range(len(scores)))
        
        window_size = min(5, len(scores))
        rolling_avg = pd.Series(scores).rolling(window=window_size).mean().tolist()
        
        fig = go.Figure()
        
        # Individual scores
        fig.add_trace(go.Scatter(
            x=timestamps, y=scores,
            mode='markers',
            name='Individual Scores',
            marker=dict(size=8, opacity=0.6)
        ))
        
        # Rolling average
        fig.add_trace(go.Scatter(
            x=timestamps, y=rolling_avg,
            mode='lines',
            name=f'{window_size}-Question Moving Average',
            line=dict(width=3, color='#ff6b6b')
        ))
        
        # Trend line
        z = np.polyfit(timestamps, scores, 1)
        p = np.poly1d(z)
        fig.add_trace(go.Scatter(
            x=timestamps, y=p(timestamps),
            mode='lines',
            name='Overall Trend',
            line=dict(width=2, dash='dash', color='#6bcb77')
        ))
        
        fig.update_layout(
            title="Score Progression Over Time",
            xaxis_title="Questions Answered",
            yaxis_title="Score",
            yaxis=dict(range=[0, 10]),
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Trend interpretation
        trend_slope = z[0]
        if trend_slope > 0.05:
            st.success("📈 **Great progress!** Your scores are trending upward!")
        elif trend_slope < -0.05:
            st.warning("📉 **Heads up!** Your scores have been declining. Consider reviewing fundamentals.")
        else:
            st.info("➡️ **Steady performance.** Try challenging yourself with harder questions!")
    
    def _display_activity_heatmap(self, session_data: List[Dict]):
        """Display GitHub-style activity heatmap"""
        st.markdown("### 🔥 Study Activity Heatmap")
        
        heatmap_data = self.analytics.generate_activity_heatmap_data(session_data)
        
        if heatmap_data['activity_count'].sum() == 0:
            st.info("No activity data yet. Start practicing to build your heatmap!")
            return
        
        # Create calendar heatmap
        # Reshape data for heatmap (7 days x 52 weeks)
        recent_data = heatmap_data.tail(365)
        
        # Create pivot table
        recent_data['week'] = recent_data['date'].dt.isocalendar().week
        recent_data['weekday'] = recent_data['date'].dt.weekday
        
        pivot = recent_data.pivot_table(
            values='activity_count', 
            index='weekday', 
            columns='week', 
            aggfunc='sum',
            fill_value=0
        )
        
        fig = px.imshow(
            pivot,
            labels=dict(x="Week of Year", y="Day of Week", color="Activity"),
            y=['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'],
            color_continuous_scale='Greens',
            aspect='auto'
        )
        
        fig.update_layout(
            title="Daily Study Activity (Past Year)",
            height=300
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Activity by day of week
        col1, col2 = st.columns(2)
        
        with col1:
            day_counts = recent_data.groupby('day_of_week')['activity_count'].sum()
            day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
            day_counts = day_counts.reindex(day_order, fill_value=0)
            
            fig2 = px.bar(
                x=day_counts.index, y=day_counts.values,
                title="Activity by Day of Week",
                color=day_counts.values,
                color_continuous_scale='Blues'
            )
            fig2.update_layout(showlegend=False, height=300)
            st.plotly_chart(fig2, use_container_width=True)
        
        with col2:
            # Activity by hour (if timestamp data is available)
            st.markdown("#### 🕐 Best Study Times")
            pattern = self.analytics.analyze_learning_patterns(session_data)
            st.write(f"**Best Time of Day:** {pattern.best_time_of_day}")
            st.write(f"**Best Day of Week:** {pattern.best_day_of_week}")
            st.write(f"**Peak Performance Hour:** {pattern.peak_performance_hour}:00")
            st.write(f"**Consistency Score:** {pattern.consistency_score:.1f}%")
    
    def _display_exam_predictor(self, user_id: int, performance_data: List[Dict]):
        """Display exam performance prediction"""
        st.markdown("### 🔮 Exam Performance Predictor")
        
        topic_perf = self.analytics.analyze_topic_performance(user_id, performance_data)
        recent_scores = [p.get('score', 0) for p in performance_data[-20:]]
        
        prediction = self.analytics.predict_exam_performance(topic_perf, recent_scores)
        
        # Predicted score gauge
        fig = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=prediction.predicted_score,
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': "Predicted Exam Score", 'font': {'size': 24}},
            gauge={
                'axis': {'range': [0, 10], 'tickwidth': 1},
                'bar': {'color': "darkblue"},
                'steps': [
                    {'range': [0, 4], 'color': '#ff6b6b'},
                    {'range': [4, 6], 'color': '#ffd93d'},
                    {'range': [6, 8], 'color': '#abe9cd'},
                    {'range': [8, 10], 'color': '#6bcb77'}
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': 6
                }
            }
        ))
        
        fig.update_layout(height=350)
        st.plotly_chart(fig, use_container_width=True)
        
        # Prediction details
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown(f"**Readiness Level:** {prediction.readiness_level}")
            st.markdown(f"**Prediction Confidence:** {prediction.prediction_confidence:.0f}%")
            st.markdown(f"**Score Range:** {prediction.confidence_interval[0]:.1f} - {prediction.confidence_interval[1]:.1f}")
            st.markdown(f"**Recommended Study Time:** {prediction.recommended_study_time:.1f} hours")
        
        with col2:
            if prediction.weak_topics:
                st.markdown("**⚠️ Focus Areas:**")
                for topic in prediction.weak_topics[:3]:
                    st.write(f"  • {topic}")
            
            if prediction.strong_topics:
                st.markdown("**✅ Strong Areas:**")
                for topic in prediction.strong_topics[:3]:
                    st.write(f"  • {topic}")
    
    def _display_insights_and_tips(self, user_id: int, performance_data: List[Dict], 
                                   session_data: List[Dict]):
        """Display personalized insights and study tips"""
        st.markdown("### 💡 Personalized Insights & Tips")
        
        pattern = self.analytics.analyze_learning_patterns(session_data)
        topic_perf = self.analytics.analyze_topic_performance(user_id, performance_data)
        
        insights = []
        
        # Study pattern insights
        if pattern.best_time_of_day != "Unknown":
            insights.append({
                'icon': '⏰',
                'title': 'Optimal Study Time',
                'insight': f"You perform best during the **{pattern.best_time_of_day}**. Try to schedule important study sessions around this time.",
                'type': 'tip'
            })
        
        if pattern.study_streak > 0:
            insights.append({
                'icon': '🔥',
                'title': 'Study Streak',
                'insight': f"Amazing! You're on a **{pattern.study_streak}-day streak**! Keep it going!",
                'type': 'success'
            })
        elif pattern.longest_streak > 5:
            insights.append({
                'icon': '💪',
                'title': 'Comeback Time',
                'insight': f"Your longest streak was **{pattern.longest_streak} days**. Let's beat that record!",
                'type': 'motivation'
            })
        
        # Topic-based insights
        weak_topics = self.analytics.detect_weak_topics(topic_perf)
        if weak_topics:
            top_weak = weak_topics[0]
            insights.append({
                'icon': '🎯',
                'title': 'Priority Focus Area',
                'insight': f"**{top_weak['topic']}** needs the most attention ({top_weak['accuracy_rate']:.0f}% accuracy). {top_weak['recommendation']}",
                'type': 'warning'
            })
        
        # Performance trend insight
        if len(performance_data) >= 10:
            recent_avg = sum(p.get('score', 0) for p in performance_data[-5:]) / 5
            earlier_avg = sum(p.get('score', 0) for p in performance_data[-10:-5]) / 5
            
            if recent_avg > earlier_avg + 0.5:
                insights.append({
                    'icon': '📈',
                    'title': 'Improving Trend',
                    'insight': f"Your recent performance is **improving**! Recent avg: {recent_avg:.1f} vs Earlier avg: {earlier_avg:.1f}",
                    'type': 'success'
                })
            elif recent_avg < earlier_avg - 0.5:
                insights.append({
                    'icon': '📉',
                    'title': 'Attention Needed',
                    'insight': "Your recent scores have dipped. Consider reviewing earlier material or taking a short break to refresh.",
                    'type': 'warning'
                })
        
        # Display insights
        for insight in insights:
            if insight['type'] == 'success':
                st.success(f"{insight['icon']} **{insight['title']}**: {insight['insight']}")
            elif insight['type'] == 'warning':
                st.warning(f"{insight['icon']} **{insight['title']}**: {insight['insight']}")
            elif insight['type'] == 'tip':
                st.info(f"{insight['icon']} **{insight['title']}**: {insight['insight']}")
            else:
                st.write(f"{insight['icon']} **{insight['title']}**: {insight['insight']}")
        
        # Study tips based on patterns
        st.markdown("---")
        st.markdown("### 📝 Study Recommendations")
        
        tips = [
            "🎯 **Spaced Repetition**: Review weak topics every 2-3 days for better retention.",
            "⏱️ **Pomodoro Technique**: Study in 25-minute focused blocks with 5-minute breaks.",
            "🗣️ **Active Recall**: Practice explaining concepts out loud as if teaching someone.",
            "📊 **Track Progress**: Review this dashboard weekly to stay motivated.",
            "😴 **Rest Well**: Good sleep is crucial for memory consolidation."
        ]
        
        for tip in tips[:3]:
            st.write(tip)


# Create global instances for import
smart_analytics = SmartAnalytics()
analytics_dashboard = AnalyticsDashboard(smart_analytics)
