import streamlit as st
import openai
import json
import time
from datetime import datetime
from typing import Dict, List, Optional
import sqlite3
import uuid
from dataclasses import dataclass, asdict
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd

# Configuration
st.set_page_config(
    page_title="AI Excel Mock Interviewer",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Initialize OpenAI client
@st.cache_resource
def get_openai_client():
    api_key = st.secrets.get("OPENAI_API_KEY") or st.sidebar.text_input("Enter OpenAI API Key", type="password")
    if not api_key:
        st.error("Please provide OpenAI API Key in secrets or sidebar")
        st.stop()
    return openai.OpenAI(api_key=api_key)

client = get_openai_client()

@dataclass
class InterviewState:
    session_id: str
    candidate_name: str
    current_phase: str
    question_count: int
    current_difficulty: int
    conversation_history: List[Dict]
    skill_scores: Dict[str, int]
    overall_performance: Dict[str, any]
    start_time: datetime
    
class ExcelInterviewer:
    def __init__(self):
        self.phases = ["introduction", "warm_up", "core_assessment", "deep_dive", "conclusion"]
        self.skills = ["formulas", "data_analysis", "pivot_tables", "vba", "problem_solving", "communication"]
        self.max_questions_per_phase = {"introduction": 2, "warm_up": 3, "core_assessment": 8, "deep_dive": 4, "conclusion": 1}
        
    def get_system_prompt(self, phase: str, difficulty: int) -> str:
        base_prompt = """You are an expert Excel interviewer conducting a technical interview. 
        Your role is to assess the candidate's Excel skills professionally and thoroughly.
        
        Key Guidelines:
        1. Ask ONE question at a time
        2. Be encouraging and professional
        3. Follow up on answers with clarifying questions when needed
        4. Adapt difficulty based on candidate responses
        5. Focus on practical Excel knowledge and problem-solving
        """
        
        phase_prompts = {
            "introduction": """You are in the INTRODUCTION phase. You are an AI Excel Interviewer. Welcome the candidate warmly, 
            explain the interview process (30-40 minutes, progressive difficulty, practical Excel focus), 
            and ask about their Excel background/experience level.
            DO NOT use placeholders like [Your Name] - identify yourself as 'AI Excel Interviewer' or similar.""",
            
            "warm_up": f"""You are in the WARM-UP phase. Ask basic Excel questions to make the candidate comfortable.
            Difficulty level: {difficulty}/10. Focus on fundamental concepts like basic formulas, cell references, formatting.""",
            
            "core_assessment": f"""You are in the CORE ASSESSMENT phase. This is the main evaluation.
            Difficulty level: {difficulty}/10. Ask about intermediate to advanced Excel topics based on their responses.
            Topics: complex formulas, data analysis, pivot tables, VLOOKUP/INDEX-MATCH, data validation, conditional formatting.""",
            
            "deep_dive": f"""You are in the DEEP DIVE phase. Explore advanced topics based on their expertise.
            Difficulty level: {difficulty}/10. Focus on: VBA basics, advanced analytics, data modeling, automation, 
            complex problem-solving scenarios.""",
            
            "conclusion": """You are in the CONCLUSION phase. Thank the candidate and inform them they'll 
            receive detailed feedback shortly. Ask if they have any questions about Excel or the interview process."""
        }
        
        return base_prompt + "\n\n" + phase_prompts.get(phase, "")
    
    def generate_question(self, state: InterviewState) -> str:
        """Generate next question based on current state"""
        try:
            messages = [
                {"role": "system", "content": self.get_system_prompt(state.current_phase, state.current_difficulty)},
            ]
            
            # Add conversation history (last 6 messages for context)
            messages.extend(state.conversation_history[-6:])
            
            response = client.chat.completions.create(
                model="gpt-4",
                messages=messages,
                max_tokens=500,
                temperature=0.7
            )
            
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            return f"I apologize, but I'm having technical difficulties. Could you please share your experience with Excel formulas?"
    
    def evaluate_answer(self, question: str, answer: str, phase: str) -> Dict:
        """Evaluate candidate's answer and provide scoring"""
        try:
            evaluation_prompt = f"""
            As an Excel expert, evaluate this candidate's response to an interview question.
            
            Question: {question}
            Answer: {answer}
            Interview Phase: {phase}
            
            Provide evaluation in this exact JSON format:
            {{
                "accuracy_score": <0-10>,
                "depth_score": <0-10>,
                "clarity_score": <0-10>,
                "practical_application": <0-10>,
                "overall_score": <0-10>,
                "strengths": ["strength1", "strength2"],
                "areas_for_improvement": ["area1", "area2"],
                "skill_demonstrated": "<primary_skill>",
                "next_difficulty_suggestion": <1-10>,
                "brief_feedback": "<2-3 sentence evaluation>"
            }}
            
            Skills categories: formulas, data_analysis, pivot_tables, vba, problem_solving, communication
            """
            
            response = client.chat.completions.create(
                model="gpt-4",
                messages=[{"role": "user", "content": evaluation_prompt}],
                max_tokens=800,
                temperature=0.3
            )
            
            # Parse JSON response
            try:
                evaluation = json.loads(response.choices[0].message.content.strip())
                return evaluation
            except json.JSONDecodeError:
                # Fallback evaluation if JSON parsing fails
                return {
                    "accuracy_score": 5,
                    "depth_score": 5,
                    "clarity_score": 5,
                    "practical_application": 5,
                    "overall_score": 5,
                    "strengths": ["Provided a response"],
                    "areas_for_improvement": ["Could provide more detail"],
                    "skill_demonstrated": "communication",
                    "next_difficulty_suggestion": 5,
                    "brief_feedback": "Thank you for your response. Let's continue with the next question."
                }
                
        except Exception as e:
            st.error(f"Evaluation error: {str(e)}")
            return {
                "accuracy_score": 5, "depth_score": 5, "clarity_score": 5,
                "practical_application": 5, "overall_score": 5,
                "strengths": [], "areas_for_improvement": [],
                "skill_demonstrated": "communication", "next_difficulty_suggestion": 5,
                "brief_feedback": "Thank you for your response."
            }
    
    def generate_final_report(self, state: InterviewState) -> Dict:
        """Generate comprehensive interview report"""
        try:
            # Calculate overall metrics
            total_score = sum(state.skill_scores.values()) / len(state.skill_scores) if state.skill_scores else 0
            interview_duration = (datetime.now() - state.start_time).total_seconds() / 60
            
            report_prompt = f"""
            Generate a comprehensive Excel interview report based on the following data:
            
            Candidate: {state.candidate_name}
            Interview Duration: {interview_duration:.1f} minutes
            Questions Asked: {state.question_count}
            Skill Scores: {state.skill_scores}
            Overall Performance: {state.overall_performance}
            
            Conversation History: {json.dumps(state.conversation_history[-10:], indent=2)}
            
            Create a detailed report in JSON format:
            {{
                "overall_rating": <0-10>,
                "competency_level": "<Basic/Intermediate/Advanced/Expert>",
                "strengths": ["detailed strength analysis"],
                "areas_for_improvement": ["specific improvement areas"],
                "technical_skills_breakdown": {{
                    "formulas": <0-10>,
                    "data_analysis": <0-10>,
                    "pivot_tables": <0-10>,
                    "vba": <0-10>,
                    "problem_solving": <0-10>
                }},
                "hiring_recommendation": "<Strong Hire/Hire/No Hire>",
                "detailed_feedback": "<comprehensive paragraph>",
                "learning_recommendations": ["specific resources/topics"],
                "interview_highlights": ["key moments"],
                "next_steps": ["actionable recommendations"]
            }}
            """
            
            response = client.chat.completions.create(
                model="gpt-4",
                messages=[{"role": "user", "content": report_prompt}],
                max_tokens=1500,
                temperature=0.3
            )
            
            return json.loads(response.choices[0].message.content.strip())
            
        except Exception as e:
            st.error(f"Report generation error: {str(e)}")
            return {
                "overall_rating": total_score,
                "competency_level": "Intermediate",
                "strengths": ["Completed the interview"],
                "areas_for_improvement": ["Continue practicing Excel skills"],
                "hiring_recommendation": "Additional Assessment Needed",
                "detailed_feedback": "Thank you for participating in the interview.",
                "learning_recommendations": ["Practice Excel fundamentals"],
                "interview_highlights": ["Engaged throughout the process"],
                "next_steps": ["Continue skill development"]
            }

def initialize_session():
    """Initialize session state variables"""
    if 'interview_state' not in st.session_state:
        st.session_state.interview_state = None
    if 'interviewer' not in st.session_state:
        st.session_state.interviewer = ExcelInterviewer()
    if 'current_question' not in st.session_state:
        st.session_state.current_question = ""
    if 'awaiting_response' not in st.session_state:
        st.session_state.awaiting_response = False

def start_interview():
    """Start a new interview session"""
    candidate_name = st.session_state.get('candidate_name', 'Candidate')
    
    st.session_state.interview_state = InterviewState(
        session_id=str(uuid.uuid4()),
        candidate_name=candidate_name,
        current_phase="introduction",
        question_count=0,
        current_difficulty=3,
        conversation_history=[],
        skill_scores={skill: 0 for skill in st.session_state.interviewer.skills},
        overall_performance={},
        start_time=datetime.now()
    )
    
    # Generate first question
    first_question = st.session_state.interviewer.generate_question(st.session_state.interview_state)
    st.session_state.current_question = first_question
    st.session_state.awaiting_response = True
    
    # Add to conversation history
    st.session_state.interview_state.conversation_history.append({
        "role": "assistant",
        "content": first_question
    })

def process_answer(user_answer: str):
    """Process user's answer and generate next question"""
    if not st.session_state.interview_state or not st.session_state.awaiting_response:
        return
    
    state = st.session_state.interview_state
    
    # Add user answer to conversation history
    state.conversation_history.append({
        "role": "user", 
        "content": user_answer
    })
    
    # Evaluate the answer if not in introduction/conclusion
    if state.current_phase not in ["introduction", "conclusion"]:
        evaluation = st.session_state.interviewer.evaluate_answer(
            st.session_state.current_question, 
            user_answer, 
            state.current_phase
        )
        
        # Update skill scores
        skill = evaluation.get("skill_demonstrated", "communication")
        if skill in state.skill_scores:
            current_score = state.skill_scores[skill]
            new_score = evaluation.get("overall_score", 5)
            state.skill_scores[skill] = max(current_score, new_score)
        
        # Adjust difficulty
        next_diff = evaluation.get("next_difficulty_suggestion", state.current_difficulty)
        state.current_difficulty = max(1, min(10, next_diff))
    
    # Update question count and phase
    state.question_count += 1
    phase_questions = st.session_state.interviewer.max_questions_per_phase.get(state.current_phase, 5)
    
    # Check if should move to next phase
    current_phase_index = st.session_state.interviewer.phases.index(state.current_phase)
    if (state.question_count >= phase_questions and 
        current_phase_index < len(st.session_state.interviewer.phases) - 1):
        
        # Move to next phase
        state.current_phase = st.session_state.interviewer.phases[current_phase_index + 1]
        state.question_count = 0
    
    # Check if interview is complete
    if state.current_phase == "conclusion" and state.question_count >= 1:
        st.session_state.awaiting_response = False
        return
    
    # Generate next question
    next_question = st.session_state.interviewer.generate_question(state)
    st.session_state.current_question = next_question
    
    # Add to conversation history
    state.conversation_history.append({
        "role": "assistant",
        "content": next_question
    })

def display_progress():
    """Display interview progress"""
    if not st.session_state.interview_state:
        return
    
    state = st.session_state.interview_state
    phases = st.session_state.interviewer.phases
    current_index = phases.index(state.current_phase)
    
    # Progress bar
    progress = (current_index + 1) / len(phases)
    st.progress(progress)
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Current Phase", state.current_phase.replace("_", " ").title())
    with col2:
        st.metric("Questions Asked", len([msg for msg in state.conversation_history if msg["role"] == "user"]))
    with col3:
        duration = (datetime.now() - state.start_time).total_seconds() / 60
        st.metric("Duration", f"{duration:.1f} min")

def display_interview_report():
    """Display final interview report with visualizations"""
    if not st.session_state.interview_state:
        return
    
    state = st.session_state.interview_state
    
    with st.spinner("Generating comprehensive interview report..."):
        report = st.session_state.interviewer.generate_final_report(state)
    
    st.header("🎯 Interview Report")
    
    # Overall metrics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Overall Rating", f"{report.get('overall_rating', 0):.1f}/10")
    with col2:
        st.metric("Competency Level", report.get('competency_level', 'Intermediate'))
    with col3:
        st.metric("Recommendation", report.get('hiring_recommendation', 'Additional Assessment'))
    with col4:
        duration = (datetime.now() - state.start_time).total_seconds() / 60
        st.metric("Interview Duration", f"{duration:.1f} min")
    
    # Skills breakdown visualization
    skills_data = report.get('technical_skills_breakdown', {})
    if skills_data:
        st.subheader("📊 Technical Skills Breakdown")
        
        # Radar chart
        skills_df = pd.DataFrame({
            'Skill': list(skills_data.keys()),
            'Score': list(skills_data.values())
        })
        
        fig = go.Figure()
        fig.add_trace(go.Scatterpolar(
            r=skills_df['Score'],
            theta=skills_df['Skill'],
            fill='toself',
            name='Skills Assessment'
        ))
        fig.update_layout(
            polar=dict(
                radialaxis=dict(visible=True, range=[0, 10])
            ),
            showlegend=False,
            height=400
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Detailed feedback sections
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("💪 Strengths")
        for strength in report.get('strengths', []):
            st.write(f"• {strength}")
            
        st.subheader("🎯 Interview Highlights")
        for highlight in report.get('interview_highlights', []):
            st.write(f"• {highlight}")
    
    with col2:
        st.subheader("📈 Areas for Improvement")
        for area in report.get('areas_for_improvement', []):
            st.write(f"• {area}")
            
        st.subheader("📚 Learning Recommendations")
        for rec in report.get('learning_recommendations', []):
            st.write(f"• {rec}")
    
    # Detailed feedback
    st.subheader("📝 Detailed Feedback")
    st.write(report.get('detailed_feedback', 'No detailed feedback available.'))
    
    # Next steps
    st.subheader("🚀 Next Steps")
    for step in report.get('next_steps', []):
        st.write(f"• {step}")
    
    # Download report option
    if st.button("📥 Download Report as JSON"):
        report_data = {
            "candidate_name": state.candidate_name,
            "interview_date": state.start_time.isoformat(),
            "session_id": state.session_id,
            **report
        }
        st.download_button(
            label="Download JSON Report",
            data=json.dumps(report_data, indent=2),
            file_name=f"excel_interview_report_{state.candidate_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
            mime="application/json"
        )

def main():
    """Main application interface"""
    initialize_session()
    
    st.title("🤖 AI-Powered Excel Mock Interviewer")
    st.markdown("### Professional Excel Skills Assessment Platform")
    
    # Sidebar for configuration
    with st.sidebar:
        st.header("Interview Configuration")
        if not st.session_state.interview_state:
            candidate_name = st.text_input("Candidate Name", value="", key="candidate_name")
            
            st.markdown("### About This Interview")
            st.info("""
            **Duration:** 30-40 minutes
            **Focus:** Practical Excel skills
            **Format:** Progressive difficulty
            **Coverage:** Formulas, Data Analysis, Pivot Tables, VBA
            """)
            
            if st.button("🚀 Start Interview", disabled=not candidate_name.strip()):
                start_interview()
        else:
            st.success(f"Interview in progress: {st.session_state.interview_state.candidate_name}")
            display_progress()
            
            if st.button("🔄 Restart Interview"):
                for key in ['interview_state', 'current_question', 'awaiting_response']:
                    if key in st.session_state:
                        del st.session_state[key]
                st.rerun()
    
    # Main interview interface
    if not st.session_state.interview_state:
        # Welcome screen
        st.markdown("""
        ## Welcome to the AI Excel Mock Interviewer! 👋
        
        This advanced AI system will assess your Microsoft Excel skills through a structured interview process.
        
        ### What to Expect:
        - **Personalized Questions**: Adaptive difficulty based on your responses
        - **Comprehensive Coverage**: From basic formulas to advanced analytics
        - **Real-time Feedback**: Immediate evaluation and guidance
        - **Detailed Report**: Complete assessment with improvement recommendations
        
        ### How It Works:
        1. **Introduction**: We'll start with basic questions about your Excel experience
        2. **Warm-up**: Simple questions to get you comfortable
        3. **Core Assessment**: Main evaluation covering key Excel skills
        4. **Deep Dive**: Advanced topics based on your expertise level
        5. **Conclusion**: Final summary and next steps
        
        **Ready to begin?** Enter your name in the sidebar and click "Start Interview"
        """)
        
        # Sample questions preview
        with st.expander("📋 Sample Questions Preview"):
            st.markdown("""
            **Basic Level:**
            - How would you calculate the sum of values in cells A1 to A10?
            - What's the difference between relative and absolute cell references?
            
            **Intermediate Level:**
            - How would you use VLOOKUP to find data across multiple sheets?
            - Explain how to create a pivot table for sales data analysis
            
            **Advanced Level:**
            - How would you automate repetitive tasks using VBA?
            - Describe your approach to data modeling for financial analysis
            """)
    
    elif st.session_state.awaiting_response:
        # Active interview interface
        st.markdown(f"### 💬 Interview with {st.session_state.interview_state.candidate_name}")
        
        # Display current question
        st.markdown("** AI Interviewer:**")
        st.info(st.session_state.current_question)
        
        # Answer input
        st.markdown("**Your Response:**")
        user_answer = st.text_area(
            "Please provide your answer:",
            height=150,
            placeholder="Type your detailed response here..."
        )
        
        col1, col2 = st.columns([1, 4])
        with col1:
            if st.button("📤 Submit Answer", disabled=not user_answer.strip()):
                process_answer(user_answer)
                st.rerun()
        
        # Conversation history
        if st.session_state.interview_state.conversation_history:
            with st.expander("📜 Conversation History"):
                for i, msg in enumerate(st.session_state.interview_state.conversation_history[:-1]):
                    role = "🤖 Interviewer" if msg["role"] == "assistant" else "👤 You"
                    st.markdown(f"**{role}:** {msg['content']}")
                    if i < len(st.session_state.interview_state.conversation_history) - 2:
                        st.markdown("---")
    
    else:
        # Interview completed - show report
        display_interview_report()
        
        # Option to start new interview
        st.markdown("---")
        if st.button("🆕 Start New Interview"):
            for key in ['interview_state', 'current_question', 'awaiting_response']:
                if key in st.session_state:
                    del st.session_state[key]
            st.rerun()
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666;'>
        <p>🔒 AI-Powered Excel Mock Interviewer | Built with Streamlit & OpenAI GPT-4</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()