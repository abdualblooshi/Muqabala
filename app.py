# Fix torch path issue with Streamlit
import os
import torch
torch.classes.__path__ = []  # Fix for torch path warning in Streamlit

import streamlit as st
import streamlit.runtime.scriptrunner.script_runner as script_runner

# Must be the first Streamlit command
st.set_page_config(
    page_title="Muqabala - AI Interviewer",
    layout="wide",
    initial_sidebar_state="expanded"
)

import requests
import json
from datetime import datetime
import os
from dotenv import load_dotenv, find_dotenv
import torch
import time
import threading
import queue
import logging

# Import debug utilities
from interview_modules.debug_utils import (
    debug_manager,
    debug_wrapper,
    StreamlitContextManager,
    test_streamlit_context
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import our custom modules
from interview_modules.speech.speech_handler import SpeechHandler
from interview_modules.evaluation.evaluator import InterviewEvaluator
from interview_modules.report_gen.report_generator import ReportGenerator

# Load environment variables
load_dotenv(find_dotenv())

# Configuration
OLLAMA_BASE_URL = os.getenv("OLLAMA_API_URL", "http://localhost:11434")
OLLAMA_API_URL = f"{OLLAMA_BASE_URL}/api/generate"
MODEL = os.getenv("MODEL", "deepseek-r1:7b")

# Cache initialization of components
@st.cache_resource
def init_speech_handler():
    return SpeechHandler(ollama_url=OLLAMA_API_URL, model=MODEL)

@st.cache_resource
def init_evaluator():
    return InterviewEvaluator()

@st.cache_resource
def init_report_generator():
    return ReportGenerator()

# Initialize components with caching
speech_handler = init_speech_handler()
evaluator = init_evaluator()
report_generator = init_report_generator()

# Dark mode CSS
st.markdown("""
    <style>
        /* Dark mode styles */
        .stApp {
            background-color: #1E1E1E;
            color: #FFFFFF;
        }
        .main { padding: 2rem; }
        .stButton>button { 
            background-color: #00AAFF;
            color: white;
            border-radius: 20px;
            padding: 0.5rem 2rem;
            border: none;
        }
        .interview-section {
            background-color: #2D2D2D;
            padding: 2rem;
            border-radius: 10px;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.2);
        }
        .metrics-section {
            background-color: #363636;
            padding: 1rem;
            border-radius: 10px;
            margin-top: 1rem;
        }
        .stTextInput>div>div>input {
            background-color: #363636;
            color: white;
        }
        .stSelectbox>div>div>div {
            background-color: #363636;
            color: white;
        }
        .speech-recognition {
            padding: 1rem;
            background-color: #363636;
            border-radius: 10px;
            margin: 1rem 0;
        }
        .recognition-status {
            color: #00AAFF;
            font-style: italic;
        }
        /* Progress bar colors */
        .stProgress > div > div > div {
            background-color: #00AAFF;
        }
        /* Chat message styles */
        .chat-message {
            padding: 1.5rem;
            border-radius: 15px;
            margin-bottom: 1rem;
            position: relative;
            width: 80%;
        }
        .ai-message {
            background-color: #2D2D2D;
            margin-left: 0;
            margin-right: auto;
        }
        .user-message {
            background-color: #00AAFF;
            margin-left: auto;
            margin-right: 0;
        }
        .message-content {
            margin: 0;
            padding: 0;
        }
        .confirmation-box {
            background-color: #363636;
            padding: 1rem;
            border-radius: 10px;
            margin: 1rem 0;
        }
    </style>
""", unsafe_allow_html=True)

# Session state initialization
if "interview_started" not in st.session_state:
    st.session_state.interview_started = False
if "current_question_index" not in st.session_state:
    st.session_state.current_question_index = 0
if "questions_and_answers" not in st.session_state:
    st.session_state.questions_and_answers = []
if "language" not in st.session_state:
    st.session_state.language = "en"
if "candidate_info" not in st.session_state:
    st.session_state.candidate_info = {
        "name": "",
        "position": "",
        "date": datetime.now()
    }
if "recording" not in st.session_state:
    st.session_state.recording = False
if "recording_stop" not in st.session_state:
    st.session_state.recording_stop = False
if "recording_thread" not in st.session_state:
    st.session_state.recording_thread = None
if "recognition_text" not in st.session_state:
    st.session_state.recognition_text = ""
if "refined_text" not in st.session_state:
    st.session_state.refined_text = ""
if "live_transcript" not in st.session_state:
    st.session_state.live_transcript = ""
if "last_transcript_update" not in st.session_state:
    st.session_state.last_transcript_update = 0
if "answers_history" not in st.session_state:
    st.session_state.answers_history = {}
if "conversation_history" not in st.session_state:
    st.session_state.conversation_history = []
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "pending_confirmation" not in st.session_state:
    st.session_state.pending_confirmation = None

# Function to safely stop recording thread
def stop_recording_thread():
    """Safely stop the recording thread and clean up."""
    try:
        if st.session_state.get('recording_thread'):
            st.session_state.recording_stop = True
            st.session_state.recording = False
            if st.session_state.recording_thread.is_alive():
                st.session_state.recording_thread.join(timeout=1.0)
            st.session_state.recording_thread = None
    except Exception as e:
        logger.error(f"Error stopping recording thread: {e}")
    finally:
        st.session_state.recording = False
        st.session_state.recording_stop = True
        st.session_state.recording_thread = None

# Function to safely start recording thread
def start_recording_thread(target):
    """Safely start a new recording thread."""
    try:
        stop_recording_thread()  # Clean up any existing thread
        st.session_state.recording_stop = False
        st.session_state.recording = True
        
        # Capture current Streamlit context
        ctx = script_runner.get_script_run_ctx()
        
        # Wrapper to ensure context in thread
        def run_with_context():
            if ctx:
                script_runner.add_script_run_ctx(ctx)
            target()
        
        # Create and start the thread
        recording_thread = threading.Thread(target=run_with_context)
        recording_thread.daemon = True
        recording_thread.start()
        
        # Store thread reference
        st.session_state.recording_thread = recording_thread
        
        return True
    except Exception as e:
        logger.error(f"Error starting recording thread: {e}")
        st.session_state.recording = False
        st.session_state.recording_stop = True
        st.session_state.recording_thread = None
        return False

# Cache the questions
@st.cache_data
def generate_interview_questions(position: str, language: str = "en") -> list:
    """Generate role-specific interview questions using DeepSeek."""
    try:
        prompt = f'''You are an expert interviewer for {position} positions. Generate 5 interview questions that will help assess the candidate's suitability for this role.

The questions should be a mix of technical and behavioral questions, focusing on:
1. Role-specific technical knowledge
2. Past experience and achievements
3. Problem-solving abilities
4. Soft skills and communication
5. Cultural fit and future goals

Format your response as a valid Python list of strings containing ONLY the questions.
Example: ["Question 1", "Question 2", "Question 3", "Question 4", "Question 5"]'''
        
        response = requests.post(
            OLLAMA_API_URL,
            json={
                "model": MODEL,
                "prompt": prompt,
                "stream": False
            }
        )
        
        if response.status_code == 200:
            # Extract just the list from the response
            response_text = response.json().get("response", "").strip()
            # Find the list portion using regex
            import re
            match = re.search(r'\[.*\]', response_text, re.DOTALL)
            if match:
                questions = eval(match.group())
                
                # Translate questions if needed
                if language == "ar":
                    translator = Translator()
                    questions = [translator.translate(q, dest='ar').text for q in questions]
                
                return questions
            else:
                return get_default_questions(language)
        
        return get_default_questions(language)
        
    except Exception as e:
        st.error(f"Error generating questions: {e}")
        return get_default_questions(language)

def get_default_questions(language: str) -> list:
    """Fallback default questions."""
    if language == "en":
        return [
            "Tell me about your background and experience.",
            "What are your key technical skills?",
            "Describe a challenging project you worked on.",
            "How do you handle difficult situations at work?",
            "Where do you see yourself in 5 years?"
        ]
    else:
        return [
            "حدثني عن خلفيتك وخبرتك.",
            "ما هي مهاراتك التقنية الرئيسية؟",
            "صف مشروعًا صعبًا عملت عليه.",
            "كيف تتعامل مع المواقف الصعبة في العمل؟",
            "أين ترى نفسك بعد 5 سنوات؟"
        ]

def get_ai_response(answer: str, context: str = "", candidate_name: str = "", position: str = "", conversation_history: list = None) -> str:
    """Get interactive AI response using DeepSeek."""
    try:
        # Build conversation context
        conversation_context = ""
        if conversation_history:
            last_exchanges = conversation_history[-4:]  # Get last 2 exchanges
            for entry in last_exchanges:
                if entry["role"] == "candidate":
                    conversation_context += f"Candidate: {entry['content']}\n"
                else:
                    conversation_context += f"Interviewer: {entry['content']}\n"

        prompt = f"""You are an advanced AI interviewer named Muqabala conducting a job interview with {candidate_name} for a {position} position.
        Act as a professional but warm interviewer, showing genuine interest and engagement. Your responses should feel natural and conversational.
        
        Previous conversation:
        {conversation_context}
        
        Current question: {context}
        Candidate's answer: {answer}
        
        Respond naturally as a skilled interviewer would, including:
        1. A thoughtful acknowledgment that shows you understood their response
        2. A brief insight or connection to demonstrate your expertise about the {position} role
        3. An engaging follow-up question or comment that encourages deeper discussion
        4. If they mention technical skills or experiences, validate them and explore further
        
        Keep your tone professional but friendly, as if having a real conversation. Use verbal acknowledgments like "I see", "That's interesting", 
        or "Tell me more about" to make the conversation flow naturally. Show genuine interest in their responses.
        
        Response:"""
        
        response = requests.post(
            OLLAMA_API_URL,
            json={
                "model": MODEL,
                "prompt": prompt,
                "stream": False
            }
        )
        
        if response.status_code == 200:
            return response.json().get("response", "").strip()
        return f"I see, {candidate_name}. That's interesting. Could you elaborate more on your experience?"
    except Exception as e:
        st.error(f"Error getting AI response: {e}")
        return f"I understand, {candidate_name}. Please tell me more about that."

def display_answer_history(question_index: int):
    """Display the answer history for a specific question."""
    if question_index in st.session_state.answers_history:
        answer_data = st.session_state.answers_history[question_index]
        with st.expander("Previous Response", expanded=True):
            st.markdown("**Your Answer:**")
            st.write(answer_data["answer"])
            
            st.markdown("**AI Interviewer Response:**")
            st.write(answer_data["ai_response"])

def generate_evaluation_criteria(position: str) -> dict:
    """Generate role-specific evaluation criteria using DeepSeek."""
    try:
        prompt = f'''As an expert interviewer, create evaluation criteria for a {position} position.
Please provide a set of evaluation criteria with their respective weights.
The weights should be decimal numbers that sum to 1.0.

Format your response as a valid Python dictionary with criteria and weights.
Example format: {{"technical_knowledge": 0.3, "communication": 0.2, "experience": 0.3, "problem_solving": 0.2}}

Requirements:
1. Use only decimal numbers (0.0 to 1.0) for weights
2. Weights must sum to 1.0
3. Include 4-6 relevant criteria for the {position} role
4. Use underscores for multi-word criteria names
5. Return ONLY the dictionary, no other text'''
        
        response = requests.post(
            OLLAMA_API_URL,
            json={
                "model": MODEL,
                "prompt": prompt,
                "stream": False
            }
        )
        
        if response.status_code == 200:
            # Extract just the dictionary from the response
            response_text = response.json().get("response", "").strip()
            # Find the dictionary portion using regex
            import re
            match = re.search(r'\{[^}]+\}', response_text)
            if match:
                criteria_dict = eval(match.group())
                # Validate the dictionary
                if isinstance(criteria_dict, dict):
                    # Convert all values to float
                    criteria_dict = {k: float(v) for k, v in criteria_dict.items()}
                    # Check if weights sum to approximately 1.0 (allowing for small floating-point errors)
                    total = sum(criteria_dict.values())
                    if abs(total - 1.0) > 0.01:  # Allow 1% margin of error
                        # Normalize weights to sum to 1.0
                        criteria_dict = {k: v/total for k, v in criteria_dict.items()}
                    return criteria_dict
            
            # Return default if parsing fails
            return {
                "technical_knowledge": 0.3,
                "leadership_skills": 0.2,
                "strategic_thinking": 0.3,
                "innovation_focus": 0.2
            }
        
        return {
            "technical_knowledge": 0.3,
            "leadership_skills": 0.2,
            "strategic_thinking": 0.3,
            "innovation_focus": 0.2
        }
    except Exception as e:
        logger.error(f"Error generating evaluation criteria: {e}")
        return {
            "technical_knowledge": 0.3,
            "leadership_skills": 0.2,
            "strategic_thinking": 0.3,
            "innovation_focus": 0.2
        }

def get_interview_context(position: str) -> str:
    """Generate interview context and background using DeepSeek."""
    try:
        prompt = f'''Create a structured interview context for a {position} position.
Format the response in clear sections with bullet points.

Required sections and guidelines:
1. Key Responsibilities (3-4 bullet points)
   - Focus on main duties and leadership aspects
   - Include strategic and operational responsibilities
   - Highlight team and resource management

2. Required Technical Skills (3-4 bullet points)
   - List specific technical competencies
   - Include relevant tools and technologies
   - Mention industry-specific technical requirements

3. Important Soft Skills (3-4 bullet points)
   - Focus on leadership and communication
   - Include team management abilities
   - Highlight strategic thinking capabilities

4. Industry Knowledge (2-3 bullet points)
   - Specify required industry expertise
   - Include current trends awareness
   - Mention regulatory knowledge if applicable

Format the response with clear section headers and bullet points.
Return ONLY the formatted text, no additional commentary.
Keep each bullet point concise and directly relevant to the {position} role.'''
        
        response = requests.post(
            OLLAMA_API_URL,
            json={
                "model": MODEL,
                "prompt": prompt,
                "stream": False
            }
        )
        
        if response.status_code == 200:
            context = response.json().get("response", "").strip()
            # Clean up the response
            import re
            # Remove any "think" or similar sections
            context = re.sub(r'<think>.*?</think>', '', context, flags=re.DOTALL)
            # Remove any extra newlines
            context = re.sub(r'\n{3,}', '\n\n', context)
            # Ensure proper section formatting
            context = re.sub(r'^(\d+)\.\s+', r'### \1. ', context, flags=re.MULTILINE)
            return context
            
        return f"""### Interview Context for {position}

### 1. Key Responsibilities
- Lead technical strategy and innovation initiatives
- Manage and mentor technical teams
- Ensure delivery of high-quality solutions

### 2. Required Technical Skills
- Strong programming and architecture knowledge
- Experience with modern development practices
- Expertise in relevant technologies

### 3. Important Soft Skills
- Excellent leadership and communication
- Strategic thinking and problem-solving
- Team management and motivation

### 4. Industry Knowledge
- Current technology trends awareness
- Understanding of business domain
- Knowledge of best practices"""
    except Exception as e:
        logger.error(f"Error generating interview context: {e}")
        return f"Preparing to interview for the {position} position..."

def display_chat_message(text: str, is_user: bool):
    """Display a chat message with appropriate styling."""
    message_class = "user-message" if is_user else "ai-message"
    st.markdown(f"""
        <div class="chat-message {message_class}">
            <p class="message-content">{"You: " if is_user else "AI: "}{text}</p>
        </div>
    """, unsafe_allow_html=True)

# Main interview interface
if st.session_state.interview_started:
    st.title("🤖 Muqabala AI Interviewer")
    
    # Generate interview context if not exists
    if "interview_context" not in st.session_state:
        st.session_state.interview_context = get_interview_context(st.session_state.candidate_info["position"])
        st.session_state.evaluation_criteria = generate_evaluation_criteria(st.session_state.candidate_info["position"])
    
    # Display interview context
    with st.expander("Interview Context", expanded=False):
        st.markdown(f"**Position:** {st.session_state.candidate_info['position']}")
        st.write(st.session_state.interview_context)
        st.markdown("**Key Evaluation Criteria:**")
        if isinstance(st.session_state.evaluation_criteria, dict):
            for criterion, weight in st.session_state.evaluation_criteria.items():
                try:
                    weight_float = float(weight)
                    st.write(f"- {criterion.replace('_', ' ').title()}: {weight_float*100:.0f}%")
                except (TypeError, ValueError):
                    st.write(f"- {criterion.replace('_', ' ').title()}: {weight}")
    
    # Generate role-specific questions if not already generated
    if "questions" not in st.session_state:
        with st.spinner("Preparing interview questions..."):
            st.session_state.questions = generate_interview_questions(
                st.session_state.candidate_info["position"],
                st.session_state.language
            )
    
    # Display chat history
    st.markdown("### Interview Chat")
    chat_container = st.container()
    with chat_container:
        # Display all previous messages
        for message in st.session_state.chat_history:
            display_chat_message(message["content"], message["role"] == "user")
        
        # Display current question if no messages yet
        if not st.session_state.chat_history:
            current_q = st.session_state.questions[0]
            display_chat_message(current_q, False)
    
    # Input section
    st.markdown("### Your Response")
    col1, col2 = st.columns([3, 1])
    
    with col1:
        # Text input for typing responses
        user_input = st.text_input("Type your response", key="user_input")
    
    with col2:
        # Record button
        if st.button("🎤 Record Response"):
            with st.spinner("Recording..."):
                # Record audio for 10 seconds
                success, text = speech_handler.record_and_transcribe(
                    language="en-US" if st.session_state.language == "en" else "ar-SA",
                    duration=10
                )
                if success:
                    st.session_state.user_input = text
                    st.experimental_rerun()
                else:
                    st.error("Failed to record audio. Please try typing your response.")
    
    # Send button
    if st.button("Send") and (user_input or st.session_state.get("user_input")):
        response_text = user_input or st.session_state.get("user_input", "")
        
        # Add user message to chat history
        st.session_state.chat_history.append({
            "role": "user",
            "content": response_text
        })
        
        # Get current question context
        current_q = st.session_state.questions[len(st.session_state.chat_history) // 2]
        
        # Get AI response
        with st.spinner("Processing response..."):
            ai_response = get_ai_response(
                response_text,
                current_q,
                st.session_state.candidate_info["name"],
                st.session_state.candidate_info["position"],
                st.session_state.chat_history
            )
        
        # Add AI response to chat history
        st.session_state.chat_history.append({
            "role": "ai",
            "content": ai_response
        })
        
        # Clear input
        st.session_state.user_input = ""
        
        # If we've gone through all questions, show the final evaluation
        if len(st.session_state.chat_history) >= len(st.session_state.questions) * 2:
            st.success("Interview completed! Click 'End Interview' to generate the report.")
        
        st.experimental_rerun()
    
    # Display interview progress
    progress = min((len(st.session_state.chat_history) + 1) // 2 / len(st.session_state.questions), 1.0)
    st.progress(progress)
    st.write(f"Question {(len(st.session_state.chat_history) // 2) + 1} of {len(st.session_state.questions)}")

else:
    # Welcome screen
    st.title("👋 Welcome to Muqabala")
    st.markdown("""
        ### AI-Powered Interview Assistant
        
        Muqabala helps conduct and evaluate job interviews with:
        - 🗣️ Speech recognition and synthesis
        - 🌐 Support for English and Arabic
        - 📊 Real-time evaluation of responses
        - 📝 Detailed PDF report generation
        
        To begin, please enter the candidate information in the sidebar.
    """)

def reset_interview():
    """Reset the interview state and return to the welcome screen."""
    for key in list(st.session_state.keys()):
        if key not in ["speech_handler", "evaluator", "report_generator"]:
            del st.session_state[key]

# Sidebar
with st.sidebar:
    st.title("👔 Interview Settings")
    
    if not st.session_state.interview_started:
        st.session_state.candidate_info["name"] = st.text_input(
            "Candidate Name",
            key="candidate_name"
        )
        st.session_state.candidate_info["position"] = st.text_input(
            "Position",
            key="position"
        )
        st.session_state.language = st.selectbox(
            "Interview Language",
            ["en", "ar"],
            format_func=lambda x: "English" if x == "en" else "Arabic"
        )
        
        # Add interview style options
        st.markdown("### Interview Style")
        interview_style = st.select_slider(
            "Conversation Style",
            options=["Formal", "Balanced", "Casual"],
            value="Balanced"
        )
        
        if st.button("Start Interview"):
            if st.session_state.candidate_info["name"] and st.session_state.candidate_info["position"]:
                position = st.session_state.candidate_info["position"]
                
                # Create a progress bar
                progress_bar = st.progress(0)
                
                # Initialize interview with loading states
                with st.spinner(f"🤖 Initializing AI interview for {position} role..."):
                    progress_bar.progress(20)
                    st.session_state.interview_started = True
                    st.session_state.interview_style = interview_style
                
                # Generate context with loading state
                with st.spinner("📝 Creating interview context..."):
                    progress_bar.progress(40)
                    st.session_state.interview_context = get_interview_context(position)
                
                # Generate evaluation criteria with loading state
                with st.spinner("⚖️ Defining evaluation criteria..."):
                    progress_bar.progress(60)
                    st.session_state.evaluation_criteria = generate_evaluation_criteria(position)
                
                # Generate questions with loading state
                with st.spinner("🎯 Preparing role-specific questions..."):
                    progress_bar.progress(80)
                    st.session_state.questions = generate_interview_questions(
                        position,
                        st.session_state.language
                    )
                
                progress_bar.progress(100)
                st.success("✨ Interview setup complete!")
                time.sleep(1)  # Give users time to see the success message
                st.rerun()
            else:
                st.error("Please fill in all candidate information.")
    
    else:
        st.info(f"Interviewing: {st.session_state.candidate_info['name']}")
        st.info(f"Position: {st.session_state.candidate_info['position']}")
        st.info(f"Language: {'English' if st.session_state.language == 'en' else 'Arabic'}")
        
        # Add interview controls
        st.markdown("### Interview Controls")
        voice_speed = st.slider("AI Voice Speed", min_value=100, max_value=200, value=150, step=10)
        voice_volume = st.slider("AI Voice Volume", min_value=0.0, max_value=1.0, value=0.9, step=0.1)
        
        if st.button("End Interview"):
            # Generate report
            output_path = f"reports/{st.session_state.candidate_info['name'].replace(' ', '_')}_report.pdf"
            os.makedirs("reports", exist_ok=True)
            
            if hasattr(st.session_state, 'final_evaluation'):
                report_generator.generate_report(
                    candidate_name=st.session_state.candidate_info["name"],
                    position=st.session_state.candidate_info["position"],
                    interview_date=st.session_state.candidate_info["date"],
                    questions_and_answers=st.session_state.questions_and_answers,
                    evaluation_metrics=st.session_state.final_evaluation,
                    output_path=output_path,
                    language=st.session_state.language
                )
                st.success(f"Report generated: {output_path}")
            
            # Reset interview
            reset_interview()
            st.rerun()

    # Add debug controls to sidebar
    if st.session_state.interview_started:
        st.markdown("### 🔧 Debug Controls")
        debug_expander = st.expander("Debug Information")
        with debug_expander:
            if st.checkbox("Enable Debug Mode", value=debug_manager.is_debug_mode):
                debug_manager.start_debug_mode()
            else:
                debug_manager.stop_debug_mode()
            
            if st.button("Test Streamlit Context"):
                if test_streamlit_context():
                    st.success("Streamlit context test passed")
                else:
                    st.error("Streamlit context test failed")
            
            if st.button("Show System Info"):
                system_info = debug_manager.get_system_info()
                st.json(system_info)
            
            if st.button("Show Thread Status"):
                thread_status = debug_manager.get_thread_status()
                st.json(thread_status)
