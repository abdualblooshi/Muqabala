# Fix torch path issue with Streamlit
import os
import torch
torch.classes.__path__ = []  # Fix for torch path warning in Streamlit

import streamlit as st
import streamlit.runtime.scriptrunner.script_runner as script_runner
import re  # Add import for regular expressions

# Must be the first Streamlit command
st.set_page_config(
    page_title="Muqabala - AI Interviewer",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Add Tailwind CSS CDN and custom styles
st.markdown("""
    <link href="https://cdn.tailwindcss.com" rel="stylesheet">
    <script src="https://cdn.tailwindcss.com"></script>
    <style>
        /* Main container styles */
        .st-emotion-cache-1vvsst3 {
            display: flex;
            flex-direction: column;
            padding: 20px;
            height: calc(100vh - 400px);
            min-height: 400px;
            overflow-y: auto;
            overflow-x: hidden;
            margin-bottom: 180px; /* Space for fixed input bar */
        }
        
        /* Message container */
        .chat-message {
            display: flex;
            align-items: flex-start;
            margin-bottom: 15px;
            animation: fadeIn 0.5s ease-in-out;
            width: 100%;
            padding: 0 10px;
        }
        
        /* User message specific styles */
        .chat-message.user {
            flex-direction: row-reverse;
        }
        
        /* Message content wrapper */
        .message-wrapper {
            display: flex;
            align-items: flex-start;
            max-width: 70%;
            gap: 12px;
        }
        
        .user .message-wrapper {
            flex-direction: row-reverse;
            margin-left: auto;
        }
        
        /* Avatar styles */
        .message-avatar {
            width: 40px;
            height: 40px;
            border-radius: 50%;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 20px;
            flex-shrink: 0;
        }
        
        .user .message-avatar {
            background: linear-gradient(135deg, #6366f1, #4f46e5);
            color: white;
        }
        
        .ai .message-avatar {
            background: linear-gradient(135deg, #10b981, #059669);
            color: white;
        }
        
        /* Message content */
        .message-content {
            padding: 12px 16px;
            border-radius: 18px;
            line-height: 1.5;
            position: relative;
            font-size: 15px;
            max-width: 100%;
            word-wrap: break-word;
        }
        
        .user .message-content {
            background: linear-gradient(135deg, #6366f1, #4f46e5);
            color: white;
            border-bottom-right-radius: 5px;
        }
        
        .ai .message-content {
            background: linear-gradient(135deg, #10b981, #059669);
            color: white;
            border-bottom-left-radius: 5px;
        }
        
        /* Input area styles */
        .input-area {
            position: fixed;
            bottom: 0;
            left: 0;
            right: 0;
            background: #1a1c1e;
            padding: 20px;
            border-top: 1px solid rgba(255, 255, 255, 0.1);
            z-index: 1000;
            display: flex;
            flex-direction: column;
            gap: 10px;
        }
        
        /* Adjust input container width */
        .input-area .stTextInput {
            width: 100%;
        }
        
        /* Style the send button container */
        .input-area .stButton {
            width: 100%;
        }
        
        /* Recording button styles */
        .recording-button {
            width: 100%;
            background: linear-gradient(135deg, #6366f1, #4f46e5);
            color: white;
            padding: 10px 20px;
            border-radius: 25px;
            border: none;
            cursor: pointer;
            display: flex;
            align-items: center;
            justify-content: center;
            transition: all 0.3s ease;
        }
        
        .recording-button:hover {
            transform: translateY(-2px);
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        }
        
        .recording-indicator {
            width: 10px;
            height: 10px;
            background: #ef4444;
            border-radius: 50%;
            margin-right: 8px;
            display: none;
        }
        
        .recording .recording-indicator {
            display: block;
            animation: pulse 1.5s infinite;
        }
        
        /* Animations */
        @keyframes pulse {
            0% { transform: scale(1); opacity: 1; }
            50% { transform: scale(1.5); opacity: 0.5; }
            100% { transform: scale(1); opacity: 1; }
        }
        
        @keyframes fadeIn {
            from { opacity: 0; transform: translateY(10px); }
            to { opacity: 1; transform: translateY(0); }
        }
        
        /* Custom scrollbar */
        .st-emotion-cache-1vvsst3::-webkit-scrollbar {
            width: 8px;
        }
        
        .st-emotion-cache-1vvsst3::-webkit-scrollbar-track {
            border-radius: 4px;
        }
        
        .st-emotion-cache-1vvsst3::-webkit-scrollbar-thumb {
            background: #888;
            border-radius: 4px;
        }
        
        .st-emotion-cache-1vvsst3::-webkit-scrollbar-thumb:hover {
            background: #555;
        }

        /* Streamlit overrides */
        .stTextInput > div > div > input {
            border: 2px solid #e5e7eb;
            border-radius: 25px !important;
            padding: 12px 20px !important;
            font-size: 15px !important;
            transition: all 0.3s ease;
        }
        
        .stTextInput > div > div > input:focus {
            border-color: #6366f1 !important;
            box-shadow: 0 0 0 3px rgba(99, 102, 241, 0.1) !important;
        }
    </style>
    <link href="static/css/dialog.css" rel="stylesheet">
    <script src="static/js/recording.js"></script>
    <script src="static/js/dialog.js"></script>
""", unsafe_allow_html=True)

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
import pandas as pd

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
MODEL = os.getenv("MODEL", "deepseek-r1:1.5b")

def get_ai_response(answer: str, context: str, candidate_name: str, position: str, conversation_history: list) -> str:
    """Get interactive AI response using DeepSeek."""
    try:
        # Initialize conversation memory if not exists
        if 'conversation_memory' not in st.session_state:
            st.session_state.conversation_memory = {
                'topics_discussed': set(),
                'claims_made': {},
                'response_history': [],
                'sentiment_history': [],
                'last_questions': []
            }

        # Check for interview completion phrases
        completion_phrases = ["i'm done", "i think i'm done", "i am done", "that's all", "that is all", 
                            "no i think this is enough", "this is enough", "i think this is enough"]
        if any(phrase in answer.lower() for phrase in completion_phrases):
            st.session_state.interview_completed = True
            st.session_state.questions_and_answers = [
                {"question": st.session_state.questions[i], "answer": conversation_history[i*2]["content"]}
                for i in range(len(conversation_history)//2)
            ]
            return "Thank you for letting me know. I'll now generate your evaluation report based on our discussion."

        # Build conversation context with proper formatting
        conversation_context = ""
        if conversation_history:
            # Get last 3 exchanges for context
            last_exchanges = conversation_history[-6:]  # Get last 3 exchanges (6 messages)
            for entry in last_exchanges:
                if entry["role"] == "user":
                    conversation_context += f"[Candidate]: {entry['content']}\n\n"
                else:
                    conversation_context += f"[Interviewer]: {entry['content']}\n\n"

        # Check if we should end the interview
        exchange_count = len(conversation_history) // 2
        max_exchanges = len(st.session_state.questions)
        
        if exchange_count >= max_exchanges:
            st.session_state.interview_completed = True
            st.session_state.questions_and_answers = [
                {"question": st.session_state.questions[i], "answer": conversation_history[i*2]["content"]}
                for i in range(max_exchanges)
            ]
            return "Thank you for your time. The interview is now complete. I will generate your evaluation report."

        # Enhanced response analysis with intent and sentiment
        analysis_prompt = f"""Analyze the following candidate response for a {position} position:

Response: {answer}

Previous topics discussed: {', '.join(st.session_state.conversation_memory['topics_discussed'])}
Previous claims made: {json.dumps(st.session_state.conversation_memory['claims_made'], indent=2)}

Provide a detailed analysis in JSON format:
{{
    "intent": {{
        "primary": str,  // main intent (e.g., "demonstrate_experience", "explain_skills", "deflect")
        "secondary": list,  // secondary intents
        "confidence": float  // 0-1 scale
    }},
    "sentiment": {{
        "overall": str,  // positive, negative, or neutral
        "confidence": float,  // 0-1 scale
        "specific_emotions": list  // detected emotions
    }},
    "content_analysis": {{
        "key_points": list,
        "new_topics_introduced": list,
        "claims_made": list,
        "potential_concerns": list,
        "credibility_score": float  // 0-1 scale
    }},
    "follow_up": {{
        "recommended_topics": list,
        "verification_needed": list,
        "clarification_needed": list
    }}
}}"""

        # Get DeepSeek analysis with improved prompt
        print("\n----------------------------------------")
        print("SENDING PROMPT TO DEEPSEEK:")
        print(analysis_prompt)
        print("----------------------------------------\n")
        
        # Add retry logic for API calls
        max_retries = 3
        retry_delay = 2  # seconds
        
        for attempt in range(max_retries):
            try:
                response = requests.post(
                    OLLAMA_API_URL,
                    json={
                        "model": MODEL,
                        "prompt": analysis_prompt,
                        "stream": False
                    },
                    timeout=30
                )
                
                print(f"\nAttempt {attempt + 1}: DeepSeek API response status code: {response.status_code}")
                
                if response.status_code == 200:
                    response_text = response.json().get("response", "").strip()
                    print("\n----------------------------------------")
                    print("DEEPSEEK OUTPUT:")
                    print(response_text)
                    print("----------------------------------------\n")
                    
                    json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
                    if json_match:
                        print("JSON pattern found in response. Parsed data:")
                        evaluation_data = json.loads(json_match.group())
                        print(json.dumps(evaluation_data, indent=2))
                        
                        required_keys = [
                            "overall_score", "technical_competency", "communication_skills",
                            "problem_solving", "cultural_fit", "strengths", "areas_for_improvement",
                            "key_observations", "recommendations", "interview_performance"
                        ]
                        
                        missing_keys = [key for key in required_keys if key not in evaluation_data]
                        if not missing_keys:
                            print("\nAll required keys present in evaluation data")
                            st.session_state.final_evaluation = evaluation_data
                            st.session_state.evaluation_generated = True
                            print("Evaluation data stored in session state")
                            st.rerun()
                        else:
                            print(f"\nMissing required keys in evaluation data: {missing_keys}")
                            raise ValueError(f"Incomplete evaluation data structure. Missing keys: {missing_keys}")
                    else:
                        print("\nNo JSON pattern found in response")
                        raise ValueError("No valid JSON found in response")
                    break
                elif response.status_code == 404:
                    raise Exception(f"API endpoint not found. Please check if Ollama is running and the model '{MODEL}' is available.")
                elif response.status_code == 500:
                    raise Exception("Internal server error. The model might be overloaded.")
                else:
                    raise Exception(f"Unexpected status code: {response.status_code}")
                    
            except (requests.exceptions.ConnectionError, requests.exceptions.Timeout) as e:
                if attempt == max_retries - 1:
                    raise Exception(f"Failed to connect to Ollama after {max_retries} attempts: {str(e)}")
                print(f"Attempt {attempt + 1} failed. Retrying in {retry_delay} seconds...")
                time.sleep(retry_delay)

        # Generate contextual response based on enhanced analysis
        response_prompt = f"""You are conducting a professional job interview as an AI Interviewer. This is an ongoing conversation.

CURRENT CONTEXT:
Position: {position}
Candidate: {candidate_name}
Current Topic: {context}
Progress: Question {exchange_count + 1} of {max_exchanges}
Interview Style: {st.session_state.interview_style}

CONVERSATION HISTORY:
        {conversation_context}
        
CANDIDATE'S LATEST RESPONSE:
{answer}

RESPONSE ANALYSIS:
Intent: {evaluation_data['intent']['primary']} (confidence: {evaluation_data['intent']['confidence']})
Sentiment: {evaluation_data['sentiment']['overall']} (emotions: {', '.join(evaluation_data['sentiment']['specific_emotions'])})
Key Points: {', '.join(evaluation_data['content_analysis']['key_points'])}
Claims to Verify: {', '.join(evaluation_data['follow_up']['verification_needed'])}
Topics to Explore: {', '.join(evaluation_data['follow_up']['recommended_topics'])}
Needs Clarification: {', '.join(evaluation_data['follow_up']['clarification_needed'])}

CONVERSATION MEMORY:
Previously Discussed: {', '.join(list(st.session_state.conversation_memory['topics_discussed'])[-3:])}
Recent Sentiments: {', '.join(st.session_state.conversation_memory['sentiment_history'][-3:])}
Unverified Claims: {json.dumps([claim for claim, info in st.session_state.conversation_memory['claims_made'].items() if info['needs_verification']], indent=2)}

INTERVIEWER TASK:
1. Acknowledge their response with specific references
2. Address any unverified claims or concerns
3. Follow up on interesting points
4. Maintain conversation flow
5. Keep professional but engaging tone

IMPORTANT GUIDELINES:
1. NEVER repeat the exact same question
2. Reference specific details from their answer
3. If they mention metrics or achievements, ask for more details about implementation
4. If they mention leadership roles, ask about specific challenges and solutions
5. Show active listening by connecting new information with previously mentioned points
6. If they seem to be finishing, ask if they would like to add anything else

Generate a natural, contextual response that shows active listening and seeks relevant details.
Ensure the response is unique and not repetitive of previous responses.
Respond as the interviewer, maintaining a professional and engaging tone:"""

        # Check response diversity
        def calculate_similarity(text1, text2):
            """Calculate simple similarity between two texts."""
            words1 = set(text1.lower().split())
            words2 = set(text2.lower().split())
            intersection = words1.intersection(words2)
            union = words1.union(words2)
            return len(intersection) / len(union) if union else 0

        def is_response_repetitive(new_response):
            """Check if response is too similar to recent responses."""
            if not st.session_state.conversation_memory['response_history']:
                return False
                
            # Check last 3 responses
            recent_responses = st.session_state.conversation_memory['response_history'][-3:]
            similarities = [calculate_similarity(new_response, resp) for resp in recent_responses]
            return any(sim > 0.7 for sim in similarities)

        # Generate response with retry logic
        max_retries = 3
        current_try = 0
        
        while current_try < max_retries:
            response = requests.post(
                OLLAMA_API_URL,
                json={
                    "model": MODEL,
                    "prompt": response_prompt,
                    "stream": False
                }
            )
            
            if response.status_code == 200:
                ai_response = response.json().get("response", "").strip()
                
                if ai_response and len(ai_response) > 20:
                    if not is_response_repetitive(ai_response):
                        st.session_state.conversation_memory['response_history'].append(ai_response)
                    return ai_response
                else:
                    current_try += 1
            else:
                current_try += 1
            
        # If we get here, use enhanced fallback responses
        if evaluation_data['content_analysis']['claims_made']:
            return f"You've mentioned some interesting points about {', '.join(evaluation_data['content_analysis']['claims_made'][:2])}. Could you provide specific examples or metrics that demonstrate the impact of your work in these areas?"
        elif evaluation_data['follow_up']['recommended_topics']:
            return f"I'd like to explore more about {evaluation_data['follow_up']['recommended_topics'][0]}. Could you tell me about your specific experience or approach in this area?"
        else:
            return f"Could you elaborate more on your experience and achievements related to the {position} role? I'm particularly interested in specific examples and measurable impacts."
        
    except Exception as e:
        logger.error(f"Error getting AI response: {e}")
        return f"I understand. Could you provide more specific examples related to the {position} role?"

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

# Initialize session state variables if they don't exist
if 'language_selected' not in st.session_state:
    st.session_state.language_selected = False
if 'interview_started' not in st.session_state:
    st.session_state.interview_started = False
if "current_question_index" not in st.session_state:
    st.session_state.current_question_index = 0
if "questions_and_answers" not in st.session_state:
    st.session_state.questions_and_answers = []
if "language" not in st.session_state:
    st.session_state.language = None
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
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "questions" not in st.session_state:
    st.session_state.questions = []
if "enable_hyde" not in st.session_state:
    st.session_state.enable_hyde = False
if "enable_graph_rag" not in st.session_state:
    st.session_state.enable_graph_rag = False
if "enable_reranking" not in st.session_state:
    st.session_state.enable_reranking = False
if "max_contexts" not in st.session_state:
    st.session_state.max_contexts = 5
if "documents_loaded" not in st.session_state:
    st.session_state.documents_loaded = False
if "processing" not in st.session_state:
    st.session_state.processing = False

# Add interview completed state
if 'interview_completed' not in st.session_state:
    st.session_state.interview_completed = False

# Language selection screen
if not st.session_state.language_selected:
    # Create header with logo and title side by side
    header_col1, header_col2 = st.columns([1, 2])
    
    with header_col1:
        st.image("images/Muqabala_Chatbot-removebg-preview.png", width=250)
    
    with header_col2:
        st.markdown("""
            <div style='padding: 30px 0px;'>
                <h1 style='font-size: 3em; margin: 0;'>Muqabala AI Interview Chatbot</h1>
            </div>
        """, unsafe_allow_html=True)
    
    # Add some space
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Language selection
    st.title("Choose Your Language / اختر لغتك")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("🇬🇧 English", use_container_width=True):
            st.session_state.language = "en"
            st.session_state.language_selected = True
            st.rerun()
    
    with col2:
        if st.button("🇦🇪 العربية", use_container_width=True):
            st.session_state.language = "ar"
            st.session_state.language_selected = True
            st.rerun()
    
    st.stop()

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

def display_chat_message(text: str, is_user: bool, is_new: bool = False):
    """Display a chat message with enhanced styling and typing animation only for new AI messages."""
    message_class = "user" if is_user else "ai"
    avatar_text = "👤" if is_user else "🤖"
    
    message_html = f"""
        <div class="chat-message {message_class}">
            <div class="message-wrapper">
            <div class="message-avatar">{avatar_text}</div>
            <div class="message-content">{text}</div>
        </div>
        </div>
    """
    
    # Add typing animation only for new AI messages
    if not is_user and is_new:
        with st.empty():
            for i in range(len(text) + 1):
                current_text = text[:i]
                st.markdown(f"""
                    <div class="chat-message {message_class}">
                        <div class="message-wrapper">
                            <div class="message-avatar">{avatar_text}</div>
                            <div class="message-content">{current_text}</div>
                </div>
            </div>
        """, unsafe_allow_html=True)
                time.sleep(0.01)  # Adjust typing speed
    else:
        st.markdown(message_html, unsafe_allow_html=True)

# Main interview interface
if st.session_state.interview_started:
    if st.session_state.interview_completed and not st.session_state.get('evaluation_generated', False):
        st.info("Generating evaluation... Please wait.")
        print("Starting evaluation generation process...")
        
        try:
            # Check if Ollama server is running
            try:
                health_check = requests.get(OLLAMA_BASE_URL)
                if health_check.status_code != 200:
                    print(f"Ollama server health check failed with status {health_check.status_code}")
                    raise Exception("Ollama server is not responding")
                print("Ollama server is running")
            except requests.exceptions.ConnectionError:
                print("Failed to connect to Ollama server")
                raise Exception("Cannot connect to Ollama server. Please ensure it is running.")
            
            # Prepare conversation history for analysis
            print(f"Number of messages in chat history: {len(st.session_state.chat_history)}")
            print(f"Number of questions: {len(st.session_state.questions)}")
            
            conversation_text = "\n".join([
                f"Q: {st.session_state.questions[i//2]}\nA: {msg['content']}"
                for i, msg in enumerate(st.session_state.chat_history)
                if msg['role'] == 'user'
            ])
            print(f"Prepared conversation text length: {len(conversation_text)}")
            print("Sample of conversation text:")
            print(conversation_text[:500] + "..." if len(conversation_text) > 500 else conversation_text)
            
            print("\nSending analysis request to DeepSeek...")
            
            # Get DeepSeek analysis with improved prompt
            analysis_prompt = f"""You are an expert AI interviewer evaluating a candidate's performance for a {st.session_state.candidate_info['position']} position.

INTERVIEW SUMMARY:
Candidate: {st.session_state.candidate_info['name']}
Position: {st.session_state.candidate_info['position']}
Total Questions: {len(st.session_state.questions)}

INTERVIEW TRANSCRIPT:
{conversation_text}

EVALUATION TASK:
Analyze the candidate's responses and provide a structured evaluation that MUST follow the exact JSON format below.
Focus on concrete examples, specific skills, and measurable achievements mentioned.

REQUIRED OUTPUT FORMAT (EXAMPLE):
{{
    "overall_score": 0.85,
    "technical_competency": 0.82,
    "communication_skills": 0.88,
    "problem_solving": 0.85,
    "cultural_fit": 0.87,
    "strengths": [
        "Strong technical background in relevant technologies",
        "Excellent communication and articulation of ideas",
        "Proven track record of problem-solving"
    ],
    "areas_for_improvement": [
        "Could provide more specific metrics for achievements",
        "Some responses could be more concise",
        "Limited examples of leadership experience"
    ],
    "key_observations": [
        "Demonstrates deep technical knowledge",
        "Shows enthusiasm and cultural alignment",
        "Good balance of technical and soft skills"
    ],
    "recommendations": [
        "Consider for senior technical roles",
        "Would benefit from leadership opportunities",
        "Recommend technical team lead position"
    ],
    "interview_performance": {{
        "confidence": 0.9,
        "clarity": 0.85,
        "relevance": 0.88,
        "depth": 0.87
    }}
}}

EVALUATION GUIDELINES:
1. ALL fields in the above JSON structure are REQUIRED
2. ALL numeric scores must be between 0.0 and 1.0
3. Each list must contain exactly 3 items
4. The interview_performance object MUST include all four metrics
5. Base scores on specific examples and achievements mentioned
6. Consider relevance of responses to the {st.session_state.candidate_info['position']} role
7. Evaluate technical claims against industry standards
8. Assess communication clarity and professionalism
9. Consider cultural fit based on values and work style mentioned

Return ONLY the JSON object with ALL required fields, no additional text or explanation."""
            
            # Get DeepSeek analysis with improved prompt
            print("\n----------------------------------------")
            print("SENDING PROMPT TO DEEPSEEK:")
            print(analysis_prompt)
            print("----------------------------------------\n")
            
            # Add retry logic for API calls
            max_retries = 3
            retry_delay = 2  # seconds
            
            for attempt in range(max_retries):
                try:
                    response = requests.post(
                        OLLAMA_API_URL,
                        json={
                            "model": MODEL,
                            "prompt": analysis_prompt,
                            "stream": False
                        },
                        timeout=30
                    )
                    
                    print(f"\nAttempt {attempt + 1}: DeepSeek API response status code: {response.status_code}")
                    
                    if response.status_code == 200:
                        response_text = response.json().get("response", "").strip()
                        print("\n----------------------------------------")
                        print("DEEPSEEK OUTPUT:")
                        print(response_text)
                        print("----------------------------------------\n")
                        
                        json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
                        if json_match:
                            print("JSON pattern found in response. Parsed data:")
                            evaluation_data = json.loads(json_match.group())
                            print(json.dumps(evaluation_data, indent=2))
                            
                            required_keys = [
                                "overall_score", "technical_competency", "communication_skills",
                                "problem_solving", "cultural_fit", "strengths", "areas_for_improvement",
                                "key_observations", "recommendations", "interview_performance"
                            ]
                            
                            missing_keys = [key for key in required_keys if key not in evaluation_data]
                            if not missing_keys:
                                print("\nAll required keys present in evaluation data")
                                st.session_state.final_evaluation = evaluation_data
                                st.session_state.evaluation_generated = True
                                print("Evaluation data stored in session state")
                                st.rerun()
                            else:
                                print(f"\nMissing required keys in evaluation data: {missing_keys}")
                                raise ValueError(f"Incomplete evaluation data structure. Missing keys: {missing_keys}")
                        else:
                            print("\nNo JSON pattern found in response")
                            raise ValueError("No valid JSON found in response")
                        break
                    elif response.status_code == 404:
                        raise Exception(f"API endpoint not found. Please check if Ollama is running and the model '{MODEL}' is available.")
                    elif response.status_code == 500:
                        raise Exception("Internal server error. The model might be overloaded.")
                    else:
                        raise Exception(f"Unexpected status code: {response.status_code}")
                        
                except (requests.exceptions.ConnectionError, requests.exceptions.Timeout) as e:
                    if attempt == max_retries - 1:
                        raise Exception(f"Failed to connect to Ollama after {max_retries} attempts: {str(e)}")
                    print(f"Attempt {attempt + 1} failed. Retrying in {retry_delay} seconds...")
                    time.sleep(retry_delay)
            
        except Exception as e:
            print(f"Error in evaluation generation: {str(e)}")
            logger.error(f"Error generating evaluation: {e}")
            st.error("Failed to generate evaluation. Please try again.")
            st.session_state.interview_completed = False
            st.session_state.evaluation_generated = False
            time.sleep(2)
            st.rerun()
    
    elif st.session_state.interview_completed and st.session_state.get('evaluation_generated', False):
        # Display evaluation results
        st.markdown("## 📊 Interview Evaluation")
        evaluation_data = st.session_state.final_evaluation
        
        # Overall Score
        st.markdown(f"### Overall Performance: {evaluation_data['overall_score']*100:.1f}%")
        st.progress(evaluation_data['overall_score'])
        
        # Key Metrics
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### 📈 Core Competencies")
            metrics = {
                "Technical Competency": evaluation_data['technical_competency'],
                "Communication Skills": evaluation_data['communication_skills'],
                "Problem Solving": evaluation_data['problem_solving'],
                "Cultural Fit": evaluation_data['cultural_fit']
            }
            
            # Create radar chart data
            import plotly.graph_objects as go
            
            fig = go.Figure(data=go.Scatterpolar(
                r=list(metrics.values()),
                theta=list(metrics.keys()),
                fill='toself'
            ))
            
            fig.update_layout(
                polar=dict(
                    radialaxis=dict(
                        visible=True,
                        range=[0, 1]
                    )),
                showlegend=False
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.markdown("### 🎯 Interview Performance")
            performance = evaluation_data['interview_performance']
            
            # Create bar chart
            import plotly.express as px
            
            df = pd.DataFrame({
                'Metric': list(performance.keys()),
                'Score': list(performance.values())
            })
            
            fig = px.bar(df, x='Metric', y='Score',
                       color='Score',
                       color_continuous_scale='viridis',
                       range_y=[0, 1])
            
            st.plotly_chart(fig, use_container_width=True)
        
        # Strengths and Improvements
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### 💪 Strengths")
            for strength in evaluation_data['strengths']:
                st.markdown(f"- {strength}")
        
        with col2:
            st.markdown("### 🎯 Areas for Improvement")
            for area in evaluation_data['areas_for_improvement']:
                st.markdown(f"- {area}")
        
        # Key Observations and Recommendations
        st.markdown("### 🔍 Key Observations")
        for observation in evaluation_data['key_observations']:
            st.markdown(f"- {observation}")
        
        st.markdown("### 📝 Recommendations")
        for recommendation in evaluation_data['recommendations']:
            st.markdown(f"- {recommendation}")
        
        # Add download report button
        if st.button("Download Detailed Report"):
            output_path = f"reports/{st.session_state.candidate_info['name'].replace(' ', '_')}_report.pdf"
            os.makedirs("reports", exist_ok=True)
            
            report_generator.generate_report(
                candidate_name=st.session_state.candidate_info["name"],
                position=st.session_state.candidate_info["position"],
                interview_date=st.session_state.candidate_info["date"],
                questions_and_answers=st.session_state.questions_and_answers,
                evaluation_metrics=evaluation_data,
                output_path=output_path,
                language=st.session_state.language
            )
            st.success(f"Report generated: {output_path}")
    
    else:
        # Create a container for the chat
        chat_container = st.container()
        
        with chat_container:
            # Display chat messages
            for i, message in enumerate(st.session_state.chat_history):
                is_new = (i == len(st.session_state.chat_history) - 1) and (i > st.session_state.last_message_id)
                display_chat_message(message["content"], message["role"] == "user", is_new)
            
            # Update last_message_id
            st.session_state.last_message_id = len(st.session_state.chat_history) - 1
        
        # Display current question if no messages yet
        if not st.session_state.chat_history:
            current_q = st.session_state.questions[0]
            display_chat_message(current_q, False, True)
        
        # Input area
        input_container = st.container()
        with input_container:
            st.markdown('<div class="input-area">', unsafe_allow_html=True)
            col1, col2 = st.columns([3, 1])
            
            with col1:
                user_input = st.text_input(
                    "Response Input",
                    placeholder="Type your response...",
                    label_visibility="collapsed",
                    key="user_input"
                )
            
            with col2:
                # Record button with new styling
                st.markdown("""
                    <div class="recording-button-container">
                        <button class="recording-button" id="recordButton" type="button">
                            <span class="recording-indicator"></span>
                            <span>🎤 Record</span>
                        </button>
                    </div>
                """, unsafe_allow_html=True)
            
            st.markdown('</div>', unsafe_allow_html=True)
            
            # Send button
            if st.button("Send 📤", use_container_width=True, key="send_button"):
                if user_input:
                    # Add user message to chat history
                    st.session_state.chat_history.append({
                        "role": "user",
                        "content": user_input
                    })
                    
                    # Get current question context safely
                    question_index = min(len(st.session_state.chat_history) // 2, len(st.session_state.questions) - 1)
                    current_q = st.session_state.questions[question_index]
                    
                    # Get AI response
                    with st.spinner("Processing response..."):
                        ai_response = get_ai_response(
                            user_input,
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
                    
                    # Check if interview is completed
                    if len(st.session_state.chat_history) >= len(st.session_state.questions) * 2:
                        st.session_state.interview_completed = True
                    
                    st.rerun()

else:
    # Welcome screen with language-specific content
    if st.session_state.language == "ar":
        st.title("👋 مرحباً بك في مقابلة")
        st.markdown("""
            ### مساعد المقابلات المدعوم بالذكاء الاصطناعي
            
            مقابلة تساعد في إجراء وتقييم المقابلات الوظيفية من خلال:
            
            - 🗣️ التعرف على الكلام والتوليف
            - 🌐 دعم اللغتين الإنجليزية والعربية
            - 📊 تقييم الإجابات في الوقت الفعلي
            - 📝 إنشاء تقرير PDF مفصل
            
            للبدء، يرجى إدخال معلومات المرشح في الشريط الجانبي.
        """)
    else:
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
        if key not in ["speech_handler", "evaluator", "report_generator", "language", "language_selected"]:
            del st.session_state[key]

# Sidebar
with st.sidebar:
    st.title("👔 Interview Settings" if st.session_state.language == "en" else "👔 إعدادات المقابلة")
    
    # Only show language switcher when interview is not in progress
    if not st.session_state.interview_started:
        # Add language switcher at the top of sidebar
        current_lang = "🇬🇧 English" if st.session_state.language == "en" else "🇦🇪 العربية"
        new_lang = st.selectbox(
            "Language / اللغة",
            ["🇬🇧 English", "🇦🇪 العربية"],
            index=0 if st.session_state.language == "en" else 1,
            key="language_selector"
        )
        
        # Handle language change
        if new_lang != current_lang:
            new_lang_code = "en" if new_lang == "🇬🇧 English" else "ar"
            if st.session_state.language != new_lang_code:
                st.session_state.language = new_lang_code
                st.rerun()
    
    if not st.session_state.interview_started:
        name_label = "Candidate Name" if st.session_state.language == "en" else "اسم المرشح"
        position_label = "Position" if st.session_state.language == "en" else "المنصب"
        
        st.session_state.candidate_info["name"] = st.text_input(
            name_label,
            key="candidate_name"
        )
        st.session_state.candidate_info["position"] = st.text_input(
            position_label,
            key="position"
        )
        
        # Add interview style options
        st.markdown("### Interview Style" if st.session_state.language == "en" else "### نمط المقابلة")
        style_label = "Conversation Style" if st.session_state.language == "en" else "نمط المحادثة"
        style_options = ["Formal", "Balanced", "Casual"] if st.session_state.language == "en" else ["رسمي", "متوازن", "غير رسمي"]
        interview_style = st.select_slider(
            style_label,
            options=style_options,
            value=style_options[1]
        )
        
        start_button_text = "Start Interview" if st.session_state.language == "en" else "بدء المقابلة"
        if st.button(start_button_text):
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
