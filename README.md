# 🎯 Muqabala: AI-Powered Interviewer

An intelligent interview assistant that conducts, evaluates, and generates reports for job interviews using speech recognition, multilingual support, and advanced NLP techniques.

## 🌟 Features

- **Speech Recognition & Synthesis**

  - Real-time speech-to-text for candidate responses
  - Text-to-speech for interview questions
  - Support for both English and Arabic

- **Interview Evaluation**

  - Real-time response analysis
  - Technical depth assessment
  - Confidence scoring
  - Language proficiency evaluation

- **Detailed Reporting**
  - Comprehensive PDF reports
  - Question-by-question analysis
  - Overall performance metrics
  - Bilingual support (English/Arabic)

## 🚀 Installation

1. **Clone the Repository**

   ```bash
   git clone https://github.com/abdualblooshi/muqabala.git
   cd muqabala
   ```

2. **Create Virtual Environment**

   ```bash
   python -m venv venv

   # Windows
   venv\Scripts\activate

   # Linux/Mac
   source venv/bin/activate
   ```

3. **Install Dependencies**

   ```bash
   pip install -r requirements.txt
   ```

4. **Set Up Environment Variables**
   Create a `.env` file with:

   ```
   OLLAMA_API_URL=http://localhost:11434
   MODEL=deepseek-r1:7b
   ```

5. **Install Ollama**
   - Download from [ollama.com](https://ollama.com)
   - Pull required models:
     ```bash
     ollama pull deepseek-r1:7b
     ```

## 🎮 Usage

1. **Start Ollama Server**

   ```bash
   ollama serve
   ```

2. **Launch Muqabala**

   ```bash
   streamlit run app.py
   ```

3. **Access the Interface**
   - Open your browser at `http://localhost:8501`
   - Enter candidate information
   - Select interview language
   - Start the interview process

## 🏗️ Project Structure

```
muqabala/
├── app.py                         # Main Streamlit application
├── interview_modules/
│   ├── speech/
│   │   └── speech_handler.py      # Speech recognition & synthesis
│   ├── evaluation/
│   │   └── evaluator.py          # Response evaluation
│   └── report_gen/
│       └── report_generator.py    # PDF report generation
├── requirements.txt
└── README.md
```

## 🔧 Technologies Used

- **Speech Processing**

  - SpeechRecognition
  - Mozilla TTS
  - pyttsx3 (fallback TTS)

- **NLP & Evaluation**

  - Transformers
  - BERT-based models
  - Zero-shot classification

- **Report Generation**

  - ReportLab
  - Arabic-Reshaper
  - python-bidi

- **Web Interface**
  - Streamlit
  - Custom CSS styling

## 👥 Team

- **Abdulrahman Alblooshi**
- **Hessa Almaazmi**
- **Mahmoud Muwafi**
- **Afzal M.H**

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
