# 🤖 Me CM Assistant (Realtime Voice AI)

This is a **Streamlit-based Multi-Agent AI Assistant** with:
- 🎤 **Realtime Voice Input** (via microphone, `streamlit-webrtc`)
- 💬 **Multi-Agent Text Responses** (Hugging Face models via OpenAI client wrapper)
- 🔊 **Voice Output** (answers spoken using Google TTS `gTTS`)
- 🌐 **Multilingual Support** (via `deep-translator`)

---

## 🚀 Features
- Ask questions by **voice or text**
- Get answers from multiple personas:
  - Indian Institution Advisor
  - Police Guideline Officer
  - Lord Krishna
  - Dr. Ambedkar
  - Bhagwan Mahaveer
  - Bhagwan Budda
  - IAS role as DC
  - IAS role as Secretary
- **Download answers as Word document**
- Supports multiple Indian languages (Hindi, Marathi, Tamil, Telugu, Bengali, etc.)
- **Realtime transcription** of speech to text
- **Spoken answers** using `gTTS`

---

## 🛠️ Installation

### 1. Clone this repo
```bash
git clone https://github.com/your-username/mecm-assistant.git
cd mecm-assistant

### 2. Create virtual environment
```bash
python -m venv venv
source venv/bin/activate      # On Linux/Mac
venv\Scripts\activate         # On Windows

### 3. Install dependencies
```bash
pip install -r requirements.txt

