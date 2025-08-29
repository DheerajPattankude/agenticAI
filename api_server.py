# app.py
import streamlit as st
from docx import Document
from io import BytesIO
import os
import openai
from deep_translator import GoogleTranslator
import time
import threading
from gtts import gTTS
import speech_recognition as sr
from streamlit_webrtc import webrtc_streamer, AudioProcessorBase, RTCConfiguration
import av
import numpy as np

# =========================
# PAGE CONFIG & STYLE
# =========================
st.set_page_config(page_title="Multi-Agent AI", page_icon="🤖", layout="centered")

# 🎨 keep your CSS (unchanged) ...

# =========================
# LANGUAGE CODE MAPPING
# =========================
LANGUAGE_CODES = {
    "English": "en",
    "Hindi": "hi",
    "Marathi": "mr",
    "Gujarati": "gu",
    "Tamil": "ta",
    "Telugu": "te",
    "Kannada": "kn",
    "Malayalam": "ml",
    "Bengali": "bn",
    "Punjabi": "pa"
}

# =========================
# SYSTEM PROMPTS FOR AGENTS (unchanged)
# =========================
AGENT_PROMPTS = {
    "Indian Institution Advisor": "...",
    "Police Guideline Officer": "...",
    "Lord Krishna": "...",
    "Dr. Ambedkar": "...",
    "Bhagwan Mahaveer": "...",
    "Bhagwan Budda": "...",
    "IAS role as DC": "...",
    "IAS role as Secretary": "..."
}

# =========================
# TRANSLATOR + AUDIO
# =========================
class RateLimitedTranslator:
    def __init__(self):
        self.last_call_time = 0
        self.min_interval = 0.34
        self.lock = threading.Lock()
    def translate(self, text, dest):
        with self.lock:
            elapsed = time.time() - self.last_call_time
            if elapsed < self.min_interval:
                time.sleep(self.min_interval - elapsed)
            try:
                result = GoogleTranslator(source="auto", target=dest).translate(text)
                self.last_call_time = time.time()
                return result
            except Exception as e:
                return f"Translation error: {str(e)}. Original text: {text}"

translator = RateLimitedTranslator()

def generate_audio(text, lang_code, filename="output.mp3"):
    try:
        tts = gTTS(text=text, lang=lang_code)
        tts.save(filename)
        return filename
    except Exception as e:
        st.error(f"Audio generation error: {str(e)}")
        return None

# =========================
# OPENAI CLIENT (unchanged)
# =========================
def get_openai_client():
    try:
        HF_TOKEN = os.environ.get("HF_TOKEN")
        if not HF_TOKEN:
            try:
                HF_TOKEN = st.secrets.get("HF_TOKEN")
            except (FileNotFoundError, AttributeError):
                pass
        if not HF_TOKEN and "hf_token" in st.session_state:
            HF_TOKEN = st.session_state.hf_token
        if not HF_TOKEN:
            st.warning("Enter Hugging Face API token.")
            api_key = st.text_input("HF API token:", type="password")
            if api_key:
                st.session_state.hf_token = api_key
                st.rerun()
            return None
        return openai.OpenAI(base_url="https://router.huggingface.co/v1", api_key=HF_TOKEN)
    except Exception as e:
        st.error(f"Error initializing client: {str(e)}")
        return None

client = get_openai_client()

# =========================
# QUERY FUNCTION (unchanged)
# =========================
def query_model_combined(user_prompt, selected_agents):
    if not client:
        return "Error: OpenAI client not initialized."
    try:
        combined_instructions = (
            "You are a multi-role assistant. For each role..."
        )
        role_texts = []
        for agent in selected_agents:
            if agent in AGENT_PROMPTS:
                role_texts.append(f"Role: {agent}\nSystem Prompt: {AGENT_PROMPTS[agent]}\n")
        if not role_texts:
            return "Error: No valid agents selected."
        final_prompt = f"Question: {user_prompt}\n\n" + "\n".join(role_texts)
        response = client.chat.completions.create(
            model="meta-llama/Llama-3.1-8B-Instruct:cerebras",
            messages=[{"role": "system", "content": combined_instructions},
                      {"role": "user", "content": final_prompt}],
            temperature=0.7,
            max_tokens=4000
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"API Error: {str(e)}"

def parse_agent_responses(raw_response, selected_agents):
    responses = {}
    if not raw_response or raw_response.startswith("Error:"):
        return {agent: "No response." for agent in selected_agents}
    blocks = raw_response.split("### ")
    for block in blocks:
        if not block.strip():
            continue
        lines = block.split("\n")
        if lines and ":" in lines[0]:
            role = lines[0].split(":")[0].strip()
            content = "\n".join(lines[1:]).strip()
            if role in selected_agents:
                responses[role] = content
    for agent in selected_agents:
        if agent not in responses:
            responses[agent] = "No response."
    return responses

def translate_batch_texts(text_list, target_lang):
    return [translator.translate(t, target_lang) if not t.startswith("Error") else t for t in text_list]

# =========================
# SPEECH RECOGNITION WRAPPER
# =========================
def speech_to_text(audio_data):
    recognizer = sr.Recognizer()
    try:
        text = recognizer.recognize_google(audio_data, language="en-IN")
        return text
    except sr.UnknownValueError:
        return ""
    except sr.RequestError as e:
        return f"Speech recognition error: {str(e)}"

class AudioProcessor(AudioProcessorBase):
    def __init__(self):
        self.recognizer = sr.Recognizer()
        self.text = ""
    def recv(self, frame):
        audio = frame.to_ndarray().flatten().astype(np.float32) / 32768.0
        wav_bytes = audio.tobytes()
        with sr.AudioFile(BytesIO(wav_bytes)) as source:
            try:
                self.text = self.recognizer.recognize_google(self.recognizer.record(source))
            except Exception:
                pass
        return frame

# =========================
# UI
# =========================
st.title("🤖 Me CM Assistant (Realtime Voice)")

if client:
    st.subheader("🎤 Speak your Question")
    rtc_configuration = RTCConfiguration({"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]})
    webrtc_ctx = webrtc_streamer(
        key="speech",
        mode=WebRtcMode.RECVONLY,
        rtc_configuration=rtc_configuration,
        audio_processor_factory=AudioProcessor,
        media_stream_constraints={"audio": True, "video": False},
    )

    if "voice_question" not in st.session_state:
        st.session_state.voice_question = ""

    if webrtc_ctx and hasattr(webrtc_ctx, "audio_processor"):
        if webrtc_ctx.audio_processor and webrtc_ctx.audio_processor.text:
            st.session_state.voice_question = webrtc_ctx.audio_processor.text
            st.success(f"Recognized Speech: {st.session_state.voice_question}")

    # Text + Voice Inputs
    user_question = st.text_area("💬 Or type your question:", value=st.session_state.voice_question, height=120)
    language = st.selectbox("🌐 Select language:", list(LANGUAGE_CODES.keys()))
    agent_options = list(AGENT_PROMPTS.keys())
    selected_agents = st.multiselect("🤝 Select agents:", options=agent_options)

    col1, col2 = st.columns(2)
    with col1:
        text_submit = st.button("📝 Get Answers (Text)")
    with col2:
        audio_submit = st.button("🔊 Get Answers (Audio)")

    if (text_submit or audio_submit) and user_question:
        with st.spinner("Generating responses..."):
            raw_response = query_model_combined(user_question, selected_agents)
            responses = parse_agent_responses(raw_response, selected_agents)
            answer_texts = [responses[a] for a in selected_agents]
            translated_texts = translate_batch_texts(answer_texts, LANGUAGE_CODES[language])

            audio_files = {}
            if audio_submit:
                for agent, ans in zip(selected_agents, translated_texts):
                    audio_files[agent] = generate_audio(ans, LANGUAGE_CODES[language], f"{agent}.mp3")

            for agent, ans in zip(selected_agents, translated_texts):
                st.subheader(agent)
                st.write(ans)
                if audio_submit and audio_files.get(agent):
                    st.audio(audio_files[agent], format="audio/mp3")
else:
    st.info("⚠️ Please set your HuggingFace API key.")

