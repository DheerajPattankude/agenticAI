# streamlit_app.py
import os
import json
import requests
from openai import OpenAI
from docx import Document
from deep_translator import GoogleTranslator
import streamlit as st
from io import BytesIO

# -----------------------
# Configuration
# -----------------------
HF_TOKEN = os.environ.get("HF_TOKEN")
# Preferred HF model (may be gated). The app will attempt this first then fallback.
MODEL = "meta-llama/Llama-3.1-8B-Instruct:cerebras"

# OpenAI-client configured to use Hugging Face OpenAI-compatible endpoint
# (we'll try this path first)
client = OpenAI(
    base_url="https://router.huggingface.co/v1",
    api_key=HF_TOKEN,
)

# -----------------------
# Helper: call via OpenAI client (OpenAI-style)
# -----------------------
def call_via_openai_client(model, messages):
    # uses the openai.OpenAI client (chat completions)
    try:
        resp = client.chat.completions.create(
            model=model,
            messages=messages,
        )
        # Try to extract content (safe access)
        return resp.choices[0].message.content if hasattr(resp.choices[0].message, "content") else resp.choices[0].message
    except Exception as e:
        # Return exception so caller can fallback
        raise

# -----------------------
# Helper: call via HF native inference API
# -----------------------
def call_hf_native(model, prompt, max_new_tokens=256):
    api_url = f"https://router.huggingface.co/v1/models/{model}"
    headers = {"Authorization": f"Bearer {HF_TOKEN}"} if HF_TOKEN else {}
    payload = {"inputs": prompt, "parameters": {"max_new_tokens": max_new_tokens}}
    r = requests.post(api_url, headers=headers, json=payload, timeout=120)
    # raise for status to be handled by caller
    r.raise_for_status()
    data = r.json()
    # Many models return [{'generated_text': '...'}]
    if isinstance(data, list) and "generated_text" in data[0]:
        return data[0]["generated_text"]
    # Some models return plain text or dict
    if isinstance(data, dict):
        # try common keys
        for k in ("generated_text", "text", "output"):
            if k in data:
                return data[k]
        return json.dumps(data)
    # fallback
    return str(data)

# -----------------------
# Unified agent caller: try openai client first, fallback to HF native
# -----------------------
def query_model_with_fallback(system_prompt, user_prompt, model=MODEL):
    # Build messages for chat
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]
    # try openai-style call first
    try:
        return call_via_openai_client(model, messages)
    except Exception as e_openai:
        # fallback: build a single prompt mixing system+user
        fallback_prompt = system_prompt + "\n\nUser: " + user_prompt + "\nAssistant:"
        try:
            return call_hf_native(model, fallback_prompt)
        except requests.exceptions.HTTPError as http_err:
            # If HF native 404 or gated, fallback to FALLBACK_MODEL native
            try:
                fallback_prompt2 = system_prompt + "\n\nUser: " + user_prompt + "\nAssistant:"
                return call_hf_native(MODEL, fallback_prompt2)
            except Exception as final_err:
                return f"[Error contacting models]\nOpenAI-client error: {e_openai}\nHF error: {http_err}\nFinal fallback error: {final_err}"
        except Exception as ex2:
            return f"[Error contacting HF native API]\nOpenAI-client error: {e_openai}\nHF error: {ex2}"

# -----------------------
# UI: Streamlit layout
# -----------------------
st.set_page_config(page_title="Me CM Assistant", layout="wide")
st.markdown(
    """
    <style>
    .card {
      border-radius: 12px;
      padding: 16px;
      background: linear-gradient(135deg, #1e3c70, #2a5290);
      background-size: cover;
      color: White;
      box-shadow: 0 4px 24px rgba(10, 40, 80, 0.08);
      margin-bottom: 12px;
    }
    .agent-title { font-weight:600; font-size:18px; }
    .small { color: #660; font-size:12px; }
    </style>
    """,
    unsafe_allow_html=True
)

st.title("ME CM Assistant (Indian Institution / Police / Lord Krishna)")
st.write("Ask a question and choose a language. Three agents will answer in parallel.")

# Input controls
with st.form("ask_form"):
    col1, col2 = st.columns([4,1])
    with col1:
        user_question = st.text_area("Enter your question", height=120)
    with col2:
        language = st.selectbox("Output language", ["English", "Kannada"])
        submit = st.form_submit_button("Ask")

if submit:
    if not user_question.strip():
        st.warning("Please enter a question.")
    else:
        # Show loader and then display results
        st.info("Generating answers... this may take a few seconds depending on model access.")
        placeholder = st.empty()
        with placeholder.container():
            st.markdown("### Generating...")

        # Define system prompts for each agent
        sys1 = "You are an experienced advisor who answers based on working experience in Indian institutions. Provide crisp, step-by-step guidance, minimal sentences, actionable steps."
        sys2 = "You are a police guideline officer who provides advice based on Indian police rules, procedures and official practice. Provide crisp, step-by-step guidance and document references if applicable."
        sys3 = "You are Lord Krishna answering with wisdom from the Bhagavad Gita and Mahabharata. Give concise guidance, optionally include a short shloka in Sanskrit and a brief explanation in simple words."

        # Query each agent (sequentially)
        with st.spinner("Calling Indian Institution agent..."):
            answer1 = query_model_with_fallback(sys1, user_question)
        with st.spinner("Calling Police Guidelines agent..."):
            answer2 = query_model_with_fallback(sys2, user_question)
        with st.spinner("Calling Lord Krishna agent..."):
            answer3 = query_model_with_fallback(sys3, user_question)

        # Translate if needed
        if language == "Kannada":
            try:
                k1 = GoogleTranslator(source='auto', target='kn').translate(answer1)
                k2 = GoogleTranslator(source='auto', target='kn').translate(answer2)
                k3 = GoogleTranslator(source='auto', target='kn').translate(answer3)
            except Exception as trans_e:
                st.error(f"Translation failed: {trans_e}")
                k1, k2, k3 = answer1, answer2, answer3
        else:
            k1, k2, k3 = answer1, answer2, answer3

        # Remove the loading placeholder and show results with visuals
        placeholder.empty()
        st.markdown("## Results")
        col_a, col_b, col_c = st.columns(3)

        def render_agent(col, title, emoji, text, language_label):
            with col:
                st.markdown(f'<div class="card"><div class="agent-title">{emoji} {title}</div><div class="small">{language_label}</div><hr/></div>', unsafe_allow_html=True)
                st.write(text)

        render_agent(col_a, "Indian Institution Advisor", "🏫", k1, language)
        render_agent(col_b, "Police Guideline Officer", "👮‍♂️", k2, language)
        render_agent(col_c, "Lord Krishna ", "🕉️", k3, language)

        # Offer a DOCX download
        doc = Document()
        doc.add_heading("AI Multi-Agent Responses", level=1)
        doc.add_paragraph(f"Question: {user_question}\n")
        doc.add_heading("1. Indian Institution Advisor", level=2)
        doc.add_paragraph(k1)
        doc.add_heading("2. Police Guideline Officer", level=2)
        doc.add_paragraph(k2)
        doc.add_heading("3. Lord Krishna (Bhagavad Gita)", level=2)
        doc.add_paragraph(k3)

        bio = BytesIO()
        doc.save(bio)
        bio.seek(0)
        st.download_button("Download responses as Word (.docx)", data=bio, file_name="AI_Agent_Responses.docx", mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document")
