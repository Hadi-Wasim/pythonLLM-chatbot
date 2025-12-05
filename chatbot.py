# chatbot_lightweight.py
import os
import time
import tempfile
from datetime import datetime
from typing import List, Tuple

import gradio as gr
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from PyPDF2 import PdfReader

# === Imports for RAG (Lightweight) ===
from sentence_transformers import SentenceTransformer, util

# === Google Gemini (Cloud) ===
try:
    import google.generativeai as genai
    HAVE_GEMINI = True
except ImportError:
    HAVE_GEMINI = False
    print("❌ Error: 'google-generativeai' library not found. Please pip install it.")

# === Optional Voice Dependencies ===
try:
    from gtts import gTTS
    HAVE_GTTS = True
except ImportError:
    HAVE_GTTS = False

try:
    # Using "tiny" model to save RAM. Change to "base" if you have >8GB RAM.
    from faster_whisper import WhisperModel
    HAVE_WHISPER = True
    # Force CPU usage to save GPU/RAM complications
    whisper_model = WhisperModel("tiny", device="cpu", compute_type="int8")
except ImportError:
    HAVE_WHISPER = False
    whisper_model = None

# ---------------- API KEY SETUP ----------------
# 👇 PASTE YOUR KEY HERE IF NOT IN ENVIRONMENT VARIABLES
API_KEY = ""

if HAVE_GEMINI:
    genai.configure(api_key=API_KEY)

# ---------------- Settings ----------------
BRAND_NAME = "Hadi Wasim — Cloud Chatbot"
ACCENT_A = "#7C3AED"
ACCENT_B = "#06B6D4"
BG = "#0b0f1a"
CARD_BG = "#0f1724"
TEXT = "#E6EEF3"
TMP_DIR = "tmp_chat_logs"
os.makedirs(TMP_DIR, exist_ok=True)

# Cleanup old temp files on start
for f in os.listdir(TMP_DIR):
    try:
        os.remove(os.path.join(TMP_DIR, f))
    except:
        pass

# ---------------- Model options (Cloud Only) ----------------
MODEL_OPTIONS = {
    "Gemini 2.5 Flash (Fastest)": "gemini-2.5-flash",
    "Gemini 2.5 Pro (Smartest)": "gemini-2.5-pro",
}
DEFAULT_MODEL = "Gemini 2.5 Flash (Fastest)"

# ---------------- RAG (Lightweight Embedding) ----------------
# This model is very small (~80MB) and runs fast on CPU
EMBEDDER = SentenceTransformer("all-MiniLM-L6-v2", device="cpu")

RAG_STORE = {"chunks": [], "embeddings": None}  # (filename, text)

def extract_text_from_pdf(file_path):
    try:
        reader = PdfReader(file_path)
        return "\n".join(page.extract_text() or "" for page in reader.pages)
    except Exception as e:
        return f"Error reading PDF: {e}"

def chunk_text(text: str, chunk_size: int = 400, overlap: int = 50):
    words = text.split()
    chunks = []
    i = 0
    while i < len(words):
        chunk = " ".join(words[i:i + chunk_size])
        if chunk.strip():
            chunks.append(chunk)
        i += chunk_size - overlap
    return chunks

def add_pdfs_to_rag(files):
    if not files:
        return "⚠️ No files uploaded."
    
    added = 0
    new_chunks_text = []
    new_chunks_meta = []

    for file in files:
        text = extract_text_from_pdf(file.name)
        chunks = chunk_text(text)
        for chk in chunks:
            new_chunks_meta.append((os.path.basename(file.name), chk))
            new_chunks_text.append(chk)
        added += len(chunks)
    
    if added > 0:
        # Compute embeddings locally (fast/free)
        embeddings = EMBEDDER.encode(new_chunks_text, convert_to_tensor=False)
        new_embs = np.array(embeddings)

        RAG_STORE["chunks"].extend(new_chunks_meta)
        
        if RAG_STORE["embeddings"] is None:
            RAG_STORE["embeddings"] = new_embs
        else:
            RAG_STORE["embeddings"] = np.vstack([RAG_STORE["embeddings"], new_embs])
            
    return f"✅ Processed {len(files)} files. Total chunks: {len(RAG_STORE['chunks'])}"

def clear_rag():
    RAG_STORE["chunks"].clear()
    RAG_STORE["embeddings"] = None
    return "🗑️ Memory cleared."

def retrieve(query: str, k: int = 4) -> List[str]:
    if not RAG_STORE["embeddings"] or len(RAG_STORE["chunks"]) == 0:
        return []
    
    q_emb = EMBEDDER.encode(query, convert_to_tensor=False)
    scores = util.cos_sim(q_emb, RAG_STORE["embeddings"])[0].numpy()
    
    # Get top K indices
    top_idx = np.argsort(scores)[-k:][::-1]
    return [RAG_STORE["chunks"][i][1] for i in top_idx]

# ---------------- Generation Logic ----------------
SESSION = {"tokens": []}

def plot_tokens():
    if not SESSION["tokens"]:
        return None
    df = pd.DataFrame(SESSION["tokens"], columns=["Time", "In", "Out"])
    fig, ax = plt.subplots(figsize=(5, 2), facecolor=BG)
    ax.plot(df["Time"], df["In"], 'o-', color=ACCENT_A, label="In", markersize=4)
    ax.plot(df["Time"], df["Out"], 'o-', color=ACCENT_B, label="Out", markersize=4)
    ax.legend(facecolor=CARD_BG, labelcolor=TEXT, fontsize="small")
    ax.set_facecolor(BG)
    ax.tick_params(colors=TEXT, labelsize=8)
    for spine in ax.spines.values():
        spine.set_color(TEXT)
    plt.xticks([]) # Hide x-axis labels to keep it clean
    plt.tight_layout()
    return fig

def transcribe_audio(audio_path: str) -> str:
    if not HAVE_WHISPER or not audio_path:
        return ""
    segments, info = whisper_model.transcribe(audio_path, beam_size=1)
    return " ".join([seg.text for seg in segments]).strip()

def generate(message, history, model_choice, temperature, use_rag):
    if not HAVE_GEMINI or "PASTE_YOUR_KEY" in API_KEY:
        return "❌ Error: Please open the Python script and paste your Google API Key in line 39.", ""

    start_time = time.time()
    
    # 1. Retrieve Context (RAG)
    context_chunks = retrieve(message, k=3) if use_rag else []
    context_str = "\n\n".join([f"[Context]: {c}" for c in context_chunks])
    
    # 2. Setup Model
    model_name = MODEL_OPTIONS.get(model_choice, "gemini-2.5-flash")
    model = genai.GenerativeModel(model_name)
    
    # 3. Build Chat History (The Fix)
    gemini_history = []
    
    # System Prompt Injection
    system_instruction = "You are a helpful AI assistant."
    if context_str:
        system_instruction += f"\n\nREFERENCE DATA:\n{context_str}\n\nInstruction: Answer using the data above if relevant."
        gemini_history.append({"role": "user", "parts": [system_instruction]})
        gemini_history.append({"role": "model", "parts": ["Understood. I will use the reference data."]})
    
    # Correctly parsing the "messages" format (List of Dictionaries)
    for msg in history:
        role = "user" if msg["role"] == "user" else "model"
        # Handle cases where content might be None (rare)
        content = msg.get("content") or ""
        gemini_history.append({"role": role, "parts": [str(content)]})
        
    # Add the current user message
    gemini_history.append({"role": "user", "parts": [message]})

    try:
        response = model.generate_content(
            gemini_history,
            generation_config=genai.types.GenerationConfig(
                temperature=temperature,
                max_output_tokens=2048
            )
        )
        reply = response.text
        
        try:
            tokens_in = response.usage_metadata.prompt_token_count
            tokens_out = response.usage_metadata.candidates_token_count
        except:
            tokens_in, tokens_out = 0, 0

    except Exception as e:
        reply = f"⚠️ API Error: {str(e)}\n\n*Tip: Check if your API key is valid.*"
        tokens_in, tokens_out = 0, 0

    duration = round(time.time() - start_time, 2)
    SESSION["tokens"].append((datetime.now().strftime("%H:%M:%S"), tokens_in, tokens_out))

    analytics = f"⏱ {duration}s | 🪙 In: {tokens_in} / Out: {tokens_out} | 📚 RAG: {len(context_chunks)} chunks"
    return reply, analytics
# ---------------- UI ----------------
css = f"""
body {{ background: {BG}; color: {TEXT}; }}
.gradio-container {{ max-width: 1200px !important; margin: auto; }}
"""

with gr.Blocks(css=css, theme=gr.themes.Soft(), title=BRAND_NAME) as demo:
    gr.Markdown(f"# ☁️ {BRAND_NAME}\nRunning on Google Gemini • Lightweight Mode")
    
    with gr.Row():
        with gr.Column(scale=3):
            chatbot = gr.Chatbot(height=550, type="messages")
            msg = gr.Textbox(placeholder="Type a message...", show_label=False)
            
            with gr.Row():
                submit = gr.Button("Send", variant="primary")
                clear = gr.Button("Clear")
            
            with gr.Accordion("🎤 Voice Input (Optional)", open=False):
                audio_in = gr.Audio(sources=["microphone"], type="filepath")
                tts_chk = gr.Checkbox(label="Read response aloud", value=False)

        with gr.Column(scale=1):
            with gr.Group():
                gr.Markdown("### ⚙️ Settings")
                model_dd = gr.Dropdown(list(MODEL_OPTIONS.keys()), value=DEFAULT_MODEL, label="Model")
                temp = gr.Slider(0.0, 1.0, 0.7, label="Creativity")
                
                gr.Markdown("### 📂 RAG (Document Chat)")
                rag_files = gr.File(file_count="multiple", label="Upload PDFs", type="filepath")
                rag_stat = gr.Markdown("No documents loaded.")
                clear_rag_btn = gr.Button("Clear Documents")

            with gr.Group():
                gr.Markdown("### 📊 Live Analytics")
                analytics_out = gr.Markdown("Waiting for input...")
                plot_out = gr.Plot(label="Token Usage")
                audio_out = gr.Audio(label="TTS Output", autoplay=True, visible=False)

    def respond(message, audio, history, model, temp, rag_files, tts_on):
        # 1. Handle Audio
        if audio:
            transcript = transcribe_audio(audio)
            if transcript:
                message = transcript
        
        if not message.strip():
            return history, "Empty message", None, None
            
        # 2. Generate
        reply, stats = generate(message, history, model, temp, use_rag=(rag_files is not None))
        
        # 3. Update History (Gradio 4.0+ style)
        # history is list of dicts or list of lists. We assume list of tuples here for conversion
        # but Chatbot(type="messages") expects [{"role": "user", "content": "x"}, ...]
        # Let's stick to simple tuple handling for compatibility
        history.append({"role": "user", "content": message})
        history.append({"role": "assistant", "content": reply})

        # 4. TTS
        out_audio = None
        if tts_on and HAVE_GTTS:
            try:
                tts = gTTS(reply, lang="en")
                fd, path = tempfile.mkstemp(suffix=".mp3", dir=TMP_DIR)
                os.close(fd)
                tts.save(path)
                out_audio = path
            except:
                pass

        return history, stats, plot_tokens(), gr.Audio(value=out_audio, visible=True)

    submit.click(respond, [msg, audio_in, chatbot, model_dd, temp, rag_files, tts_chk], [chatbot, analytics_out, plot_out, audio_out])
    msg.submit(respond, [msg, audio_in, chatbot, model_dd, temp, rag_files, tts_chk], [chatbot, analytics_out, plot_out, audio_out])
    
    clear.click(lambda: [], None, chatbot)
    rag_files.upload(add_pdfs_to_rag, rag_files, rag_stat)
    clear_rag_btn.click(clear_rag, outputs=rag_stat)

if __name__ == "__main__":
    demo.launch()
