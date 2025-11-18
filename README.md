# ☁️ CHAT-BOT (Gemini 2.5 + RAG)

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python&logoColor=white)
![Gemini API](https://img.shields.io/badge/Google%20Gemini-2.5%20Flash-8E75B2?logo=google&logoColor=white)
![Gradio](https://img.shields.io/badge/UI-Gradio-F97316?logo=gradio&logoColor=white)
![RAG](https://img.shields.io/badge/RAG-Enabled-green)

A lightweight, cloud-powered AI chatbot capable of **Document Q&A (RAG)**, **Voice Interaction**, and **Real-time Analytics**. Built with Python, Gradio, and Google's latest **Gemini 2.5 Flash** model.

---

## ✨ Key Features

* **🧠 Smart Cloud Intelligence:** Powered by Google's **Gemini 2.5 Flash** & **Gemini 3.0 Pro (Preview)** for high-speed, accurate responses.
* **📂 RAG (Document Chat):** Upload PDF documents and chat with them instantly. The bot uses `SentenceTransformer` (CPU-optimized) to find relevant answers within your files.
* **🎤 Full Voice Support:**
    * **Input:** Speak to the bot using **Faster-Whisper** (Tiny model, RAM efficient).
    * **Output:** The bot replies with voice using **gTTS** (Google Text-to-Speech).
* **📊 Live Analytics:** Real-time dashboard showing **Token Usage**, **Response Time**, and **RAG Chunk Hits**.
* **🚀 Lightweight Mode:** Optimized to run on low-RAM machines (no heavy local LLMs required).

---

## 🛠️ Installation

### 1. Clone the Repository
```bash
git clone [https://github.com/your-username/your-repo-name.git](https://github.com/your-username/your-repo-name.git)
cd your-repo-name
````

### 2\. Install Dependencies

This project uses lightweight libraries to ensure it runs smoothly on standard hardware.

```bash
pip install gradio google-generativeai sentence-transformers pypdf2 gtts faster-whisper pandas matplotlib
```

### 3\. Get Your API Key

You need a free Google Gemini API key to run the model.

1.  Go to [Google AI Studio](https://aistudio.google.com/app/apikey).
2.  Click **"Create API Key"**.
3.  Copy the key.

-----

## ⚙️ Configuration

Open `chatbot_final_v2.py` in your code editor and look for **line 39**. Paste your API key inside the quotes:

```python
# ==========================================
# 👇👇 PASTE YOUR API KEY BELOW 👇👇
# ==========================================
API_KEY = "AIzaSyD..._YOUR_ACTUAL_KEY_HERE_..."
```

> **⚠️ Security Note:** If you plan to share this code publicly (e.g., on GitHub), **do not** commit your API key. Instead, use Environment Variables (`os.getenv`).

-----

## 🚀 Usage

Run the chatbot with the following command:

```bash
python chatbot_final_v2.py
```

Once running, click the local URL provided in the terminal (usually `http://127.0.0.1:7860`).

### **How to use the Features:**

1.  **Chat:** Type any question in the text box.
2.  **Document Chat (RAG):**
      * Click "Upload PDFs" on the right panel.
      * Upload one or more PDF files.
      * Ask questions like *"Summarize the document"* or *"What does the file say about X?"*.
3.  **Voice Mode:**
      * Click the Microphone icon to record your question.
      * Check "Read response aloud" to hear the bot speak back.

-----

## 📂 Project Structure

```plaintext
├── chatbot_final_v2.py    # Main application file
├── tmp_chat_logs/         # Temporary folder for audio files (auto-cleans)
├── README.md              # Documentation
└── requirements.txt       # (Optional) List of dependencies
```

-----

## 🔮 Future Roadmap

  - [ ] Add Docker support for easy deployment.
  - [ ] Add "Save Chat History" feature.
  - [ ] Support for Word (.docx) and Text (.txt) files.
  - [ ] Dark/Light mode toggle.

-----

## 🤝 Contributing

Contributions are welcome\! Please open an issue or submit a pull request if you have suggestions for improvements.

-----

**Built with ❤️ using Python & Gemini**

