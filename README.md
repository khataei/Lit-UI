# ✨ Lit-UI

A lightweight, Streamlit-based multi-LLM web interface for **OpenAI**, **Azure OpenAI**, and **OpenRouter**.  
Chat with powerful language models, upload files (CSV, TXT, images) for contextual AI responses, customize prompts, and manage your conversation history — all in one elegant UI.

---

## 🚀 Features
- 💬 **Multi-LLM Chat** – Switch instantly between OpenAI, Azure OpenAI, and OpenRouter models
- 📝 **Custom System Prompts** – Change AI behavior dynamically from the UI
- 📁 **File Upload Support** – CSV, TXT, and Image uploads for in-context AI responses
- 💾 **Conversation History** – Save, load, search, and delete chats
- 🗑️ **Trash Management** – Restore or permanently remove old conversations
- 📥 **Export Chats** – Download conversations as plain text
- 🔄 **Automatic Saving** – Optional auto-save after each assistant reply
- 🛡️ **Safe Rendering** – Sanitized AI responses to prevent HTML injection
- 🎯 **Minimal Setup** – Just plug in your API keys and start chatting

---

## 📦 Installation

1. **Clone the repository**
```bash
git clone https://github.com/YOUR_USERNAME/Lit-UI.git
cd Lit-UI
```

2. **Install dependencies**
```bash
pip install -r requirements.txt
```

3. **Set up your API keys**
Copy config_example.cfg → config.cfg
Edit config.cfg and add your real API keys.
cp config_example.cfg config.cfg

## ▶️ Usage
Run the app from the project folder:

`streamlit run app.py`

Open in your browser:
http://localhost:8501

## ⚙️ Config File Example
config.cfg (not committed to GitHub — keep your keys secret)

[API_KEYS]
GPT4O_AZURE_API_KEY = your_gpt4o_key
GPT4O_AZURE_API_URL = https://your-azure-endpoint
GPT4O_AZURE_API_VERSION = 2024-xx-xx

GPT4.1_AZURE_API_KEY = your_gpt41_key
GPT4.1_AZURE_API_URL = https://your-azure-endpoint
GPT4.1_AZURE_API_VERSION = 2024-xx-xx

GPT5_API_VERSION = 2024-xx-xx

OR_API_KEY = your_openrouter_key


## 📂 Project Structure
Lit-UI/
├── app.py                 # Main Streamlit app
├── config_example.cfg     # Template config file
├── requirements.txt       # Python dependencies
├── README.md              # This file
├── LICENSE                # License file (MIT recommended)
├── .gitignore             # Ignore secrets & cache files
├── chats/                 # Saved conversation history
└── trash/                 # Deleted chats

## 🛡 Security Notes
API keys are never hardcoded — you must provide your own in config.cfg
config.cfg is ignored via .gitignore so it’s never uploaded to GitHub
AI responses are HTML-sanitized before rendering to prevent malicious injection
Uploaded files are stored locally and never sent anywhere except the AI APIs you use
No arbitrary code execution from user input


## 📌 Roadmap
- 🎙️ Voice input & transcription
- 🔊 Text-to-speech for AI responses
- 🖼 AI image generation integration
- 📊 Richer CSV analytics in responses
- 📜 License


This project is licensed under the MIT License — see the LICENSE file for details.