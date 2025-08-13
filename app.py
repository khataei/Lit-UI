from openai import OpenAI, AzureOpenAI
import streamlit as st
import configparser
import pandas as pd
import json
import datetime
import os  
import re
import shutil
import base64
import html


page_icon = '3.jpeg' #'🤖' 
page_layout = 'wide'
page_title = 'CarfaxGPT'
st.set_page_config(page_title=page_title, page_icon=page_icon, layout=page_layout)

# Read API key from config file
config = configparser.ConfigParser()
config.read('config.cfg')
api_key = config.get('API_KEYS', 'GPT4O_AZURE_API_KEY')
api_key_4_1 = config.get('API_KEYS', 'GPT4.1_AZURE_API_KEY')
api_url = config.get('API_KEYS', 'GPT4O_AZURE_API_URL')
api_url_4_1 = config.get('API_KEYS', 'GPT4.1_AZURE_API_URL')
api_version= config.get('API_KEYS', 'GPT4O_AZURE_API_VERSION')
api_version_4_1= config.get('API_KEYS', 'GPT4.1_AZURE_API_VERSION')
api_version_5= config.get('API_KEYS', 'GPT5_API_VERSION')
api_key_or= config.get('API_KEYS', 'OR_API_KEY')
deployment = None


# st.title("💬 Carfax Canada Document Chat Assistant")
# st.caption("🚀 Upload your CSV or TXT files and chat about them!")
st.title("🚀 Chat Assistant")

# --- Ensure chats folder exists ---
CHATS_FOLDER = "chats"
TRASH_FOLDER = "trash"
os.makedirs(CHATS_FOLDER, exist_ok=True)
os.makedirs(TRASH_FOLDER, exist_ok=True)


# --- Utility to create timestamp string for filenames ---
def get_timestamp_str():
    return datetime.datetime.now().strftime('%Y_%m-%d--%H-%M')

# Move file to trash
def move_to_trash(filename):
    source_path = os.path.join(CHATS_FOLDER, filename)
    trash_path = os.path.join(TRASH_FOLDER, filename)
    if os.path.exists(source_path):
        shutil.move(source_path, trash_path)
        return True
    return False


def encode_image_file(uploaded_file):
    return base64.b64encode(uploaded_file.read()).decode("utf-8")


def sanitize_filename(name: str):
    # Allow only alphanumeric, dashes, and underscores; lowercase it.
    name = name.strip().lower()
    name = re.sub(r'[^a-z0-9_\- ]', '', name)
    name = name.replace(' ', '_')
    # If result is empty, fallback to timestamp.
    return name if name else get_timestamp_str()

def generate_chat_filename(client, deployment, first_user_message):
    prompt = (
        "Generate a short, descriptive, file-system-safe filename based on user's message."
        "Only output the file name, without extension or extra punctuation.\n"
        'Example: carfax_report_questions, policy_analysis, csv_file_summary\n\n'
        f'First user message: "{first_user_message}"'
    )
    try:
        response = client.chat.completions.create(
            model=deployment,
            messages=[
                {"role": "system", "content": (
                    "You help select short, file-system-safe summary filenames for conversations. "
                    "You always reply with just a descriptive short file name and nothing else."
                )},
                {"role": "user", "content": prompt}
            ],
            stream=False,
            max_tokens=30,  # keep very short
            temperature=0.3
        )
    except:
        response = client.chat.completions.create(
            model=deployment,
            messages=[
                {"role": "system", "content": (
                    "You help select short, file-system-safe summary filenames for conversations. "
                    "You always reply with just a descriptive short file name and nothing else."
                )},
                {"role": "user", "content": prompt}
            ],
            stream=False,
            max_completion_tokens=30,  # gpt5 is different
            # temperature=0.3
        )
        
    # The LLM reply is expected to be a single string (possibly with or without quotes)
    raw_name = response.choices[0].message.content.strip().replace('"', '').replace("'", '')
    name = sanitize_filename(raw_name)
    
    # Add date & time prefix
    timestamp = get_timestamp_str()
    return f"{name}_{deployment[:6]}.json"

def rename_conversation(old_filename, new_name):
    """Renames the whole conversation file name to a user-chosen name."""
    # Remove leading/trailing spaces
    new_name = new_name.strip()
    if not new_name:
        return False, "Name cannot be empty."

    # Ensure .json extension
    if not new_name.lower().endswith(".json"):
        new_name += ".json"

    # Sanitize minimally to avoid OS conflicts
    new_name = re.sub(r'[<>:"/\\|?*]', '', new_name)  # Remove illegal filename chars

    old_path = os.path.join(CHATS_FOLDER, old_filename)
    new_path = os.path.join(CHATS_FOLDER, new_name)

    if os.path.exists(new_path):
        return False, "A conversation with this name already exists."

    try:
        os.rename(old_path, new_path)
        if "chat_filename" in st.session_state and st.session_state["chat_filename"] == old_filename:
            st.session_state["chat_filename"] = new_name
        return True, new_name
    except Exception as e:
        return False, str(e)

# --- Adjust save_conversation to incorporate this ---
def save_conversation(messages, client, deployment):
    # Check if we're already tracking a filename
    filename = st.session_state.get('chat_filename')
    if not filename:
        # Generate a name and store it
        first_user_message = next(
            (msg['content'] for msg in messages if msg['role'] == 'user'),
            "chat"
        )
        filename = generate_chat_filename(client, deployment, first_user_message)
        st.session_state['chat_filename'] = filename
    filepath = os.path.join(CHATS_FOLDER, filename)
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(messages, f, ensure_ascii=False, indent=2)
    return filename


# --- Utility to list saved conversations in the chats folder, sorted by date descending ---
def list_saved_conversations():
    files = [f for f in os.listdir(CHATS_FOLDER) if f.endswith(".json")]
    files.sort(key=lambda f: os.path.getmtime(os.path.join(CHATS_FOLDER, f)), reverse=True)
    return files

# --- Utility to load a conversation from file ---
def load_conversation(filename):
    filepath = os.path.join(CHATS_FOLDER, filename)
    with open(filepath, "r", encoding="utf-8") as f:
        messages = json.load(f)
    st.session_state['chat_filename'] = filename
    return messages

def read_file_content(uploaded_file):
    if uploaded_file.type == "text/csv":
        df = pd.read_csv(uploaded_file)
        return df.to_string()
    elif uploaded_file.type == "text/plain":
        return uploaded_file.getvalue().decode('utf-8')
    return None
def list_trash_conversations():
    files = [f for f in os.listdir(TRASH_FOLDER) if f.endswith(".json")]
    files.sort(key=lambda f: os.path.getmtime(os.path.join(TRASH_FOLDER, f)), reverse=True)
    return files

def restore_conversation(filename):
    """Move the file from trash back to chats."""
    source_path = os.path.join(TRASH_FOLDER, filename)
    target_path = os.path.join(CHATS_FOLDER, filename)
    if os.path.exists(source_path):
        if os.path.exists(target_path):
            return False, "A conversation with this name already exists in Chats."
        shutil.move(source_path, target_path)
        return True, None
    return False, "File not found in trash."

def delete_permanently(filename):
    """Delete the file entirely from the trash folder."""
    file_path = os.path.join(TRASH_FOLDER, filename)
    if os.path.exists(file_path):
        os.remove(file_path)
        return True, None
    return False, "File not found."


# Function to format conversation for download
def format_conversation_for_download(messages):
    formatted_text = "Carfax Document Chat Conversation\n"
    formatted_text += f"Generated on: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
    
    for msg in messages:
        role = "Assistant" if msg["role"] == "assistant" else "User"
        formatted_text += f"{role}: {msg['content']}\n\n"
    
    return formatted_text

# +
with st.sidebar:
    # Dictionary mapping display names to actual model names
    model_options = {
        "GPT-4o Turbo": "gpt-4o-2411", 
        # "GPT-4o Mini": "gpt-4o-mini", 
        "O3 Mini": "o3-mini", 
        "GPT-4.1": "gpt-4.1",
        "GPT-5-Chat": "gpt-5-chat",
        "GPT-5-Mini": "gpt-5-mini",
        "O4 Mini High": "o4-mini-high",
        "R1" : "deepseek/deepseek-r1-0528:free",
        "Claude 3.7 Sonnet": "anthropic/claude-3.7-sonnet", 
        "Claude 3.7 Sonnet (Thinking)": "anthropic/claude-3.7-sonnet:thinking",
        "Sonnet 4": "anthropic/claude-sonnet-4",
        "Grok 4": "x-ai/grok-4",
        "Qwen 3": "qwen/qwen3-coder",
    }

    # Display the friendly names in the radio button
    selected_display_name = st.selectbox(
        "Choose LLM Model",
        options=list(model_options.keys()),
        index=3,
        help="Select the model you want to use"
    )
    col1, col2 = st.sidebar.columns(2)

    auto_save = col1.toggle('Auto Save', value=True)
    show_audio = col2.toggle('🔊 Audio', value=False)
    


    # Get the actual model name from the dictionary
    deployment = model_options[selected_display_name]
    
if (deployment == "gpt-4.1") : 
    # need a different api key and url
    client = AzureOpenAI(
    api_key=api_key_4_1,
    azure_endpoint=api_url_4_1,
    api_version=api_version_4_1,
)
elif (deployment=='anthropic/claude-3.7-sonnet' or  # all from openrouter
    deployment=='anthropic/claude-3.7-sonnet:thinking' or 
    deployment=='o4-mini-high' or 
    deployment == 'deepseek/deepseek-r1-0528:free' or
    deployment =='anthropic/claude-sonnet-4' or
    deployment =='qwen/qwen3-coder' or
    deployment =='x-ai/grok-4'):
    client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=api_key_or,
        )
elif (deployment == "gpt-5-chat") or (deployment == "gpt-5-mini"):    # both have the same key as 4.1
    client = AzureOpenAI(
    api_key=api_key_4_1,
    azure_endpoint=api_url_4_1,
    api_version=api_version_5, #only the api version is different, the rest same as 4.1
)
    
else:
    client = AzureOpenAI(
    api_key=api_key,
    azure_endpoint=api_url,
    api_version=api_version,
)
# -

# --- Assignments and initialization ---
if 'file_contents' not in st.session_state:
    st.session_state['file_contents'] = {}

if "messages" not in st.session_state:
    st.session_state["messages"] = [{"role": "assistant",
                                    "content": "Hello! You can upload CSV, TXT files or images and ask me questions about them."}]



        
# Sidebar tabs
tab_convos, tab_settings, tab_uploads, tab_trash = st.sidebar.tabs([
    "💾 History", 
    "🔧 Model Settings", 
    "📁 File Uploads",
    "🗑 Trash"
])

with tab_settings:
    st.markdown("#### System Prompt")
    default_system_prompt = (
        "You are a helpful assistant that understands text, images, and CSV files. "
        "When answering questions, be clear, accurate, and reference uploaded files by their names if relevant. "
        "Always be concise, and when unsure, ask follow-up questions."
    )
    system_prompt = st.text_area(
        "Customize Assistant Behavior (system prompt)",
        value=st.session_state.get('system_prompt', default_system_prompt),
        help="Change how the AI responds. The default prompt encourages helpful, structured, and multimodal behavior."
    )
    st.session_state['system_prompt'] = system_prompt

with tab_uploads:
    uploaded_files = st.file_uploader(
        "Upload your files (CSV, TXT, or Images)",
        type=['csv', 'txt', 'jpg', 'jpeg', 'png', 'gif', 'webp'],
        accept_multiple_files=True
    )

    if uploaded_files:
        for uploaded_file in uploaded_files:
            try:
                if uploaded_file.type in ["text/csv", "text/plain"]:
                    file_content = read_file_content(uploaded_file)
                    if file_content:
                        st.session_state['file_contents'][uploaded_file.name] = file_content

                        with st.expander(f"Preview {uploaded_file.name}"):
                            if uploaded_file.type == "text/csv":
                                uploaded_file.seek(0)
                                df = pd.read_csv(uploaded_file)
                                st.dataframe(df)
                            else:
                                st.text(file_content[:1000] + "..." if len(file_content) > 1000 else file_content)

                elif uploaded_file.type.startswith("image/"):
                    encoded_image = encode_image_file(uploaded_file)
                    st.session_state['file_contents'][uploaded_file.name] = {
                        "type": "image",
                        "mime": uploaded_file.type,
                        "base64": encoded_image
                    }

                    with st.expander(f"Preview {uploaded_file.name}"):
                        st.image(uploaded_file)

            except Exception as e:
                st.error(f"Error reading file {uploaded_file.name}: {str(e)}")

        st.success(f"Successfully uploaded {len(uploaded_files)} file(s)!")

with tab_convos:
    # Everything here is ONLY visible under the "History" tab
    st.markdown("### 💾 Saved Conversations")
    saved_files = list_saved_conversations()
    df = pd.DataFrame({
        'Name': [f[:-5] if f.endswith('.json') else f for f in saved_files],  # display sans '.json'
        'File': saved_files  # full filename for actions
    })

    search_term = st.text_input("Search conversations", "")
    try:
        filtered_df = df[df['Name'].str.contains(search_term, case=False, regex=False)]
    except:
        filtered_df = pd.DataFrame()
    if not filtered_df.empty:
        if "editing_index" not in st.session_state:
            st.session_state["editing_index"] = None

        for idx, row in filtered_df.iterrows():
            is_active = (
                "chat_filename" in st.session_state
                and st.session_state["chat_filename"] == row['File']
            )

            cols = st.columns([6, 1, 1, 1])
            with cols[0]:
                if st.session_state["editing_index"] == idx:
                    new_name_input = st.text_input(
                        "Edit name",
                        value=row['Name'],
                        key=f"rename_input_{idx}"
                    )
                    if st.button("💾", key=f"save_{idx}"):
                        success, msg = rename_conversation(row['File'], new_name_input)
                        if success:
                            st.session_state["editing_index"] = None
                            st.rerun()
                        else:
                            st.error(f"Rename failed: {msg}")
                    if st.button("❌", key=f"cancel_{idx}"):
                        st.session_state["editing_index"] = None
                else:
                    display_text = f"{row['Name']} ✅" if is_active else row['Name']
                    st.markdown(display_text)
            with cols[1]:
                if st.button("🔄", key=f"load_{row['File']}"):
                    st.session_state["messages"] = load_conversation(row['File'])
                    st.session_state["chat_filename"] = row['File']
                    st.rerun()
            with cols[2]:
                if st.button("✏️", key=f"edit_{idx}"):
                    st.session_state["editing_index"] = idx
            with cols[3]:
                if st.button("🗑️", key=f"delete_{row['File']}"):
                    moved = move_to_trash(row['File'])
                    if moved:
                        st.success(f"Moved {row['File']} to trash.")
                        if is_active:
                            del st.session_state["chat_filename"]
                        st.rerun()
                    else:
                        st.warning(f"File {row['File']} not found.")
    else:
        st.info("No conversations found or no match for search.")


with tab_trash:
    st.markdown("### 🗑 Deleted Conversations")
    trash_files = list_trash_conversations()
    if trash_files:
        df_trash = pd.DataFrame({
            'Name': [f[:-5] if f.endswith('.json') else f for f in trash_files],
            'File': trash_files
        })

        search_term_trash = st.text_input("Search trashed conversations", "")
        try:
            filtered_trash = df_trash[df_trash['Name'].str.contains(search_term_trash, case=False, regex=False)]
        except:
            filtered_trash = pd.DataFrame()

        if not filtered_trash.empty:
            for idx, row in filtered_trash.iterrows():
                cols = st.columns([5, 1, 1, 1])
                with cols[0]:
                    st.write(row['Name'])

                # PREVIEW
                with cols[1]:
                    if st.button("👁️", key=f"preview_trash_{row['File']}"):
                        filepath = os.path.join(TRASH_FOLDER, row['File'])
                        with open(filepath, "r", encoding="utf-8") as f:
                            messages = json.load(f)
                        st.session_state["messages"] = messages
                        st.session_state["preview_only"] = True  # mark that this isn't restored

                # RESTORE
                with cols[2]:
                    if st.button("♻️", key=f"restore_{row['File']}"):
                        success, msg = restore_conversation(row['File'])
                        if success:
                            st.success(f"Restored {row['File']} back to conversations.")
                            st.rerun()
                        else:
                            st.error(msg)

                # PERMANENT DELETE
                with cols[3]:
                    if st.button("❌", key=f"delete_trash_{row['File']}"):
                        success, msg = delete_permanently(row['File'])
                        if success:
                            st.warning(f"Permanently deleted {row['File']}.")
                            st.rerun()
                        else:
                            st.error(msg)

        else:
            st.info("No trashed conversations found or no match.")

    else:
        st.info("Trash is empty.")


# Add global buttons to the sidebar ("Download", "Clear History") AFTER tabs
col1, col2 = st.sidebar.columns(2)
# Download button
if col1.button("📥 Download Conversation"):
    if len(st.session_state.messages) > 1:
        conversation_text = format_conversation_for_download(st.session_state.messages)
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        st.sidebar.download_button(
            label="Download Chat",
            data=conversation_text,
            file_name=f"carfax_chat_{timestamp}.txt",
            mime="text/plain",
            key="download_conversation"
        )
    else:
        st.sidebar.info("Start a conversation before downloading.")
# Clear button
if col2.button("📝 New Chat"):
    st.session_state.messages = [{"role": "assistant", 
                                "content": "Hello! You can upload CSV or TXT files and ask me questions about them."}]
    st.session_state['file_contents'] = {}
    try:
        del st.session_state['chat_filename']
    except:
        print("No file name")
    st.rerun()

# PRint chats
for idx, msg in enumerate(st.session_state.messages):
    chat_container = st.chat_message(msg["role"])
    chat_container.write(msg["content"])
    if (msg["role"] == "assistant") and (show_audio):
        if chat_container.button("🔊 Listen", key=f"tts_{idx}"):
            print(".", flush=True)  # Just a placeholder
    
# Chat input

if prompt := st.chat_input():
    # if not st.session_state['file_contents']:
    #     st.warning("Please upload at least one file first!")
    
    content_parts = []

    # Add text/csv file contents
    for filename, filedata in st.session_state['file_contents'].items():
        if isinstance(filedata, str):  # It's text data
            content_parts.append({
                "type": "text",
                "text": f"<file_{filename}>\n{filedata}\n</file_{filename}>"
            })
        elif isinstance(filedata, dict) and filedata["type"] == "image":
            content_parts.append({
                "type": "image_url",
                "image_url": {
                    "url": f"data:{filedata['mime']};base64,{filedata['base64']}"
                }
            })

    # User’s actual question goes first
    content_parts.insert(0, {"type": "text", "text": prompt})
    # Add user's question to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user").write(prompt)
    
    # Create a placeholder for the assistant's response
    assistant_response = st.chat_message("assistant")
    message_placeholder = assistant_response.empty()

    # Initialize an empty string to collect the full response
    full_response = ""
    
    print("Starting API call with streaming...")
    
    try:
        # Call the API with streaming enabled
        stream = client.chat.completions.create(
            model=deployment,

            messages=[
                {"role": "system", "content": st.session_state['system_prompt']},  # uses user or default
                *st.session_state.messages[1:],
                {"role": "user", "content": content_parts}
            ],
            stream=True
        )
        
        # print("Stream object created successfully")
        
        chunk_count = 0
        for chunk in stream:
            chunk_count += 1
            # print(f"Processing chunk #{chunk_count}")
            # print(f"Chunk type: {type(chunk)}")
            # print(f"Chunk structure: {dir(chunk)}")
            # print(f"Chunk content: {chunk}")
            
            try:
                # Check if choices exists and has elements
                if not hasattr(chunk, 'choices') or len(chunk.choices) == 0:
                    print(f"No choices in chunk #{chunk_count}")
                    continue
                
                # print(f"Choices length: {len(chunk.choices)}")
                # print(f"First choice: {chunk.choices[0]}")
                
                # Check if delta exists in the first choice
                if not hasattr(chunk.choices[0], 'delta'):
                    print(f"No delta in chunk #{chunk_count}")
                    continue
                
                print(f"Delta: {chunk.choices[0].delta}")
                
                # Check if content exists in delta
                if not hasattr(chunk.choices[0].delta, 'content') or chunk.choices[0].delta.content is None:
                    print(f"No content in delta for chunk #{chunk_count}")
                    continue
                
                # Extract the content from the chunk
                content = chunk.choices[0].delta.content
                print(f"Content: {content}")
                
                # Append the new content to the full response
                full_response += content
                # Update the placeholder with the current state of the full response
                full_response = html.unescape(full_response) # since we are allowing unsafe html, let's clean it first
                message_placeholder.markdown(full_response + "▌", unsafe_allow_html=True)
                
            except Exception as e:
                print(f"Error processing chunk #{chunk_count}: {str(e)}")
                print(f"Chunk details: {chunk}")
        
        print(f"Streaming complete. Processed {chunk_count} chunks.")
        
        # Replace the blinking cursor with the final response
        message_placeholder.markdown(full_response)
            
        # Once streaming is complete, add the full response to the chat history
        st.session_state.messages.append({"role": "assistant", "content": full_response})
        # --- SAVE THE CONVERSATION EVERY TIME BOT REPLIES ---
        if auto_save:
            save_conversation(st.session_state["messages"], client, deployment)
        st.rerun()        

            
    except Exception as e:
        print(f"Error during API call: {str(e)}")
        message_placeholder.error(f"An error occurred: {str(e)}")

# with col2:
#     if st.button("🎤", help="Record voice message", key="record_voice_btn"):
#         st.info("Recording feature coming soon!")
if show_audio:
    if st.button("🎤 Record", help="Record and send audio"):
        print("REcord")

