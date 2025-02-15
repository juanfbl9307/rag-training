import streamlit as st
import requests
import os
st.title("Chatbot with External API")
GENERATION_SERVER_URL = os.environ.get("GENERATION_SERVER_URL")
GENERATION_SERVER_PORT = os.environ.get("GENERATION_SERVER_PORT")
if not GENERATION_SERVER_URL:
    exit("Please set GENERATION_SERVER_URL environment variable")
if not GENERATION_SERVER_PORT:
    exit("Please set GENERATION_SERVER_PORT environment variable")

API_URL = f'{GENERATION_SERVER_URL}:{GENERATION_SERVER_PORT}/chat'  # Replace with your actual API URL

if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# User input
user_input = st.chat_input("Ask me anything...")
if user_input:
    # Append user input to chat history
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    # Make request to external HTTP server
    try:
        response = requests.post(API_URL, json={"message": user_input}, timeout=5)
        response.raise_for_status()
        bot_reply = response.json().get("reply", "No response from server")
    except requests.exceptions.RequestException as e:
        bot_reply = f"Error: {str(e)}"

    # Append bot response to chat history
    st.session_state.messages.append({"role": "assistant", "content": bot_reply})
    with st.chat_message("assistant"):
        st.markdown(bot_reply)