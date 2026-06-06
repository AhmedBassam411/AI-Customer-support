import streamlit as st
import requests
import uuid

# --- UI Configuration ---
st.set_page_config(page_title="AI Support Escalation System", page_icon="🎧")
st.title("🎧 E-Commerce Support Desk")

# --- Session Management ---
# Generate a unique thread ID for LangGraph memory tracking
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())

# Initialize local chat history for the UI
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Hi! I am the AI Support Assistant. How can I help you today?", "escalated": False}
    ]

# --- Render Chat History ---
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("escalated"):
            st.error("🚨 System Status: Ticket Escalated to Human Agent.")

# --- Chat Input & Execution ---
if prompt := st.chat_input("Type your issue here (e.g., 'What is your return policy?' or 'I am angry!')"):
    
    # 1. Display User Message
    st.session_state.messages.append({"role": "user", "content": prompt, "escalated": False})
    with st.chat_message("user"):
        st.markdown(prompt)

    # 2. Call FastAPI Backend
    with st.chat_message("assistant"):
        with st.spinner("Analyzing intent and searching knowledge base..."):
            try:
                # Send HTTP POST to our new FastAPI route
                resp = requests.post(
                    "http://localhost:8000/api/chat", 
                    json={"session_id": st.session_state.session_id, "message": prompt}
                )
                
                if resp.status_code == 200:
                    data = resp.json()
                    ai_reply = data["response"]
                    is_escalated = data["escalated"]
                    
                    # Display AI output
                    st.markdown(ai_reply)
                    if is_escalated:
                        st.error("🚨 System Status: Ticket Escalated to Human Agent.")
                        
                    # Save to local UI memory
                    st.session_state.messages.append({
                        "role": "assistant", 
                        "content": ai_reply, 
                        "escalated": is_escalated
                    })
                else:
                    st.error(f"Backend Error: {resp.status_code}")
                    
            except requests.exceptions.ConnectionError:
                st.error("Connection Failed. Is the FastAPI backend running on port 8000?")