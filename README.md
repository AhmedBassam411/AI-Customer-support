# 🎧 AI Customer Support System with Intelligent Escalation

A highly scalable, multi-agent conversational AI system built to handle Tier-1 customer support inquiries. This platform autonomously resolves repetitive tickets using Retrieval-Augmented Generation (RAG), monitors real-time customer sentiment, and seamlessly escalates complex or frustrated interactions to a human agent—preserving the full conversation state.

## 🚀 Key Features

* **Multi-Agent Orchestration:** Utilizes a Directed Acyclic Graph (DAG) to intelligently route intents, analyze sentiment, retrieve knowledge, and execute actions.
* **Stateful Memory:** Maintains persistent conversation context across interactions using LangGraph checkpointers, allowing the bot to "remember" previous messages in the thread.
* **RAG-Powered Knowledge Base:** Uses ChromaDB and HuggingFace local embeddings (`all-MiniLM-L6-v2`) to instantly retrieve accurate company policies (e.g., Returns, Shipping, Refunds).
* **Human-in-the-Loop (HITL) Escalation:** Automatically detects negative sentiment or complex complaints, safely halting the AI pipeline to initiate a seamless handoff to human support.
* **Decoupled Architecture:** A headless FastAPI backend communicates securely with a reactive Streamlit frontend via REST APIs.

## 🧠 Agentic Workflow

The system is powered by a LangGraph state machine coordinating the following virtual agents:
1. **Conversation Manager:** Categorizes user intent (FAQ, REFUND, COMPLAINT, etc.).
2. **Sentiment Analyzer:** Detects conversational tone to flag urgency or frustration.
3. **Knowledge Base Agent:** Queries the ChromaDB vector store for relevant solutions.
4. **Action Agent:** Mocks backend API transactions (e.g., tracking an order status).
5. **Escalation Decider:** A strict logic gate that triggers human handoff if negative sentiment or high-complexity intents are detected.
6. **Response Generator:** Synthesizes the RAG context and intent into a natural, conversational response.

## 📂 Project Structure

```text
ai-support-system/
├── .env                    # Environment variables (API Keys)
├── requirements.txt        # Python dependencies
├── kb_setup.py             # Script to initialize and populate ChromaDB
├── agents.py               # LangGraph state machine & LLM prompts
├── app.py                  # FastAPI REST backend
└── frontend.py   # Streamlit chat interface
