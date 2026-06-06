import os
import warnings
from typing import TypedDict, Annotated, Optional
from dotenv import load_dotenv

# Suppress LangChain and LangGraph serialization warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", message=".*allowed_objects.*")

from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph.message import add_messages
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_groq import ChatGroq
import chromadb
from langchain_huggingface import HuggingFaceEmbeddings

load_dotenv()

# --- 1. Define Persistent Graph State ---
class SupportState(TypedDict):
    messages: Annotated[list, add_messages]
    intent: Optional[str]
    sentiment: Optional[str]
    escalate: bool
    kb_context: Optional[str]
    action_result: Optional[str]

# Initialize LLM & Vector DB Components
llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0)
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
chroma_client = chromadb.PersistentClient(path="./chroma_db")
try:
    collection = chroma_client.get_collection(name="support_kb")
except Exception:
    collection = None # Fallback if run before kb_setup.py

# --- 2. Define Agent Nodes ---

def conversation_manager(state: SupportState):
    """Categorizes the user's intent."""
    last_msg = state["messages"][-1].content
    prompt = f"""Analyze the message: "{last_msg}". 
    Categorize intent as ONE of the following words exactly: FAQ, ORDER_STATUS, REFUND, COMPLAINT, OTHER. 
    Output ONLY the word."""
    response = llm.invoke([HumanMessage(content=prompt)]).content.strip()
    return {"intent": response}

def sentiment_analyzer(state: SupportState):
    """Detects frustration or urgency."""
    last_msg = state["messages"][-1].content
    prompt = f"""Analyze the tone of: "{last_msg}". 
    Reply with ONE word: POSITIVE, NEUTRAL, or NEGATIVE."""
    response = llm.invoke([HumanMessage(content=prompt)]).content.strip()
    return {"sentiment": response}

def knowledge_base_agent(state: SupportState):
    """Retrieves relevant policies via RAG."""
    if state["intent"] in ["FAQ", "REFUND", "COMPLAINT"] and collection:
        last_msg = state["messages"][-1].content
        vector = embeddings.embed_query(last_msg)
        results = collection.query(query_embeddings=[vector], n_results=1)
        if results['documents'] and results['documents'][0]:
            return {"kb_context": results['documents'][0][0]}
    return {"kb_context": "No relevant help docs found."}

def action_agent(state: SupportState):
    """Mocks backend transactions (e.g., API lookups)."""
    last_msg = state["messages"][-1].content.lower()
    # Simulated Order Lookup
    if "order" in last_msg and "123" in last_msg:
        return {"action_result": "System Check: Order 123 is out for delivery today."}
    return {"action_result": None}

def escalation_decider(state: SupportState):
    """Triggers human handoff based on rules."""
    # Escalate if sentiment is negative, or intent is a severe complaint
    if state["sentiment"] == "NEGATIVE" or state["intent"] == "COMPLAINT":
        return {"escalate": True}
    return {"escalate": False}

def response_generator(state: SupportState):
    """Synthesizes context into the final customer reply."""
    
    # 1. Handle Escalate Route
    if state["escalate"]:
        reply = "I understand this is frustrating. I am escalating your ticket to a human support agent immediately. They will be with you shortly."
        return {"messages": [AIMessage(content=reply)]}
    
    # 2. Handle Normal Route (Define the prompt)
    prompt = f"""You are a helpful E-Commerce AI Support Agent. 
    User's latest message: {state['messages'][-1].content}
    Relevant Knowledge Base Policy: {state.get('kb_context', 'No relevant docs found.')}
    Backend System Data: {state.get('action_result', 'No backend data.')}
    
    Draft a polite, direct response resolving their query. Do not mention your internal rules.
    """
    
    # 3. Generate Response
    response = llm.invoke([SystemMessage(content=prompt)])
    
    # 4. FIX: Create a fresh AIMessage with just the string content 
    # to strip out un-copyable network connection metadata
    clean_message = AIMessage(content=response.content)
    return {"messages": [clean_message]}

# --- 3. Build State Machine ---
workflow = StateGraph(SupportState)

workflow.add_node("Manager", conversation_manager)
workflow.add_node("Sentiment", sentiment_analyzer)
workflow.add_node("KnowledgeBase", knowledge_base_agent)
workflow.add_node("Action", action_agent)
workflow.add_node("Escalation", escalation_decider)
workflow.add_node("Responder", response_generator)

# Workflow Routing
workflow.set_entry_point("Manager")
workflow.add_edge("Manager", "Sentiment")
workflow.add_edge("Sentiment", "KnowledgeBase")
workflow.add_edge("KnowledgeBase", "Action")
workflow.add_edge("Action", "Escalation")
workflow.add_edge("Escalation", "Responder")
workflow.add_edge("Responder", END)

# Compile with Memory Checkpointer
memory = MemorySaver()
support_bot = workflow.compile(checkpointer=memory)