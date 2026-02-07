import streamlit as st
import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent
from langchain_core.messages import HumanMessage


# --- 1. SETUP & TOOLS ---
st.set_page_config(page_title="Medical AI System", page_icon="🏥", layout="wide")
st.title("🏥 Private Medical AI System")

# Define the Tools for the Agent
@tool
def check_vitals(systolic: int, diastolic: int, heart_rate: int) -> str:
    """Analyzes vital signs. Returns 'Normal', 'Elevated', or 'Critical'."""
    risk = "Normal"
    if systolic > 140 or diastolic > 90:
        risk = "Elevated"
    if systolic > 180 or diastolic > 120 or heart_rate > 100:
        risk = "Critical"
    return f"Vitals Analysis: {risk} Risk. (BP: {systolic}/{diastolic}, HR: {heart_rate})"

@tool
def write_referral_letter(patient_name: str, diagnosis: str, recommendation: str) -> str:
    """Writes a referral letter to disk. Use ONLY if risk is Critical."""
    filename = f"Referral_{patient_name.replace(' ', '_')}.txt"
    content = f"""
    URGENT MEDICAL REFERRAL
    -----------------------
    Patient: {patient_name}
    Date: 2026-02-07
    
    Diagnosis: {diagnosis}
    
    Recommendation:
    {recommendation}
    
    Signed,
    AI Triage Agent
    """
    with open(filename, "w") as f:
        f.write(content)
    return f"✅ Referral letter saved to {filename}"

tools = [check_vitals, write_referral_letter]

# --- 2. SIDEBAR (Configuration) ---
with st.sidebar:
    st.header("⚙️ System Status")
    st.success("🟢 Model: Medical-Llama-3-8B")
    st.success("🟢 Device: NVIDIA RTX 3050 Ti")
    st.info("🔒 Mode: Offline / Private")

# --- 3. MAIN TABS ---
tab1, tab2 = st.tabs(["📚 Knowledge Base (RAG)", "🚑 Clinical Triage (Agent)"])

# =========== TAB 1: RAG (Chat with PDF) ============
with tab1:
    st.header("Chat with Medical Guidelines")
    uploaded_file = st.file_uploader("Upload PDF (e.g., WHO Guidelines)", type="pdf")
    
    if uploaded_file:
        save_path = os.path.join("data", uploaded_file.name)
        with open(save_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
            
        if st.button("🧠 Ingest Document"):
            with st.spinner("Reading and memorizing..."):
                loader = PyPDFLoader(save_path)
                docs = loader.load()
                splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
                chunks = splitter.split_documents(docs)
                embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
                vector_db = Chroma.from_documents(chunks, embeddings, persist_directory="vectorstore/")
                st.success("✅ Knowledge Updated!")

    # Chat Interface
    if "messages" not in st.session_state:
        st.session_state.messages = []

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    if prompt_text := st.chat_input("Ask a question about the guidelines..."):
        st.session_state.messages.append({"role": "user", "content": prompt_text})
        with st.chat_message("user"):
            st.markdown(prompt_text)

        with st.chat_message("assistant"):
            # RAG Logic
            embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
            vector_db = Chroma(persist_directory="vectorstore/", embedding_function=embeddings)
            retriever = vector_db.as_retriever(search_kwargs={"k": 2})
            llm = ChatOpenAI(base_url="http://localhost:1234/v1", api_key="lm-studio", model="medical-llama-3-8b", temperature=0.3)
            
            rag_chain = (
                {"context": retriever, "question": RunnablePassthrough()}
                | ChatPromptTemplate.from_template("Answer based on context:\n{context}\n\nQuestion: {question}")
                | llm
                | StrOutputParser()
            )
            response = rag_chain.invoke(prompt_text)
            st.markdown(response)
            st.session_state.messages.append({"role": "assistant", "content": response})

# =========== TAB 2: AGENT (Clinical Triage) ============
with tab2:
    st.header("Autonomous Clinical Triage")

    col1, col2 = st.columns(2)
    with col1:
        p_name = st.text_input("Patient Name", "John Doe")
        systolic = st.number_input("Systolic BP (mmHg)", value=120)
        diastolic = st.number_input("Diastolic BP (mmHg)", value=80)
    with col2:
        heart_rate = st.number_input("Heart Rate (BPM)", value=72)
        symptoms = st.text_area("Symptoms / Notes", "Patient complains of mild chest pain.")

    if st.button("🚑 Run Triage Analysis"):
        with st.spinner("Dr. Llama is analyzing vitals..."):
            # LLM
            llm = ChatOpenAI(
                base_url="http://localhost:1234/v1",
                api_key="lm-studio",
                model="medical-llama-3-8b",
                temperature=0
            )

            # LangGraph Agent (REPLACEMENT for AgentExecutor)
            agent = create_react_agent(
                model=llm,
                tools=tools
            )

            # User message
            user_message = f"""
            Patient: {p_name}
            Vitals: BP {systolic}/{diastolic}, HR {heart_rate}
            Symptoms: {symptoms}

            Analyze vitals using tools. If critical, generate a referral letter.
            """

            result = agent.invoke(
                {"messages": [HumanMessage(content=user_message)]}
            )

            final_answer = result["messages"][-1].content

            st.info("📋 Agent Output")
            st.write(final_answer)

            # Check file creation
            referral_file = f"Referral_{p_name.replace(' ', '_')}.txt"
            if os.path.exists(referral_file):
                st.success("🚨 CRITICAL: Referral Letter Generated")
                with open(referral_file, "r") as f:
                    st.download_button(
                        "Download Referral Letter",
                        f,
                        file_name=referral_file
                    )
    