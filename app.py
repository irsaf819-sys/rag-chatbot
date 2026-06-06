import os
import streamlit as st
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_classic.memory import ConversationBufferMemory
from langchain_classic.chains import ConversationalRetrievalChain
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_groq import ChatGroq
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
import tempfile, urllib.parse

st.set_page_config(page_title="Multilingual Agentic RAG Chatbot", page_icon="🤖", layout="wide")

GROQ_KEY = "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"
llm = ChatGroq(api_key=GROQ_KEY, model_name="llama-3.3-70b-versatile")

lang_prompts = {
    "English": "Answer in English.",
    "Urdu": "Jawab Urdu mein do.",
    "Arabic": "أجب باللغة العربية.",
    "Spanish": "Responde en español."
}

with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/bot.png", width=70)
    st.title("⚙️ Settings")
    st.markdown("---")
    language = st.selectbox("🌐 Language", ["English", "Urdu", "Arabic", "Spanish"])
    st.markdown("---")
    uploaded_file = st.file_uploader("📄 PDF Upload", type="pdf")
    if uploaded_file:
        st.success(f"✅ {uploaded_file.name}")
    st.markdown("---")
    if st.button("🗑️ Clear Chat"):
        st.session_state.messages = []
        st.session_state.qa = None
        st.rerun()
    if st.button("🗑️ Clear Research"):
        st.session_state.research_messages = []
        st.rerun()

st.title("🤖 Multilingual Agentic RAG Chatbot")

if "messages" not in st.session_state:
    st.session_state.messages = []
if "qa" not in st.session_state:
    st.session_state.qa = None
if "last_file" not in st.session_state:
    st.session_state.last_file = None
if "research_messages" not in st.session_state:
    st.session_state.research_messages = []
if "research_memory" not in st.session_state:
    st.session_state.research_memory = []

if uploaded_file and (st.session_state.last_file != uploaded_file.name):
    with st.spinner("⏳ PDF processing..."):
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as f:
            f.write(uploaded_file.read())
            tmp_path = f.name
        loader = PyMuPDFLoader(tmp_path)
        docs = loader.load()
        splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        chunks = splitter.split_documents(docs)
        embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        db = Chroma.from_documents(chunks, embeddings)
        memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
        st.session_state.qa = ConversationalRetrievalChain.from_llm(
            llm=llm, retriever=db.as_retriever(), memory=memory
        )
        st.session_state.last_file = uploaded_file.name
        st.session_state.messages = []
        st.success("✅ PDF Ready!")

tab1, tab2 = st.tabs(["💬 Chat", "🧠 Research Paper Generator"])

# ---- TAB 1 - CHAT ----
with tab1:
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])

    if prompt := st.chat_input("Apna sawal likho..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.write(prompt)
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                if st.session_state.qa:
                    full_q = f"{lang_prompts[language]} {prompt}"
                    result = st.session_state.qa({"question": full_q})
                    answer = result["answer"]
                else:
                    answer = "⚠️ first upload pdf!"
            st.write(answer)
        st.session_state.messages.append({"role": "assistant", "content": answer})

    if st.session_state.messages:
        st.markdown("---")
        st.markdown("### 📤 Share Chat")
        full_chat = ""
        for msg in st.session_state.messages:
            role = "You" if msg["role"] == "user" else "Bot"
            full_chat += f"{role}: {msg['content']}\n\n"
        st.text_area("📋 Chat History", full_chat, height=150)
        wa_text = urllib.parse.quote(full_chat[:500])
        wa_link = f"https://wa.me/?text={wa_text}"
        email_subject = urllib.parse.quote("My RAG Chatbot Conversation")
        email_body = urllib.parse.quote(full_chat[:1000])
        email_link = f"mailto:?subject={email_subject}&body={email_body}"
        li_link = f"https://www.linkedin.com/sharing/share-offsite/?url=https://glorious-trust-production-5eb0.up.railway.app"
        st.markdown(f'''
        <a href="{wa_link}" target="_blank" style="background:#25D366;color:white;padding:6px 14px;border-radius:8px;text-decoration:none;margin:3px">📱 WhatsApp</a>
        <a href="{email_link}" style="background:#EA4335;color:white;padding:6px 14px;border-radius:8px;text-decoration:none;margin:3px">📧 Email</a>
        <a href="{li_link}" target="_blank" style="background:#0077B5;color:white;padding:6px 14px;border-radius:8px;text-decoration:none;margin:3px">💼 LinkedIn</a>
        ''', unsafe_allow_html=True)

# ---- TAB 2 - RESEARCH ----
with tab2:
    st.markdown("### 🧠 Agentic Research Paper Generator")
    st.info("No need to pdf_write any topic!")

    for msg in st.session_state.research_messages:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])

    research_topic = st.text_input("Topic likho (e.g. AI in Islamic Finance)")

    if st.button("📝 Generate the research paper"):
        if research_topic:
            with st.spinner("Agent is working..."):
                history_context = ""
                for msg in st.session_state.research_memory:
                    history_context += f"{msg['role']}: {msg['content']}\n"

                steps = [
                    f"Write a detailed introduction about {research_topic}.",
                    f"What are the key concepts of {research_topic}?",
                    f"What are the real world applications of {research_topic}?",
                    f"What are the challenges and limitations of {research_topic}?",
                    f"Write a conclusion for a research paper on {research_topic}."
                ]
                sections = ["Introduction", "Key Concepts", "Applications", "Challenges", "Conclusion"]
                paper_content = {}
                progress = st.progress(0)

                for i, (step, section) in enumerate(zip(steps, sections)):
                    full_q = f"{lang_prompts[language]} Previous context: {history_context} Now: {step}"
                    response = llm.invoke(full_q)
                    paper_content[section] = response.content
                    progress.progress((i+1)*20)

                st.session_state.research_memory.append({"role": "user", "content": f"Research on: {research_topic}"})
                st.session_state.research_memory.append({"role": "assistant", "content": str(paper_content)})
                st.session_state.research_messages.append({"role": "user", "content": f"📝 Research: {research_topic}"})
                st.session_state.research_messages.append({"role": "assistant", "content": f"✅ Paper generated on: {research_topic}"})

                pdf_path = tempfile.mktemp(suffix=".pdf")
                c = canvas.Canvas(pdf_path, pagesize=letter)
                w, h = letter
                c.setFont("Helvetica-Bold", 18)
                c.drawString(100, h-60, f"Research Paper: {research_topic}")
                y = h - 100
                for section, content in paper_content.items():
                    c.setFont("Helvetica-Bold", 13)
                    c.drawString(60, y, section)
                    y -= 20
                    c.setFont("Helvetica", 10)
                    words = content.replace("\n", " ").split(" ")
                    line = ""
                    for word in words:
                        if len(line + word) < 90:
                            line += word + " "
                        else:
                            c.drawString(60, y, line)
                            y -= 14
                            line = word + " "
                            if y < 60:
                                c.showPage()
                                y = h - 60
                    if line:
                        c.drawString(60, y, line)
                        y -= 25
                c.save()

                with open(pdf_path, "rb") as f:
                    pdf_bytes = f.read()

                st.download_button("📥 Download the pdf", pdf_bytes,
                                   file_name="research_paper.pdf",
                                   mime="application/pdf")

                paper_text = "\n\n".join([f"{s}:\n{c}" for s, c in paper_content.items()])
                wa_text = urllib.parse.quote(f"Research Paper on {research_topic}:\n\n{paper_text[:400]}")
                wa_link = f"https://wa.me/?text={wa_text}"
                email_subject = urllib.parse.quote(f"Research Paper: {research_topic}")
                email_body = urllib.parse.quote(paper_text[:1000])
                email_link = f"mailto:?subject={email_subject}&body={email_body}"
                li_link = f"https://www.linkedin.com/sharing/share-offsite/?url=https://glorious-trust-production-5eb0.up.railway.app"

                st.markdown(f'''
                <a href="{wa_link}" target="_blank" style="background:#25D366;color:white;padding:6px 14px;border-radius:8px;text-decoration:none;margin:3px">📱 WhatsApp</a>
                <a href="{email_link}" style="background:#EA4335;color:white;padding:6px 14px;border-radius:8px;text-decoration:none;margin:3px">📧 Email</a>
                <a href="{li_link}" target="_blank" style="background:#0077B5;color:white;padding:6px 14px;border-radius:8px;text-decoration:none;margin:3px">💼 LinkedIn</a>
                ''', unsafe_allow_html=True)
                st.success("✅ Research Paper ready!")
        else:
            st.warning("First write about topic!")
