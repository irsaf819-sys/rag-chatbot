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

st.set_page_config(page_title="RAG Chatbot", page_icon="🤖", layout="wide")

st.markdown("""
<style>
    .main { background-color: #f0f2f6; }
    .share-btn { display: inline-block; padding: 6px 14px; border-radius: 8px;
                 color: white; text-decoration: none; font-size: 13px; margin: 3px; }
    .wa { background-color: #25D366; }
    .li { background-color: #0077B5; }
</style>
""", unsafe_allow_html=True)

GROQ_KEY = os.environ.get("GROQ_API_KEY", "")

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

st.title("🤖 Multilingual RAG Chatbot")
st.markdown("---")

lang_prompts = {
    "English": "Answer in English.",
    "Urdu": "Jawab Urdu mein do.",
    "Arabic": "أجب باللغة العربية.",
    "Spanish": "Responde en español."
}

if "messages" not in st.session_state:
    st.session_state.messages = []
if "qa" not in st.session_state:
    st.session_state.qa = None
if "last_file" not in st.session_state:
    st.session_state.last_file = None

if uploaded_file and (st.session_state.last_file != uploaded_file.name):
    with st.spinner("⏳ PDF processing..."):
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as f:
            f.write(uploaded_file.read())
            tmp_path = f.name
        llm = ChatGroq(api_key=GROQ_KEY, model_name="llama-3.3-70b-versatile")
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
        st.success("✅ Ready!")

# ---- AGENTIC RESEARCH PAPER ----
st.markdown("### 🧠 Agentic AI — Research Paper Generator")
research_topic = st.text_input("Topic likho (e.g. AI in Islamic Finance)")
if st.button("📝 Research Paper Generate Karo"):
    if st.session_state.qa and research_topic:
        with st.spinner("Agent kaam kar raha hai..."):
            steps = [
                f"What is {research_topic}? Give a detailed introduction.",
                f"What are the key concepts and principles of {research_topic}?",
                f"What are the current applications and use cases of {research_topic}?",
                f"What are the challenges and future directions of {research_topic}?",
                f"Write a conclusion for a research paper on {research_topic}."
            ]
            sections = ["Introduction", "Key Concepts", "Applications", "Challenges & Future", "Conclusion"]
            paper_content = {}
            progress = st.progress(0)
            for i, (step, section) in enumerate(zip(steps, sections)):
                full_q = f"{lang_prompts[language]} {step}"
                result = st.session_state.qa({"question": full_q})
                paper_content[section] = result["answer"]
                progress.progress((i+1)*20)

            # PDF banao
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_pdf:
                pdf_path = tmp_pdf.name
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
                if y < 80:
                    c.showPage()
                    y = h - 60
            c.save()

            with open(pdf_path, "rb") as f:
                st.download_button("📥 PDF Download Karo", f,
                                   file_name="research_paper.pdf",
                                   mime="application/pdf")
            st.success("✅ Research Paper ready!")
    else:
        st.warning("Pehle PDF upload karo aur topic likho")

st.markdown("---")

# ---- CHAT ----
st.markdown("### 💬 Chat")
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

if not uploaded_file:
    st.info("👈 Sidebar se PDF upload karo")
else:
    if prompt := st.chat_input("Apna sawal likho..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.write(prompt)
        with st.chat_message("assistant"):
            with st.spinner("Soch raha hun..."):
                full_q = f"{lang_prompts[language]} {prompt}"
                result = st.session_state.qa({"question": full_q})
                answer = result["answer"]
            st.write(answer)
        st.session_state.messages.append({"role": "assistant", "content": answer})

# ---- SHARING ----
if st.session_state.messages:
    st.markdown("---")
    st.markdown("### 📤 Chat Share Karo")

    full_chat = ""
    for msg in st.session_state.messages:
        role = "You" if msg["role"] == "user" else "Bot"
        full_chat += f"{role}: {msg['content']}\n\n"

    # Clipboard
    st.text_area("📋 Chat History", full_chat, height=150)

    # WhatsApp
    wa_text = urllib.parse.quote(full_chat[:500])
    wa_link = f"https://wa.me/?text={wa_text}"

    # LinkedIn
    li_text = urllib.parse.quote(full_chat[:500])
    li_link = f"https://www.linkedin.com/sharing/share-offsite/?url=https://chatbot.app&summary={li_text}"

    st.markdown(f"""
    <a href="{wa_link}" target="_blank" class="share-btn wa">📱 WhatsApp Share</a>
    <a href="{li_link}" target="_blank" class="share-btn li">💼 LinkedIn Share</a>
    """, unsafe_allow_html=True)

    # Email
    email_subject = urllib.parse.quote("My RAG Chatbot Conversation")
    email_body = urllib.parse.quote(full_chat[:1000])
    email_link = f"mailto:?subject={email_subject}&body={email_body}"
    st.markdown(f'<a href="{email_link}" class="share-btn" style="background:#EA4335">📧 Email Share</a>',
                unsafe_allow_html=True)
