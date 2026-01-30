import os
import streamlit as st
from dotenv import load_dotenv
from endee import Endee, Precision
from endee.exceptions import NotFoundException
from groq import Groq

from utils import extract_chunks, get_embedding, get_embeddings


# Setup

load_dotenv()

client = Endee()  # connects to local Dockerized Endee
groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))

EMBEDDING_DIM = 384
BATCH_SIZE = 16

st.set_page_config(page_title="RAG Chatbot", layout="wide")


# Sidebar: Document Ingestion

with st.sidebar:
    st.header("📁 Knowledge Base")

    index_name = st.text_input(
        "Index Name",
        placeholder="e.g. index1"
    )

    if index_name:
        index_name = index_name.strip()

    uploaded_file = st.file_uploader("Upload a PDF", type="pdf")

    index_clicked = st.button("Index Document")

    if index_clicked:
        if not index_name:
            st.warning("Please enter an index name.")
        elif not uploaded_file:
            st.warning("Please upload a PDF.")
        else:
            with st.spinner("Indexing document..."):
                try:
                    client.create_index(
                        name=index_name,
                        dimension=EMBEDDING_DIM,
                        space_type="cosine",
                        precision=Precision.INT8D
                    )
                except Exception:
                    # Index already exists
                    pass

                index = client.get_index(index_name)

                chunks = extract_chunks(uploaded_file)

                for i in range(0, len(chunks), BATCH_SIZE):
                    batch_chunks = chunks[i: i + BATCH_SIZE]
                    embeddings = get_embeddings(batch_chunks)

                    records = []
                    for j, emb in enumerate(embeddings):
                        records.append({
                            "id": f"{index_name}_doc_{i + j}",
                            "vector": emb,
                            "meta": {"text": batch_chunks[j]}
                        })

                    index.upsert(records)

                st.success(f"Indexed {len(chunks)} chunks successfully.")
                st.info("Data is stored locally in the Endee volume.")


# Main Chat Interface

st.title("🤖 Rag Assistant")
st.markdown("Ask questions based on your uploaded documents.")

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])


# Chat Input

if prompt := st.chat_input("What would you like to know?"):
    if not index_name:
        st.error("Please enter an index name in the sidebar.")
        st.stop()

    # Safe index access
    try:
        index = client.get_index(index_name)
    except NotFoundException:
        st.warning(
            f"Index **'{index_name}'** does not exist.\n\n"
            "Please upload a document and index it first."
        )
        st.stop()

    st.session_state.messages.append(
        {"role": "user", "content": prompt}
    )

    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Searching knowledge base..."):
            query_vector = get_embedding(prompt)

            results = index.query(
                vector=query_vector,
                top_k=3
            )

            if not results:
                st.info("No relevant content found in this index.")
                st.stop()

            context_text = "\n\n".join(
                res["meta"]["text"] for res in results
            )

            system_prompt = f"""
You are a helpful assistant.
Use ONLY the context below to answer the user's question.
If the answer is not present, say you don't know.

Context:
{context_text}
"""

            completion = groq_client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.2
            )

            response = completion.choices[0].message.content
            st.markdown(response)

    st.session_state.messages.append(
        {"role": "assistant", "content": response}
    )
