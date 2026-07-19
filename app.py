"""MilkLab RAG Chatbot (S3).

Run locally: streamlit run app.py
Deploy: push to GitHub then Actions deploys to HuggingFace Space

นักศึกษาต้องเติม TODO 5 จุด ใน Session 3 Lab 2.2
"""

import os

from google import genai
from google.genai import types
from dotenv import load_dotenv

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
import streamlit as st


@st.cache_resource
def load_index():
    """TODO 1+2+3: โหลด menu_kb.md, split เป็น chunk, encode ด้วย sentence-transformers,
    สร้าง faiss index. Cache เพราะโหลด model ครั้งแรกใช้เวลา 30 วินาที

    Returns: (model, index, chunks_list)
    """

    model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")

    with open("menu_kb.md", "r", encoding="utf-8") as f:
        text = f.read()

    chunks = [c.strip() for c in text.split("\n\n") if c.strip()]

    embeddings = model.encode(chunks, convert_to_numpy=True)

    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings.astype("float32"))

    return model, index, chunks


def retrieve_top_k(query: str, model, index, chunks: list[str], k: int = 3) -> list[str]:
    """TODO 4: encode query, search index, return top-k chunks"""

    query_embedding = model.encode([query], convert_to_numpy=True)

    distances, indices = index.search(
        query_embedding.astype("float32"),
        k
    )

    results = []

    for idx in indices[0]:
        if idx != -1:
            results.append(chunks[idx])

    return results


def generate_answer(query: str, context_chunks: list[str]) -> str:

    api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")

    if not api_key:
        return "ไม่พบ GOOGLE_API_KEY"

    client = genai.Client(api_key=api_key)

    context = "\n\n".join(context_chunks)

    prompt = f"""
    ตอบจากข้อมูลต่อไปนี้เท่านั้น

    หากไม่มีข้อมูลใน Context ให้ตอบว่า
    "ขออภัย ไม่พบข้อมูลในฐานข้อมูล"

    Context
    --------
    {context}

    คำถาม
    --------
    {query}
    """

    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt,
        config=types.GenerateContentConfig(
            temperature=0.2
        )
    )

    return response.text


def main():
    load_dotenv()
    st.set_page_config(page_title="MilkLab° RAG", page_icon="🥛")
    st.title("MilkLab° RAG Chatbot")
    st.caption("ถามอะไรเกี่ยวกับ MilkLab ได้ ตอบจาก menu_kb.md")

    try:
        model, index, chunks = load_index()
    except NotImplementedError as exc:
        st.error(f"TODO not implemented: {exc}")
        st.stop()

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])

    if prompt := st.chat_input("ถามอะไรเกี่ยวกับ MilkLab"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.write(prompt)

        with st.chat_message("assistant"):
            with st.spinner("กำลังค้นข้อมูล..."):
                context = retrieve_top_k(prompt, model, index, chunks)
                answer = generate_answer(prompt, context)
            st.write(answer)
            with st.expander("Source chunks"):
                for i, c in enumerate(context, 1):
                    st.markdown(f"**[{i}]** {c}")
        st.session_state.messages.append(
            {"role": "assistant", "content": answer})


if __name__ == "__main__":
    main()
