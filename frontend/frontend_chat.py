import streamlit as st
import requests

API_URL = "http://localhost:8000/search"

st.set_page_config(page_title="CV Screener Chat", page_icon="📄", layout="centered")

st.title("Olocau-CV Screener")
st.write("Credits: Txus Blasco")
st.write("https://github.com/TxusBlasco/Olocau-CV")
st.write("Ask a question about the CVs and get an answer with sources.")

# Chat-like input
question = st.text_input("Your question:")

if st.button("Ask") and question.strip():
    try:
        response = requests.get(API_URL, params={"q": question})
        if response.status_code == 200:
            data = response.json()
            st.markdown(f"**Answer:** {data['answer']}")
            if data.get("sources"):
                st.markdown(
                    "**Sources:** " +
                    ", ".join([src.get("file", "Unknown") for src in data["sources"]])
                )
        else:
            st.error(f"Error {response.status_code}: {response.text}")
    except Exception as e:
        st.error(f"Request failed: {e}")
