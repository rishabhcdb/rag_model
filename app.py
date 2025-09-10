import os
import tempfile
import sqlite3
import streamlit as st
from hi_res import ChatPDF

# --- DB Utils ---
def get_logs():
    conn = sqlite3.connect("rag_logs.db")
    cursor = conn.cursor()
    cursor.execute("SELECT question, answer, created_at, doc_name FROM rag_logs ORDER BY id DESC LIMIT 50")
    rows = cursor.fetchall()
    conn.close()
    return rows


# --- Streamlit frontend ---
def main():
    st.set_page_config(page_title="ChatPDF - Legal Document Q&A", layout="wide")

    # Initialize ChatPDF
    if "chat_pdf" not in st.session_state:
        st.session_state.chat_pdf = ChatPDF()

    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Sidebar
    with st.sidebar:
        st.header("PDF Management")
        uploaded_file = st.file_uploader("Upload a PDF", type="pdf")

        if st.button("Clear Session"):
            st.session_state.chat_pdf.clear()
            st.session_state.messages = []
            st.session_state.pop("pdf_processed", None)  # reset flag
            st.success("Session cleared!")

        st.markdown("---")
        st.subheader("⚡ Options")
        tab_choice = st.radio("Choose view:", ["Chat", "History"])

    # Handle upload 
    if uploaded_file and "pdf_processed" not in st.session_state:
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                tmp_file.write(uploaded_file.read())
                tmp_file_path = tmp_file.name

            with st.spinner("Processing PDF..."):
                st.session_state.chat_pdf.ingest(tmp_file_path)
                st.session_state.pdf_processed = True
                st.session_state.doc_name = uploaded_file.name  # save doc name
                st.success(f"PDF processed successfully! ({uploaded_file.name})")

            os.unlink(tmp_file_path)
        except Exception as e:
            st.error(f"Error processing PDF: {str(e)}")


    # Chat tab
    if tab_choice == "Chat":
        st.title("Chat with your Legal Docs")

        # Render past conversation
        for role, content in st.session_state.messages:
            with st.chat_message(role):
                st.markdown(content)

        # Bottom input box
        if user_input := st.chat_input("Ask a question..."):
            st.session_state.messages.append(("user", user_input))
            with st.chat_message("user"):
                st.markdown(user_input)

            with st.spinner("Thinking..."):
                response = st.session_state.chat_pdf.ask(user_input)

            # Log into DB with doc_name
            from db_utils import log_interaction
            log_interaction(user_input, response, st.session_state.get("doc_name", "Unknown"))

            st.session_state.messages.append(("assistant", response))
            with st.chat_message("assistant"):
                st.markdown(response)


    # History tab
    elif tab_choice == "History":
        st.subheader("Past Queries")
        logs = get_logs()
        if logs:
            for q, a, ts, doc in logs:
                st.markdown(f"**[{doc}]**")
                st.markdown(f"[{ts}]")
                st.markdown(f"**Question:**")
                st.markdown(f"{q}")
                st.markdown("**Answer:**")
                st.markdown(f" {a}")
                st.markdown("---")
        else:
            st.info("No history found yet!")


if __name__ == "__main__":
    main()
