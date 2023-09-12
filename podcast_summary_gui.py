import streamlit as st
from podcast_summarizer_v2 import summarize_podcast_transcript
import os
import io
from fpdf import FPDF

# Function to convert text to PDF bytes
def string_to_pdf_bytes(text):
    pdf = FPDF()
    pdf.add_page()
    pdf.add_font("DejaVu", "", "DejaVuSans.ttf", uni=True)
    pdf.set_font("DejaVu", size=12)
    pdf.multi_cell(0, 10, text)

    pdf_buffer = io.BytesIO()
    pdf.output(pdf_buffer)
    pdf_file_bytes = pdf_buffer.getvalue()
    pdf_buffer.close()

    return pdf_file_bytes

# Streamlit app
def main():
    st.title("Podcast Summarizer")

    # Initialize state
    if 'summary' not in st.session_state:
        st.session_state.summary = ""

    # Sidebar for input
    st.sidebar.header("User Input")
    openai_api_key = st.sidebar.text_input("Enter OpenAI API Key")
    google_api_key = st.sidebar.text_input("Enter Google API Key")
    podcast_url = st.sidebar.text_input("Enter Podcast URL")

    # Set the API keys in environment variables
    if openai_api_key and google_api_key:
        os.environ["OPENAI_API_KEY"] = openai_api_key
        os.environ["GOOGLE_API_KEY"] = google_api_key

        # Add a button to initiate processing
        if st.sidebar.button("Summarize"):
            with st.spinner("Generating summary..."):
                # Call function to generate summary
                st.session_state.summary = summarize_podcast_transcript(podcast_url, 'gpt-3.5-turbo', 0.8, 0.2)

    # Show summary
    if st.session_state.summary:
        st.header("Podcast Summary")
        st.write(st.session_state.summary)

        # Generate a PDF and allow the user to download it
        pdf_bytes = string_to_pdf_bytes(st.session_state.summary)
        st.download_button(
            label="Download Summary as PDF",
            data=pdf_bytes,
            file_name="podcast_summary.pdf",
            mime="application/pdf",
        )

if __name__ == "__main__":
    main()
