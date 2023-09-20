import streamlit as st
from podcast_summarizer_v2 import summarize_podcast_transcript, init_vector_store
import os
import io
from fpdf import FPDF
from youtube_video_chapters import fetch_chapters, fetch_video_info

# Constants
ALL_CHAPTERS = "All Chapters"

def string_to_pdf_bytes(podcast_info, summary_text):
    pdf_buffer = io.BytesIO()

    pdf = FPDF(format='A4')  # Explicitly specify A4 format
    pdf.add_page()

    # Add a Unicode font
    pdf.add_font("DejaVu", "", "DejaVuSansCondensed.ttf", uni=True)
    
    # Set Auto Page Break
    pdf.set_auto_page_break(auto=1, margin=15)
    
    # Set Wider Margins
    pdf.set_left_margin(10)  # Set left margin to 10
    pdf.set_right_margin(10)  # Set right margin to 10

    # Use the Unicode font
    pdf.set_font("DejaVu", size=12)

    # Add the Podcast Information
    pdf.cell(0, 10, "Podcast Information:", ln=True)
    pdf.multi_cell(0, 10, podcast_info)
    pdf.cell(0, 10, "-----", ln=True)

    # Add the summary text
    pdf.multi_cell(0, 10, summary_text)
    
    pdf.output(pdf_buffer)

    pdf_file_bytes = pdf_buffer.getvalue()
    pdf_buffer.close()

    return pdf_file_bytes


def set_environment_keys(api_keys):
    for key, value in api_keys.items():
        os.environ[key] = value

def initialize_session_state():
    for var, initial_value in [
        ('summary', []),
        ('chapters_list', []),
        ('video_chapters', {}),
        ('vector_store', None)
    ]:
        if var not in st.session_state:
            st.session_state[var] = initial_value

def get_selected_chapters():
    selected_chapters = st.multiselect(
        "Select Chapters to Summarize",
        st.session_state.chapters_list,
        default=[ALL_CHAPTERS]
    )
    is_invalid_chapter_selection = ALL_CHAPTERS in selected_chapters and len(selected_chapters) > 1

    if is_invalid_chapter_selection:
        st.warning("You can't select 'All Chapters' along with specific chapters.")

    return selected_chapters, not is_invalid_chapter_selection

def display_summaries(summaries):
    for summary_dict in summaries:
        st.markdown(f"**Chapter: {summary_dict['chapter']}**")
        st.write(summary_dict['summary'])
        st.write("---")

def main():
    st.title("Podcast Summarizer")
    initialize_session_state()

    st.sidebar.header("User Input")

    api_keys = {
        "OPENAI_API_KEY": st.sidebar.text_input("Enter OpenAI API Key", type="password"),
        "GOOGLE_API_KEY": st.sidebar.text_input("Enter Google API Key", type="password"),
        "PINECONE_API_KEY": st.sidebar.text_input("Enter Pinecone API Key", type="password"),
        "PINECONE_ENV": st.sidebar.text_input("Enter Pinecone Environment", type="password")
    }

    podcast_url = st.sidebar.text_input("Enter YouTube Podcast URL")

    # Allow the user to select an OpenAI model
    model_choice = st.sidebar.selectbox("Choose OpenAI Model", ["gpt-3.5-turbo", "gpt-4"])

    # Add a sidebar selectbox for summary type
    summary_type = st.sidebar.selectbox("Choose Summary Type", ["paragraph", "bullet points"])

    all_fields_filled = all(api_keys.values()) and podcast_url

    if all_fields_filled:
        st.sidebar.success("All fields are filled!")
        set_environment_keys(api_keys)

    start_button = st.sidebar.button("Start", disabled=not all_fields_filled)

    if start_button and podcast_url:
        # Clear the previous summaries
        st.session_state.summary = []

        with st.spinner("Initializing..."):
            # Get video information (channel title, podcast title, tags)
            video_info = fetch_video_info(podcast_url)
            st.session_state.channelTitle = video_info.get('channelTitle', 'Channel not available')
            st.session_state.title = video_info.get('title', 'Title not available')
            st.session_state.tags = video_info.get('tags', 'Tags not available')

            with st.spinner("Getting podcast transcript..."):
                st.session_state.vector_store = init_vector_store(podcast_url)

            with st.spinner("Fetching podcast chapters..."):
                video_chapters = fetch_chapters(podcast_url)

            st.session_state.video_chapters = video_chapters
            if 'Chapters' in video_chapters:
                st.session_state.chapters_list = [ALL_CHAPTERS] + video_chapters['Chapters']
            else:
                st.error("Issue with video transcript fetching")

    if st.session_state.get('channelTitle') and st.session_state.get('title') and st.session_state.get('tags'):
        st.header("Podcast Information")
        st.write(f"Channel Title: {st.session_state.channelTitle}")
        st.write(f"Title: {st.session_state.title}")
        st.write(f"Tags: {st.session_state.tags}")

    # (The rest of the code remains unchanged)
    if st.session_state.chapters_list:
        st.header("Chapter Selection")
        selected_chapters, valid_selection = get_selected_chapters()

        if st.button("Summarize", disabled=not valid_selection):
            # Clear previous summaries
            st.session_state.summary = []

            with st.spinner("Summarizing..."):
                if ALL_CHAPTERS in selected_chapters:
                    for chapter in st.session_state.video_chapters['Chapters']:
                        summary = summarize_podcast_transcript(model_choice, chapter, st.session_state.vector_store, summary_type)
                        if summary:  # Check if summary is not empty
                            st.session_state.summary.append(summary)
                        else:
                            st.warning(f"Could not generate a summary for chapter: {chapter}. There might be an issue with the video.")
                else:
                    for chapter in selected_chapters:
                        summary = summarize_podcast_transcript(model_choice, chapter, st.session_state.vector_store, summary_type)
                        if summary:  # Check if summary is not empty
                            st.session_state.summary.append(summary)
                        else:
                            st.warning(f"Could not generate a summary for chapter: {chapter}. There might be an issue with the video.")

    # Check if the summaries already exist and display them
    if st.session_state.summary:
        st.header("Podcast Summary")
        display_summaries(st.session_state.summary)

    if st.session_state.summary:
        # Prepare podcast information as text
        podcast_info_text = f"Channel Title: {st.session_state.channelTitle}\nTitle: {st.session_state.title}\nTags: {st.session_state.tags}\n"

        # Prepare summaries as text
        full_summary_text = "\n".join([
            f"Chapter: {summary_dict['chapter']}\n{summary_dict['summary']}\n---"
            for summary_dict in st.session_state.summary
        ])

        # Generate the PDF
        pdf_bytes = string_to_pdf_bytes(podcast_info_text, full_summary_text)

        st.download_button(
            label="Download Summary as PDF",
            data=pdf_bytes,
            file_name="podcast_summary.pdf",
            mime="application/pdf",
        )

if __name__ == "__main__":
    main()
