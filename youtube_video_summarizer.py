__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

import streamlit as st
from transcript_processing_util import generate_video_summary, initialize_vector_store
import os
import io
from fpdf import FPDF
from youtube_chapters_extractor import fetch_chapters, fetch_video_info

# Constants
ALL_CHAPTERS = "All Chapters"

import unicodedata

def string_to_pdf_bytes(video_info, summary_text):
    pdf_buffer = io.BytesIO()

    pdf = FPDF(format='A4')  # Explicitly specify A4 format
    pdf.add_page()

    # Set Auto Page Break
    pdf.set_auto_page_break(auto=1, margin=15)
    
    # Set Wider Margins
    pdf.set_left_margin(10)  # Set left margin to 10
    pdf.set_right_margin(10)  # Set right margin to 10

    # Use the built-in Arial font
    pdf.set_font("Arial", size=12)

    # Add the video Information
    pdf.cell(0, 10, "Video Information:", ln=True)
    pdf.multi_cell(0, 10, video_info.encode('UTF-8').decode('latin1'))
    pdf.cell(0, 10, "-----", ln=True)

    # Add the summary text
    pdf.multi_cell(0, 10, summary_text.encode('UTF-8').decode('latin1'))
    
    # Save the PDF content to the BytesIO object
    pdf_content = pdf.output(dest='S')  # Avoid encoding the whole PDF content
    pdf_buffer.write(pdf_content.encode('latin1'))  # Encode to 'latin1' when writing to buffer

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
        chapter_title = f"<h2 style='font-size:20px;'>Chapter: {summary_dict['chapter']}</h2>"
        
        # If the summary contains bullet points, replace them with HTML list tags.
        if '\n- ' in summary_dict['summary']:  # Assuming bullet points are represented by '•' character.
            summary_text = "<ul>"
            # Split the summary string by newline character and wrap each bullet point with <li> tags.
            summary_text += ''.join(f"<li style='font-size:18px;'>{line[2:]}</li>" for line in summary_dict['summary'].split('\n') if line)
            summary_text += "</ul>"
        else:  # If no bullet points, just wrap the summary text with <p> tags.
            summary_text = f"<p style='font-size:18px;'>{summary_dict['summary']}</p>"
        
        st.markdown(chapter_title, unsafe_allow_html=True)
        st.markdown(summary_text, unsafe_allow_html=True)
        st.write("---")

def main():
    youtube_logo_url = "https://upload.wikimedia.org/wikipedia/commons/0/09/YouTube_full-color_icon_(2017).svg"  # URL of the YouTube logo without text
    title_with_logo = f'<img src="{youtube_logo_url}" width="100" style="vertical-align: middle;"> <span style="font-size:50px; vertical-align: middle; font-weight: bold; margin-left: 10px;"> Video Summarizer</span>'
    st.markdown(title_with_logo, unsafe_allow_html=True)  # Display the title with the logo
    
    initialize_session_state()

    st.sidebar.header("User Input")

    api_keys = {
        "OPENAI_API_KEY": st.sidebar.text_input("Enter OpenAI API Key", type="password"),
        "GOOGLE_API_KEY": st.sidebar.text_input("Enter Google API Key", type="password"),
    }

    video_url = st.sidebar.text_input("Enter YouTube video URL")

    all_fields_filled = all(api_keys.values()) and video_url

    if all_fields_filled:
        st.sidebar.success("All fields are filled!")
        set_environment_keys(api_keys)

    start_button = st.sidebar.button("Start", disabled=not all_fields_filled)

    if start_button and video_url:
        # Clear the previous summaries
        st.session_state.summary = []

        with st.spinner("Initializing..."):
            # Get video information (channel title, video title, tags)
            video_info = fetch_video_info(video_url)
            st.session_state.channelTitle = video_info.get('channelTitle', 'Channel not available')
            st.session_state.title = video_info.get('title', 'Title not available')
            st.session_state.tags = video_info.get('tags', 'Tags not available')

            with st.spinner("Getting video transcript..."):
                st.session_state.vector_store = initialize_vector_store(video_url)

            with st.spinner("Fetching video chapters..."):
                video_chapters = fetch_chapters(video_url)

            st.session_state.video_chapters = video_chapters
            if 'Chapters' in video_chapters:
                st.session_state.chapters_list = [ALL_CHAPTERS] + video_chapters['Chapters']
            else:
                st.error("Issue with video transcript fetching")

    if st.session_state.get('channelTitle') and st.session_state.get('title') and st.session_state.get('tags'):
        st.header("Video Information")
        st.markdown(f"<p style='font-size:18px;'><strong>Channel Title:</strong> {st.session_state.channelTitle}</p>", unsafe_allow_html=True)
        st.markdown(f"<p style='font-size:18px;'><strong>Title:</strong> {st.session_state.title}</p>", unsafe_allow_html=True)
        st.markdown(f"<p style='font-size:18px;'><strong>Tags:</strong> {', '.join(st.session_state.tags)}</p>", unsafe_allow_html=True)


    if st.session_state.chapters_list:
        st.header("Chapter Selection")
        selected_chapters, valid_selection = get_selected_chapters()

        # Move the summary type selectbox here, after the chapters and before the Summarize button
        #summary_type = st.selectbox("Choose Summary Type", ["paragraph", "bullet points"])
        st.header("Summary Type")
        summary_type = st.selectbox("Select Summary Type", ["paragraph", "bullet points"])
        

        if st.button("Summarize", disabled=not valid_selection):
            # Clear previous summaries
            st.session_state.summary = []

            with st.spinner("Summarizing..."):
                if ALL_CHAPTERS in selected_chapters:
                    for chapter in st.session_state.video_chapters['Chapters']:
                        summary = generate_video_summary("gpt-3.5-turbo", chapter, st.session_state.vector_store, summary_type)
                        if summary:  # Check if summary is not empty
                            st.session_state.summary.append(summary)
                        else:
                            st.warning(f"Could not generate a summary for chapter: {chapter}. There might be an issue with the video.")
                else:
                    for chapter in selected_chapters:
                        summary = generate_video_summary("gpt-3.5-turbo", chapter, st.session_state.vector_store, summary_type)
                        if summary:  # Check if summary is not empty
                            st.session_state.summary.append(summary)
                        else:
                            st.warning(f"Could not generate a summary for chapter: {chapter}. There might be an issue with the video.")

    # Check if the summaries already exist and display them
    if st.session_state.summary:
        st.header("Video Summary")
        display_summaries(st.session_state.summary)

    if st.session_state.summary:
        # Prepare video information as text
        video_info_text = f"Channel Title: {st.session_state.channelTitle}\nTitle: {st.session_state.title}\nTags: {st.session_state.tags}\n"

        # Prepare summaries as text
        full_summary_text = "\n".join([
            f"Chapter: {summary_dict['chapter']}\n{summary_dict['summary']}\n---"
            for summary_dict in st.session_state.summary
        ])

        # Generate the PDF
        pdf_bytes = string_to_pdf_bytes(video_info_text, full_summary_text)

        st.download_button(
            label="Download Summary as PDF",
            data=pdf_bytes,
            file_name="video_summary.pdf",
            mime="application/pdf",
        )

if __name__ == "__main__":
    main()
