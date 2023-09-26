# YouTube Video Summarizer

YouTube Video Summarizer is a Streamlit application designed to generate concise summaries of YouTube videos. It fetches video information, chapters, and transcripts, allowing users to select specific chapters to summarize and download the summary as a PDF.

## Features
- Fetch and display YouTube video information including channel title, video title, and tags.
- Extract video chapters from the description or transcript.
- Generate summaries for the entire video or selected chapters.
- Download the generated summary as a PDF.
- Convert strings to PDFs with Unicode support.
- User-friendly interface with Streamlit.

## Technology Stack
- Python
- Streamlit
- OpenAI GPT-3.5 Turbo
- Google YouTube Data API v3
- FPDF for PDF generation
- Tiktoken for token counting
- Langchain for text processing and embeddings

## Installation & Setup
```sh
# Clone the repository
gh repo clone bencmc/youtube_video_summarizer

# Navigate to the project directory
cd [Project Directory]

# Install the required packages
pip install -r requirements.txt

