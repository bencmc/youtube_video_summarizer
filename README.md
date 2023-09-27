# YouTube Video Summarizer

Youtube Video Summarizer is an open-source Streamlit application that allows users to generate summaries for YouTube videos. The application fetches video transcripts, allows users to select specific chapters to summarize, and provides the option to download the summary as a PDF.

![Screenshot from 2023-09-27 16-51-49](https://github.com/bencmc/youtube_video_summarizer/assets/9975447/37af5077-81d9-4576-b135-83b782cab44d)

## Features
- Fetch and display YouTube video information including channel title, video title, and tags.
- Extract video chapters from the description or transcript.
- Generate summaries for the entire video or selected chapters.
- Download the generated summary as a PDF.
- User-friendly interface.

## How to Use
1. Enter your OpenAI API Key and Google API Key in the appropriate fields in the sidebar under 'User Input'.
2. Input the YouTube video URL you want to summarize in the 'Enter YouTube video URL' field in the sidebar.
3. Click the 'Start' button once all fields are filled.
4. Once the video information is fetched, select the chapters you want to summarize in the 'Chapter Selection' section. You can select 'All Chapters' or specific chapters.
5. Choose the summary type, either 'paragraph' or 'bullet points', from the dropdown list.
6. Click the 'Summarize' button to generate summaries.
7. Review the generated summaries in the 'Video Summary' section.
8. Download the entire summary as a PDF by clicking the 'Download Summary as PDF' button.

## Technology Stack
- Python
- Streamlit
- OpenAI GPT-3.5 Turbo
- Google YouTube Data API v3
- FPDF for PDF generation
- Langchain for text processing and embeddings

## Contributing
We welcome contributions! Please see the CONTRIBUTING.md for details on how to contribute to this project.

## License
This project is licensed under the <Insert License Here> - see the LICENSE file for details.

## Installation & Setup
```sh
# Clone the repository
gh repo clone bencmc/youtube_video_summarizer

# Navigate to the project directory
cd [Project Directory]

# Install the required packages
pip install -r requirements.txt

# Run application
streamlit run <filename>.py



