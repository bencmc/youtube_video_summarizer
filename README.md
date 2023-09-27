# YouTube Video Summarizer
Youtube Video Summarizer is an application that allows users to generate summaries for YouTube videos. The application fetches video transcripts, allows users to select specific chapters to summarize, and provides the option to download the summary as a PDF. This application is particularly useful for extracting valuable insights from content-rich videos such as podcasts, lectures, and interviews.

## Motivation
As an avid podcast enthusiast, I found myself consuming a plethora of content on YouTube. However, I often felt the need for a tool that could help me quickly recall and reflect on the vast array of information I was absorbing. This led to the inception of Youtube Video Summarizer, a tool that not only serves as a memory aid by generating succinct summaries of the viewed content but also allows users to download these summaries as PDFs, creating a readable reference that can be revisited at any time.


![Screenshot from 2023-09-27 16-51-49](https://github.com/bencmc/youtube_video_summarizer/assets/9975447/37af5077-81d9-4576-b135-83b782cab44d)


## Features
- Fetch and display YouTube video information including channel title, video title, and tags.
- Extract video chapters from the description or transcript.
- Generate summaries for the entire video or selected chapters.
- Download the generated summary as a PDF.
- User-friendly interface.

## How to Use
1. Enter your [OpenAI API Key](https://platform.openai.com/account/api-keys) and [Google API Key](https://console.cloud.google.com/apis/credentials/key) in the appropriate fields in the sidebar under 'User Input'.
2. Input the YouTube video URL you want to summarize in the 'Enter YouTube video URL' field in the sidebar.
3. Click the 'Start' button once all fields are filled.
4. Once the video information is fetched, select the chapters you want to summarize in the 'Chapter Selection' section. You can select 'All Chapters' or specific chapters.
5. Choose the summary type, either 'paragraph' or 'bullet points', from the dropdown list.
6. Click the 'Summarize' button to generate summaries.
7. Review the generated summaries in the 'Video Summary' section.
8. Download the entire summary as a PDF by clicking the 'Download Summary as PDF' button.

## Try it out

https://youtubevideosummarizer.streamlit.app/

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
This project is licensed under the MIT license - see the LICENSE file for details.

## Installation & Setup

```sh
# Clone the repository
git clone git@github.com:bencmc/youtube_video_summarizer.git

# Navigate to the project directory
cd youtube_video_summarizer/

# Set up the virtual environment
python3 -m venv venv

# Activate the virtual environment
# For Windows
.\venv\Scripts\activate

# For MacOS/Linux
source venv/bin/activate

# Install the required packages
pip install -r requirements.txt

# Run application
streamlit run youtube_video_summarizer.py
