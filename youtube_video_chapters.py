import os
import re
import ast
import json
from googleapiclient.discovery import build
from langchain.output_parsers import ResponseSchema, StructuredOutputParser
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.document_loaders import YoutubeLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.summarize import load_summarize_chain
from langchain.docstore.document import Document
from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate
)

# Constants
GOOGLE_API_KEY_ENV = "GOOGLE_API_KEY"
OPENAI_API_KEY_ENV = "OPENAI_API_KEY"

# Function to fetch video info
def fetch_video_info(url):
    video_id = re.search(r"(?<=v=)[^&#]+", url)
    video_id = video_id.group(0) if video_id else None
    if not video_id:
        raise ValueError("Invalid YouTube URL")
        
    youtube = build('youtube', 'v3', developerKey=os.environ["GOOGLE_API_KEY"])
    request = youtube.videos().list(part="snippet", id=video_id)
    response = request.execute()
    
    items = response.get("items")
    
    if not items:
        raise ValueError("Video not found")
        
    snippet = items[0]["snippet"]
    
    info_dict = {
        'description': snippet.get('description', ''),
        'title': snippet.get('title', ''),
        'channelTitle': snippet.get('channelTitle', ''),
        'tags': snippet.get('tags', [])
    }
    return info_dict

# Function to load the document (transcript)
def load_transcript(video_url):
    loader = YoutubeLoader.from_youtube_url(video_url, add_video_info=True)
    document = loader.load()
    # Check if document is empty or doesn't contain page_content
    if not document or not hasattr(document[0], 'page_content') or not document[0].page_content:
        return None
    return document[0].page_content

# Function to fetch chapters based on video info and transcript
def fetch_chapters(video_url):
    # Format instructions
    chapters_existence_schema = ResponseSchema(name="Status",
                             description="Are there any chapters in the description, answer true or false ? \
                             output them as a Python boolean.")
    chapters_schema = ResponseSchema(name="Chapters",
                                description="Extract any \
                                Chapters in the description and \
                                output them as a Python list.")
    response_schemas = [chapters_existence_schema, 
                    chapters_schema]
    output_parser = StructuredOutputParser.from_response_schemas(response_schemas)
    format_instructions = output_parser.get_format_instructions()

    # Create PromptTemplate object
    prompt_template = """
    Your task is to extract chapter titles from the given YouTube video description. The description is enclosed between triple ticks. 
    Chapters are often listed under sections labeled as 'Chapters,' 'Outline,' 'Show Notes,' or even in unnamed sections. 
    They usually come with timestamps, either before or after the text description.

    {format_instructions}

    For example, chapter entries might appear as follows:

    - 0:00 - Introduction
    - (1:04:58) Fraught NATO summit, ammo crisis, Sweden joins NATO
    - 09:48 Semantic memory
    - Seth details the real power our decisions have on our happiness. [11:52]

    Your task is to focus only on the chapter titles, disregarding the timestamps.

    Here's the video description to analyze:

    '''{text}\n'''
    """

    video_info = fetch_video_info(video_url)
    description = video_info.get('description', 'Description not available')
    prompt = ChatPromptTemplate.from_template(template=prompt_template)
    messages = prompt.format_messages(text=description, 
                                format_instructions=format_instructions)
    llm = ChatOpenAI(temperature=0, model='gpt-3.5-turbo', openai_api_key=os.environ["OPENAI_API_KEY"])
    response = llm(messages)
    output_dict = output_parser.parse(response.content)

    if not output_dict["Status"]:  # Assuming 'Status' indicates whether chapters were found
        # No chapters found in the description, proceed to analyze transcript
        transcript = load_transcript(video_url)
        #print("Transcript: "+ transcript)
        if not transcript:
            return []
        
        # Split the transcript
        text_splitter = RecursiveCharacterTextSplitter(separators=["\n\n", "\n", " "], chunk_size=10000, chunk_overlap=2200)
        segmented_documents = text_splitter.split_text(transcript)
        
        # Prepare the documents
        #docs = [{"page_content": t} for t in segmented_documents]  # Adjust this based on your actual Document class
        docs = [Document(page_content=t) for t in segmented_documents]

        # Set up your map and combine prompts
        template="""
        You're an assistant tasked with summarizing podcast topics.
        Guidelines:
        - Consolidate similar and close topics into overarching themes or categories.
        - Pair each consolidated topic with a brief description, format 'Topic: Description.'
        - Keep topics concise but clear.
        - Include only topics that are substantially discussed, not fleeting comments.
        - Use podcast-specific vocabulary.
        - Use text-only bullet points.
        - Extract information solely from the podcast transcript.
        - Output in a Python list.

        % EXAMPLES FOR ILLUSTRATION
        - Animal Cohabitation: Discusses challenges with turkeys.
        - Crisis Management: Covers polycrisis and investor influence.
        - Risk & Precaution: Talks about personal risk and precautionary measures.
        % END OF EXAMPLES
        """
        system_message_prompt_map = SystemMessagePromptTemplate.from_template(template)

        human_template="Transcript: {text}" # Simply just pass the text as a human message
        human_message_prompt_map = HumanMessagePromptTemplate.from_template(human_template)

        chat_prompt_map = ChatPromptTemplate.from_messages(messages=[system_message_prompt_map, human_message_prompt_map])

        template="""
        You're an assistant that summarizes podcast topics.
        Guidelines:
        - Output in a Python list.
        - Consolidate similar and close topics into broader categories.
        - Use podcast-specific vocabulary.
        - Add a brief description like 'Topic: Description.'
        - Remove duplicates.
        - Use only transcript topics, no examples.

        % EXAMPLES FOR ILLUSTRATION
        ["Environment: Covers climate change and sustainability", "Finance and Crisis: Discusses financial instability and crisis management", 
        "Power and Influence: Talks about the role of investors and decision-makers", 
        "Risk Management: Covers precautionary principles and personal stakes"]
        % END OF EXAMPLES
        """
        system_message_prompt_map = SystemMessagePromptTemplate.from_template(template)

        human_template="Transcript: {text}" # Simply just pass the text as a human message
        human_message_prompt_map = HumanMessagePromptTemplate.from_template(human_template)

        chat_prompt_combine = ChatPromptTemplate.from_messages(messages=[system_message_prompt_map, human_message_prompt_map])
        
        # Set up your summarization chain here
        chain = load_summarize_chain(llm,
                                    chain_type="map_reduce",
                                    map_prompt=chat_prompt_map,
                                    combine_prompt=chat_prompt_combine,
                                    #verbose=True
                                    )

        # Extract topics (chapters) from the transcript
        topics_found = chain.run({"input_documents": docs, "format_instructions": format_instructions})

        try:
            chapters_list = ast.literal_eval(topics_found)
        except (SyntaxError, ValueError) as e:
            chapters_list = []

        # Now, chapters_list is a list of strings
        output_dict['Chapters'] = chapters_list
        output_dict['Status'] = True
    return output_dict

# Function to format chapters
def format_fetched_chapters(data):
    if 'Chapters' in data:
        chapters = data['Chapters']
        result = '\n'.join([f'{i+1}. {chapter}' for i, chapter in enumerate(chapters)])
        return result
    else:
        return "No chapters found."

# Main function
def main():
    openai_api_key = "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"
    os.environ[OPENAI_API_KEY_ENV] = openai_api_key
    google_api_key = 'AIzaSyDFx4jp1RNN9ilcomlA6I0wd0lk3W9-ATY'
    os.environ[GOOGLE_API_KEY_ENV] = google_api_key

    youtube_podcast_url = 'https://www.youtube.com/watch?v=KopLe5NZBJc'
    chapters_data = fetch_chapters(youtube_podcast_url)
    formatted_chapters = format_fetched_chapters(chapters_data)
    print(formatted_chapters)

if __name__ == "__main__":
    main()