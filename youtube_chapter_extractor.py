import os
import re
import ast
from googleapiclient.discovery import build
from langchain.output_parsers import ResponseSchema, StructuredOutputParser
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.prompts.chat import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain.document_loaders import YoutubeLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.summarize import load_summarize_chain
from langchain.docstore.document import Document


def fetch_video_info(url):
    """Fetches the information of a given YouTube video URL."""
    video_id = re.search(r"(?<=v=)[^&#]+", url)
    if not video_id:
        raise ValueError("Invalid YouTube URL")
    video_id = video_id.group(0)
    
    youtube = build('youtube', 'v3', developerKey=os.environ["GOOGLE_API_KEY"])
    response = youtube.videos().list(part="snippet", id=video_id).execute()
    items = response.get("items")
    if not items:
        raise ValueError("Video not found")
    
    snippet = items[0]["snippet"]
    return {
        'description': snippet.get('description', ''),
        'title': snippet.get('title', ''),
        'channelTitle': snippet.get('channelTitle', ''),
        'tags': snippet.get('tags', [])
    }

def load_transcript(video_url):
    """Loads the transcript of the video."""
    loader = YoutubeLoader.from_youtube_url(video_url, add_video_info=True)
    document = loader.load()
    if not document or not hasattr(document[0], 'page_content') or not document[0].page_content:
        return None
    return document[0].page_content

def get_video_description(video_url):
    """Fetches the video description for the given video URL."""
    video_info = fetch_video_info(video_url)
    return video_info.get('description', 'Description not available')

def extract_chapters_from_description(description):
    """Extracts chapters from the description and returns them."""
    # Set up the StructuredOutputParser with the required ResponseSchemas
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

    # Define the prompt template for extracting chapter titles
    prompt_template = """
    Your task is to extract chapter titles from the given YouTube video description. The description is enclosed between triple ticks. 
    Chapters are often listed under sections labeled as 'Chapters', 'Outline', 'Show Notes', 'Timestamps' or even in unnamed sections. 
    They usually come with timestamps, either before or after the text description.

    {format_instructions}

    For example, chapter entries might appear as follows:

    - 0:00 - Introduction
    - (1:04:58) Fraught NATO summit, ammo crisis, Sweden joins NATO
    - 09:48 Semantic memory
    - Seth details the real power our decisions have on our happiness. [11:52]
    - 00:12:49 Disagreeableness, Social Resistance; Loneliness & Group Think 

    Your task is to focus only on the chapter titles, disregarding the timestamps.

    Here's the video description to analyze:

    '''{text}\n'''
    """

    # Create a list of messages using the ChatPromptTemplate
    prompt = ChatPromptTemplate.from_template(template=prompt_template)
    messages = prompt.format_messages(text=description, format_instructions=format_instructions)

    # Set up the language model
    llm = ChatOpenAI(temperature=0, model='gpt-3.5-turbo', openai_api_key=os.environ["OPENAI_API_KEY"])
    response = llm(messages)
    
    # Parse the response content to extract chapters
    return output_parser.parse(response.content)

def extract_chapters_from_transcript(video_url):
    """Extracts chapters from the transcript and returns them."""
    transcript = load_transcript(video_url)
    if not transcript:
        return []

    # Segment the transcript into manageable chunks
    text_splitter = RecursiveCharacterTextSplitter(separators=["\n\n", "\n", " "], chunk_size=10000, chunk_overlap=2200)
    segmented_documents = text_splitter.split_text(transcript)
    docs = [Document(page_content=t) for t in segmented_documents]

    # Set up your map and combine prompts
    template="""
    You're an assistant tasked with summarizing video topics.
    Guidelines:
    - Consolidate similar and close topics into overarching themes or categories.
    - Pair each consolidated topic with a brief description, format 'Topic: Description.'
    - Keep topics concise but clear.
    - Include only topics that are substantially discussed, not fleeting comments.
    - Use video transcript vocabulary.
    - Use text-only bullet points.
    - Extract information solely from the video transcript.
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
    You're an assistant that summarizes video topics.
    Guidelines:
    - Output in a Python list.
    - Consolidate similar and close topics into broader categories.
    - Use video-specific vocabulary.
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

    # Set up the language model
    llm = ChatOpenAI(temperature=0, model='gpt-3.5-turbo', openai_api_key=os.environ["OPENAI_API_KEY"])

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
        return ast.literal_eval(topics_found)
    except (SyntaxError, ValueError) as e:
        return []

def fetch_chapters(video_url):
    """Fetches chapters from the YouTube video's description or its transcript."""
    description = get_video_description(video_url)
    output_dict = extract_chapters_from_description(description)

    if not output_dict["Status"]:
        chapters_list = extract_chapters_from_transcript(video_url)
        output_dict['Chapters'] = chapters_list
        output_dict['Status'] = bool(chapters_list)

    return output_dict

# def fetch_chapters(video_url):
#     """Fetches chapters from the YouTube video's description or its transcript."""
#     # Format instructions
#     chapters_existence_schema = ResponseSchema(name="Status",
#                              description="Are there any chapters in the description, answer true or false ? \
#                              output them as a Python boolean.")
#     chapters_schema = ResponseSchema(name="Chapters",
#                                 description="Extract any \
#                                 Chapters in the description and \
#                                 output them as a Python list.")
#     response_schemas = [chapters_existence_schema, 
#                     chapters_schema]
#     output_parser = StructuredOutputParser.from_response_schemas(response_schemas)
#     format_instructions = output_parser.get_format_instructions()

#     # Create PromptTemplate object
#     prompt_template = """
#     Your task is to extract chapter titles from the given YouTube video description. The description is enclosed between triple ticks. 
#     Chapters are often listed under sections labeled as 'Chapters', 'Outline', 'Show Notes', 'Timestamps' or even in unnamed sections. 
#     They usually come with timestamps, either before or after the text description.

#     {format_instructions}

#     For example, chapter entries might appear as follows:

#     - 0:00 - Introduction
#     - (1:04:58) Fraught NATO summit, ammo crisis, Sweden joins NATO
#     - 09:48 Semantic memory
#     - Seth details the real power our decisions have on our happiness. [11:52]
#     - 00:12:49 Disagreeableness, Social Resistance; Loneliness & Group Think 

#     Your task is to focus only on the chapter titles, disregarding the timestamps.

#     Here's the video description to analyze:

#     '''{text}\n'''
#     """

#     video_info = fetch_video_info(video_url)
#     description = video_info.get('description', 'Description not available')
#     prompt = ChatPromptTemplate.from_template(template=prompt_template)
#     messages = prompt.format_messages(text=description, 
#                                 format_instructions=format_instructions)
#     llm = ChatOpenAI(temperature=0, model='gpt-3.5-turbo', openai_api_key=os.environ["OPENAI_API_KEY"])
#     response = llm(messages)
#     output_dict = output_parser.parse(response.content)

#     if not output_dict["Status"]:  # Assuming 'Status' indicates whether chapters were found
#         # No chapters found in the description, proceed to analyze transcript
#         transcript = load_transcript(video_url)
#         #print("Transcript: "+ transcript)
#         if not transcript:
#             return []
        
#         # Split the transcript
#         text_splitter = RecursiveCharacterTextSplitter(separators=["\n\n", "\n", " "], chunk_size=10000, chunk_overlap=2200)
#         segmented_documents = text_splitter.split_text(transcript)
        
#         # Prepare the documents
#         #docs = [{"page_content": t} for t in segmented_documents]  # Adjust this based on your actual Document class
#         docs = [Document(page_content=t) for t in segmented_documents]

#         # Set up your map and combine prompts
#         template="""
#         You're an assistant tasked with summarizing video topics.
#         Guidelines:
#         - Consolidate similar and close topics into overarching themes or categories.
#         - Pair each consolidated topic with a brief description, format 'Topic: Description.'
#         - Keep topics concise but clear.
#         - Include only topics that are substantially discussed, not fleeting comments.
#         - Use video transcript vocabulary.
#         - Use text-only bullet points.
#         - Extract information solely from the video transcript.
#         - Output in a Python list.

#         % EXAMPLES FOR ILLUSTRATION
#         - Animal Cohabitation: Discusses challenges with turkeys.
#         - Crisis Management: Covers polycrisis and investor influence.
#         - Risk & Precaution: Talks about personal risk and precautionary measures.
#         % END OF EXAMPLES
#         """
#         system_message_prompt_map = SystemMessagePromptTemplate.from_template(template)

#         human_template="Transcript: {text}" # Simply just pass the text as a human message
#         human_message_prompt_map = HumanMessagePromptTemplate.from_template(human_template)

#         chat_prompt_map = ChatPromptTemplate.from_messages(messages=[system_message_prompt_map, human_message_prompt_map])

#         template="""
#         You're an assistant that summarizes video topics.
#         Guidelines:
#         - Output in a Python list.
#         - Consolidate similar and close topics into broader categories.
#         - Use video-specific vocabulary.
#         - Add a brief description like 'Topic: Description.'
#         - Remove duplicates.
#         - Use only transcript topics, no examples.

#         % EXAMPLES FOR ILLUSTRATION
#         ["Environment: Covers climate change and sustainability", "Finance and Crisis: Discusses financial instability and crisis management", 
#         "Power and Influence: Talks about the role of investors and decision-makers", 
#         "Risk Management: Covers precautionary principles and personal stakes"]
#         % END OF EXAMPLES
#         """
#         system_message_prompt_map = SystemMessagePromptTemplate.from_template(template)

#         human_template="Transcript: {text}" # Simply just pass the text as a human message
#         human_message_prompt_map = HumanMessagePromptTemplate.from_template(human_template)

#         chat_prompt_combine = ChatPromptTemplate.from_messages(messages=[system_message_prompt_map, human_message_prompt_map])
        
#         # Set up your summarization chain here
#         chain = load_summarize_chain(llm,
#                                     chain_type="map_reduce",
#                                     map_prompt=chat_prompt_map,
#                                     combine_prompt=chat_prompt_combine,
#                                     #verbose=True
#                                     )

#         # Extract topics (chapters) from the transcript
#         topics_found = chain.run({"input_documents": docs, "format_instructions": format_instructions})

#         try:
#             chapters_list = ast.literal_eval(topics_found)
#         except (SyntaxError, ValueError) as e:
#             chapters_list = []

#         # Now, chapters_list is a list of strings
#         output_dict['Chapters'] = chapters_list
#         output_dict['Status'] = True
#     return output_dict


def format_fetched_chapters(data):
    """Formats the fetched chapters in a readable format."""
    chapters = data.get('Chapters')
    return '\n'.join([f'{i+1}. {chapter}' for i, chapter in enumerate(chapters)]) if chapters else "No chapters found."


def main():
    openai_api_key = os.getenv(OPENAI_API_KEY_ENV, 'Your OpenAI API Key Here')
    google_api_key = os.getenv(GOOGLE_API_KEY_ENV, 'Your Google API Key Here')
    youtube_video_url = 'https://www.youtube.com/watch?v=KopLe5NZBJc'  # This can also be passed as an argument or read from a config file
    
    video_info = fetch_video_info(youtube_video_url, google_api_key)
    chapters_data = fetch_chapters(youtube_video_url)
    formatted_chapters = format_fetched_chapters(chapters_data)
    print(formatted_chapters)


if __name__ == "__main__":
    main()