import re
import os
from langchain.output_parsers import ResponseSchema, StructuredOutputParser
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from googleapiclient.discovery import build

def get_video_description(url):
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
    return items[0]["snippet"]["description"]


def get_video_chapters(video_url):
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
    Based on the provided YouTube video description transcript, your task is to extract the video chapters. 
    These can often be found in a section labeled 'Chapters', 'Outline', 'Show Notes', or sometimes in an unnamed section. 
    The chapters typically have associated timestamps, either preceding or following the text.
    {format_instructions}

    Examples of chapters might look like:

    0:00 - Introduction
    (1:04:58) Fraught NATO summit, ammo crisis, Sweden joins NATO
    09:48 Semantic memory.
    Seth details the real power our decisions have on our happiness. [11:52]

    Your task is to disregard these timestamps and extract only the chapter titles.

    Here's the podcast description for you to analyze:

    {text}\n
    """

    description=get_video_description(video_url)
    prompt = ChatPromptTemplate.from_template(template=prompt_template)
    messages = prompt.format_messages(text=description, 
                                format_instructions=format_instructions)
    llm = ChatOpenAI(temperature=0, model='gpt-3.5-turbo', openai_api_key=os.environ["OPENAI_API_KEY"])
    response = llm(messages)
    output_dict = output_parser.parse(response.content)
    return output_dict

# format chapters
def format_chapters(data):
    if 'Chapters' in data:
        chapters = data['Chapters']
        result = '\n'.join([f'{i+1}. {chapter}' for i, chapter in enumerate(chapters)])
        return result
    else:
        return "No chapters found."

if __name__ == "__main__":
    
    import os
    openai_api_key="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"
    os.environ["OPENAI_API_KEY"] = openai_api_key
    google_api_key='AIzaSyDFx4jp1RNN9ilcomlA6I0wd0lk3W9-ATY'
    os.environ["GOOGLE_API_KEY"] = google_api_key

    # podcasts videos (testing)
    # https://www.youtube.com/watch?v=HekHk6yLmF0&t=3468s
    # https://www.youtube.com/watch?v=Mde2q7GFCrw&t=5453s
    # https://www.youtube.com/watch?v=3TV3dNJGqGI&t=3491s
    # https://www.youtube.com/watch?v=F80WmftF5YY&t=12s
    # https://www.youtube.com/watch?v=ofAOwyKuKxI
    # https://www.youtube.com/watch?v=aZ-BjJZxNoA
    youtube_podcast_url='https://www.youtube.com/watch?v=HekHk6yLmF0&t=3468s'
    dict=get_video_chapters(youtube_podcast_url)
    print(dict)
    print(format_chapters(dict))
