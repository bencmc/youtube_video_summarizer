from youtube_video_chapters import format_chapters, get_video_chapters
import os
import textwrap
from langchain.chat_models import ChatOpenAI
from langchain import PromptTemplate
from langchain.chains.summarize import load_summarize_chain
from langchain.docstore.document import Document
from langchain.text_splitter import CharacterTextSplitter
from langchain.callbacks import get_openai_callback
from langchain.document_loaders import YoutubeLoader
from langchain.chains import LLMChain
from langchain.text_splitter import RecursiveCharacterTextSplitter
import tiktoken
from enum import Enum
from collections import defaultdict
from fpdf import FPDF


def string_to_pdf(input_str, filename="output.pdf", font_path="ARIALUNI.TTF"):
    pdf = FPDF()
    pdf.add_page()

    # Add the font ensuring it supports Unicode characters
    pdf.add_font("ArialUnicode", style="", fname=font_path, uni=True)
    pdf.set_font("ArialUnicode", size=12)

    # Add a cell
    pdf.multi_cell(0, 10, input_str)

    # Output the PDF to a file
    pdf.output(filename)

# count the tokens
def num_tokens_from_string(string: str, encoding_name: str) -> int:
    """Returns the number of tokens in a text string."""
    encoding = tiktoken.encoding_for_model(encoding_name)
    num_tokens = len(encoding.encode(string))
    return num_tokens

# calculate the cost of OpenAI API calls
def openaipricing(input_text: str, output_text: str, model_name: str = None) -> float:
    in_costs = {
        'gpt-3.5-turbo-16k': 0.003,
        'gpt-3.5-turbo': 0.0015,
        }
    out_costs = { 
        'gpt-3.5-turbo-16k': 0.004,
        'gpt-3.5-turbo': 0.002,
        }
    input_tokens= num_tokens_from_string(input_text, model_name)
    output_tokens= num_tokens_from_string(output_text, model_name)
    input_cost=input_tokens*in_costs[model_name]/1000.
    output_cost=output_tokens*out_costs[model_name]/1000.
    return input_cost+output_cost

# load the transcript
def load_document(video_url):
    loader = YoutubeLoader.from_youtube_url(video_url, add_video_info=True)
    document=loader.load()
    # metadata
    # for dictionary in result[0].metadata:
    # for key, value in result[0].metadata.items():
    #     print(f'Key: {key}, Value: {value}')
    return document[0].page_content

# split document 
def TranscriptChunking(document, model_name, input_text_percentage, overlap_percentage):
    # Define model context limitations
    model_context = {
        "gpt-3.5-turbo-16k": 16384,
        "gpt-3.5-turbo": 4096,
        "gpt-4": 8192

    }

    # Get the maximum token limit for the specified model
    max_tokens = model_context[model_name]

    # Calculate the chunk size and overlap size in tokens
    chunk_size = max_tokens * input_text_percentage // 1
    overlap_size = chunk_size * overlap_percentage // 1

    # Convert chunk size and overlap size to characters
    chunk_size_char = chunk_size * 4
    overlap_size_char = overlap_size * 4

    # define the text splitter 
    text_splitter = RecursiveCharacterTextSplitter(separators=["\n\n", "\n", " "], chunk_size=chunk_size_char, chunk_overlap=overlap_size_char)
    
    # Use the text splitter to split the document into chunks
    segmented_documents = text_splitter.split_text(document)

    docs = [Document(page_content=t) for t in segmented_documents]
    print(len(docs))

    return docs


# summarize podcast transcript
def summarize_podcast_transcript(video_url,model_name,input_text_percentage,overlap_percentage):
    # load the transcript
    result=load_document(video_url)
    # split the transcript
    docs=TranscriptChunking(result, model_name, input_text_percentage, overlap_percentage)
    docs_num=len(docs)

    # load ChatOpenAI object
    llm = ChatOpenAI(temperature=0, model=model_name)
    # get video chapters
    video_chapters=get_video_chapters(video_url)
    formatted_chapters=format_chapters(video_chapters)
    print(formatted_chapters)
    
    if(docs_num==1):
        prompt_summary_template = """
        Based on the provided podcast transcript, generate an incisive summary that distills the salienthe structure outlined below:t points and the actionable insights of each topic. 
        Highlight any resources mentioned during the discussion. The summaries should follow 
        
        Topic: Name the topic from this list: {formatted_chapters}

        Takeaways: 
        - Make a bullet-point list of the main ideas and steps from the conversation. Avoid starting sentences with repetitive phrases such as "The podcast discusses...". Instead, focus on summarizing the essence of the discussion directly.
        - Keep your summary short and to the point, without repeating information. 
        - Don't forget to mention any tools, books, websites, etc., talked about in the podcast that can help listeners learn more.
        {formatted_chapters}\n
        
        Podcast Transcript:
        {text}\n
        """

        PROMPT = PromptTemplate(
            template=prompt_summary_template,
            input_variables=["formatted_chapters","text"])
        chain = load_summarize_chain(llm, 
                        chain_type="stuff", 
                        prompt=PROMPT)
        output_summary = chain.run({'formatted_chapters': formatted_chapters,
                            'text': docs,
                            'input_documents':""})
        
    else:
        map_prompt = """
        You're a virtual assistant designed to summarize podcast segments. Follow these rules:
        - You'll receive a transcript segment enclosed within double quotes.
        - You'll also find a provided list of potential topics enclosed within triple backticks.
        - Read through the segment and list first.
        - Match the content of the segment to the potential topics from the list.
        - Extract core arguments, opinions, and insights related to each topic. Avoid simply stating that a discussion or mention occurred.
        - Only include summaries for topics that are explicitly discussed in the segment. If a topic is not discussed, do not include it in the summary.
        - Present the summaries in the order that topics appear in the list.

        Format:
        Topic: Use names from the list.
        - Be concise but thorough.
        - Include any relevant resources like tools, books, or websites mentioned in the discussion.

        **List of Topics (between triple backticks)**: 
        ```{formatted_chapters}```

        **Transcript Segment**: 
        ""{text}""
        """

        map_prompt_template = PromptTemplate(
            template=map_prompt,
            input_variables=["text","formatted_chapters"])

        combine_prompt = """
        You are a virtual assistant set to consolidate duplicate topic summaries from a podcast. Follow these rules:

        - You'll receive multiple summaries enclosed within double quotes.
        - Read through the summaries first.
        - Merge summaries with identical topic names. Each topic should be unique in the final output.
        - If a topic is mentioned but not actually discussed in the summaries, exclude it entirely from the consolidated summaries. There should be no topic headers without content.
        - Extract and emphasize the core arguments, opinions, and insights from across the summaries.
        - Be concise but thorough, and remove redundant info.
        - Follow the list order of topics when merging summaries. Pay special attention to frequently mentioned resources or important points.

        For each set of duplicate topics:
        - Condense unique main ideas into bullet points.
        - Include frequently mentioned resources like tools, books, or websites.

        Before concluding, ensure that the consolidated summaries are accurate, clear, and devoid of redundancies.

        Order the consolidated summaries based on the list of topics provided between triple backticks.

        **Topics Discussed**: 
        ""{text}""
        **List of Topics (between triple backticks)**: 
        ```{formatted_chapters}```
        """

        combine_prompt_template = PromptTemplate(
            template=combine_prompt,
            input_variables=["text","formatted_chapters"])      
    
        chain = load_summarize_chain(llm=llm, 
                                chain_type='map_reduce',
                                map_prompt=map_prompt_template,
                                combine_prompt=combine_prompt_template,
                                verbose=True
                                )
        
        output_summary = chain.run({'formatted_chapters': formatted_chapters,
                    'input_documents': docs})
        

        # output_summary = chain.run({'formatted_chapters': formatted_chapters,
        #                     'text': chunks,
        #                     'input_documents':""})        
    return output_summary

if __name__ == "__main__":
    
    import os
    openai_api_key="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"
    os.environ["OPENAI_API_KEY"] = openai_api_key
    google_api_key='AIzaSyDFx4jp1RNN9ilcomlA6I0wd0lk3W9-ATY'
    os.environ["GOOGLE_API_KEY"] = google_api_key
    #default_model='gpt-3.5-turbo'
    #os.environ["OPENAI_MODEL"] = default_model

    # podcasts videos (testing)
    # https://www.youtube.com/watch?v=HekHk6yLmF0&t=3468s
    # https://www.youtube.com/watch?v=Mde2q7GFCrw&t=5453s
    # https://www.youtube.com/watch?v=3TV3dNJGqGI&t=3491s
    # https://www.youtube.com/watch?v=F80WmftF5YY&t=12s
    # https://www.youtube.com/watch?v=ofAOwyKuKxI
    # https://www.youtube.com/watch?v=aZ-BjJZxNoA
    # https://www.youtube.com/watch?v=h6yIyks16FA
    youtube_podcast_url='https://www.youtube.com/watch?v=0oTMnSwFyn0'

    #result=load_document(youtube_podcast_url)
    #chunks=TranscriptChunking(result, 'gpt-3.5-turbo-16k', 0.8, 0.1)

    #summarize the podcast 
    summary=summarize_podcast_transcript(youtube_podcast_url,'gpt-3.5-turbo',0.8,0.2)

    #print(summary)
    # write summary to a file named podcast_summary.txt
    #with open("podcast_summary.txt", "w") as f:
    #    f.write(summary)

    string_to_pdf(summary, "summary.pdf")





    
