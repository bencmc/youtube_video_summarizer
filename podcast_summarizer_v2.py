# Others
import os
from youtube_video_chapters import fetch_chapters, load_transcript
from fpdf import FPDF
# LangChain basics
from langchain.chat_models import ChatOpenAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document
# Chat Prompt templates for dynamic values
from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate
)
# Vector Store and retrievals
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chains import RetrievalQA
from langchain.vectorstores import Pinecone
import tiktoken
import pinecone


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

def TranscriptChunking(document):
    docs = []  # Initialize as empty list so it's always defined

    if document is None:
        return docs  # Return an empty list, or raise an exception if you prefer
    elif not isinstance(document, (str, bytes)):
        return docs  # Return an empty list, or raise an exception if you prefer
    else:
        text_splitter = RecursiveCharacterTextSplitter(separators=["\n\n", "\n", " "], chunk_size=100, chunk_overlap=20)
        # Use the text splitter to split the document into chunks
        segmented_documents = text_splitter.split_text(document)
        docs = [Document(page_content=t) for t in segmented_documents]
    return docs

def print_embedding_cost(texts):
    import tiktoken
    enc = tiktoken.encoding_for_model('text-embedding-ada-002')
    total_tokens = sum([len(enc.encode(page.page_content)) for page in texts])
    print(f'Total Tokens: {total_tokens}')
    print(f'Embedding Cost in USD: {total_tokens / 1000 * 0.0004:.6f}')
        
def create_vector_store(docs):
    # Download embeddings from OpenAI
    embeddings = OpenAIEmbeddings()
    # initialize Pinecone
    pinecone.init(api_key=os.environ.get('PINECONE_API_KEY'), environment=os.environ.get('PINECONE_ENV'))
    # deleting all indexes
    # indexes = pinecone.list_indexes()
    # for i in indexes:
    #     print('Deleting all indexes ... ', end='')
    #     pinecone.delete_index(i)
    #     print('Done')

    # creating an index
    index_name = 'podcast-summarizer'
    if index_name not in pinecone.list_indexes():
        print(f'Creating index {index_name} ...')
        pinecone.create_index(index_name, dimension=1536, metric='cosine')
        print('Done!')

    # If you want to delete your vectors in your index to start over, run the code below!
    index = pinecone.Index(index_name)
    index.delete(delete_all='true')
    
    vector_store = Pinecone.from_documents(docs, embeddings, index_name=index_name)
    
    return vector_store

def init_vector_store(video_url):
    # Initialize as empty to make sure it's always defined
    vector_store = None
    
    # Load the transcript
    result = load_transcript(video_url)
    
    # Check if the document is valid
    if result is None:
        return vector_store  # Return None or an empty structure, or raise an exception if you prefer
    elif not isinstance(result, (str, bytes)):
        return vector_store  # Return None or an empty structure, or raise an exception if you prefer
    
    # Split the transcript
    docs = TranscriptChunking(result)
    
    # If TranscriptChunking returns an empty list, you might want to handle it here
    if not docs:
        return vector_store  # Return None or an empty structure, or raise an exception if you prefer
    
    # Create vector store
    vector_store = create_vector_store(docs)
    
    return vector_store

    
# summarize podcast transcript
def summarize_podcast_transcript(model_name, chapter, vector_store, summary_type):

    # Initialize an empty result dictionary
    result = {}
    
    # Validate vector_store
    if vector_store is None:
        return result  # Return empty result or raise an exception if you prefer
    
    
    # Validate summary_type value
    if summary_type not in ["bullet points", "paragraph"]:
        raise ValueError("Invalid summary_type. Accepted values are 'bullet points' or 'paragraph'.")

    # Load ChatOpenAI object
    llm = ChatOpenAI(temperature=0, model=model_name)
    
    system_template = f"""
    You will be given a transcript from a podcast that discusses multiple topics. The transcript will be enclosed within triple backticks (` ``` `).
    Your task is to create a focused, yet comprehensive summary centering on the specific topic requested by the user. 
    - The summary should not exceed 100 words.
    - Emphasize key arguments, contrasting perspectives, and notable insights related to the chosen topic.
    - Enumerate any referenced resources, such as books, websites, films, or documentaries.
    - Remove any redundant or irrelevant details to maintain conciseness.
    - Utilize {summary_type} format to coherently structure and present the essential ideas.
    - Include only information that pertains directly to the topic specified by the user.
    ```{{context}}```

    """

    messages = [
        SystemMessagePromptTemplate.from_template(system_template),
        HumanMessagePromptTemplate.from_template("{question}"),
    ]
    
    CHAT_PROMPT = ChatPromptTemplate.from_messages(messages)
    
    qa = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vector_store.as_retriever(),
        chain_type_kwargs={
            'prompt': CHAT_PROMPT
        }
    )
    
    # Handling a single chapter
    query = chapter
    expanded_topic = qa.run(query)
    
    return {'chapter': chapter, 'summary': expanded_topic}

if __name__ == "__main__":
    # Initialize environment variables
    openai_api_key = "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"
    os.environ["OPENAI_API_KEY"] = openai_api_key
    google_api_key = 'AIzaSyDFx4jp1RNN9ilcomlA6I0wd0lk3W9-ATY'
    os.environ["GOOGLE_API_KEY"] = google_api_key
    pinecone_api_key = '0aa48981-829c-47f4-a0d7-378961cb11be'
    os.environ["PINECONE_API_KEY"] = pinecone_api_key
    pinecone_env = 'asia-southeast1-gcp-free'
    os.environ["PINECONE_ENV"] = pinecone_env
    
    # Sample video URL
    video_url = "https://www.youtube.com/watch?v=KopLe5NZBJc"
    
    # Initialize the vector store
    vector_store = init_vector_store(video_url)
    
    # Check if vector store initialization was successful
    if vector_store is None:
        print("Failed to initialize vector store.")
        exit(1)
    
    # Sample chapter to summarize
    # https://www.youtube.com/watch?v=h6yIyks16FA&t=2s
    # https://www.youtube.com/watch?v=KopLe5NZBJc
    youtube_podcast_url = 'https://www.youtube.com/watch?v=h6yIyks16FA&t=2s'
    chapters_data = fetch_chapters(youtube_podcast_url)
    print(chapters_data)
    chapter_to_summarize = chapters_data['Chapters'][2]
    print(chapter_to_summarize)
    
    # # Specify the model and summary type
    model_name = "gpt-3.5-turbo"
    summary_type = "bullet points"
    
    # Summarize the chapter
    summary_result = summarize_podcast_transcript(model_name, chapter_to_summarize, vector_store, summary_type)
    
    # Display the summary
    print(f"Summary for {chapter_to_summarize}:")
    print(summary_result.get('summary', 'Summary not available'))






    
