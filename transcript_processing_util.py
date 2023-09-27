import os
from typing import Union, List, Optional, Dict
from youtube_chapters_extractor import load_transcript
from fpdf import FPDF
from langchain.chat_models import ChatOpenAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document
from langchain.prompts.chat import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chains import RetrievalQA
import tiktoken
from langchain.vectorstores import Chroma


def convert_string_to_pdf(input_str: str, filename: str = "output.pdf", font_path: str = "ARIALUNI.TTF") -> None:
    """
    Converts a string to a PDF file.
    :param input_str: The string to be converted to PDF.
    :param filename: The name of the output PDF file. Defaults to "output.pdf".
    :param font_path: The path to the font file supporting Unicode characters. Defaults to "ARIALUNI.TTF".
    """
    pdf = FPDF()
    pdf.add_page()
    pdf.add_font("ArialUnicode", style="", fname=font_path, uni=True)
    pdf.set_font("ArialUnicode", size=12)
    pdf.multi_cell(0, 10, input_str)
    pdf.output(filename)


def count_tokens_in_string(string: str, encoding_name: str) -> int:
    """
    Counts the number of tokens in a given string.
    :param string: The input string to count tokens.
    :param encoding_name: The encoding name for the token counting model.
    :return: The number of tokens in the input string.
    """
    encoding = tiktoken.encoding_for_model(encoding_name)
    return len(encoding.encode(string))


def calculate_openai_api_cost(input_text: str, output_text: str, model_name: str) -> float:
    """
    Calculates the cost of OpenAI API calls based on the input and output text.
    :param input_text: The input text string.
    :param output_text: The output text string.
    :param model_name: The name of the model used for the API call.
    :return: The cost of the API call in USD.
    """
    in_costs = {'gpt-3.5-turbo-16k': 0.003, 'gpt-3.5-turbo': 0.0015}
    out_costs = {'gpt-3.5-turbo-16k': 0.004, 'gpt-3.5-turbo': 0.002}
    input_tokens = count_tokens_in_string(input_text, model_name)
    output_tokens = count_tokens_in_string(output_text, model_name)
    input_cost = input_tokens * in_costs[model_name] / 1000.
    output_cost = output_tokens * out_costs[model_name] / 1000.
    return input_cost + output_cost


def chunk_transcript(document: Union[str, bytes]) -> List[Document]:
    """
    Splits the document into chunks.
    :param document: The input document string or bytes.
    :return: A list of Document objects containing the chunks of the input document.
    """
    if document is None or not isinstance(document, (str, bytes)):
        return []
    text_splitter = RecursiveCharacterTextSplitter(separators=["\n\n", "\n", " "], chunk_size=2000, chunk_overlap=200)
    segmented_documents = text_splitter.split_text(document)
    return [Document(page_content=t) for t in segmented_documents]


def display_embedding_cost(texts: List[Document]) -> None:
    """
    Displays the embedding cost of the texts in USD.
    :param texts: A list of Document objects containing the texts.
    """
    enc = tiktoken.encoding_for_model('text-embedding-ada-002')
    total_tokens = sum([len(enc.encode(page.page_content)) for page in texts])
    print(f'Total Tokens: {total_tokens}')
    print(f'Embedding Cost in USD: {total_tokens / 1000 * 0.0004:.6f}')


def initialize_vector_store(video_url: str) -> Optional[Chroma]:
    """
    Initializes the vector store from the transcript of the video URL.
    :param video_url: The URL of the video to load the transcript.
    :return: The initialized Chroma object containing the vector store or None if unsuccessful.
    """
    result = load_transcript(video_url)
    if result is None or not isinstance(result, (str, bytes)):
        return None
    docs = chunk_transcript(result)
    if not docs:
        return None
    return Chroma.from_documents(docs, OpenAIEmbeddings())


def generate_video_summary(model_name: str, chapter: str, vector_store: Chroma, summary_type: str) -> Dict[str, str]:
    """
    Generates a summary of the video transcript.
    :param model_name: The name of the model used for summarization.
    :param chapter: The chapter of the video to summarize.
    :param vector_store: The Chroma object containing the vector store.
    :param summary_type: The type of summary, either "bullet points" or "paragraph".
    :return: A dictionary containing the chapter and the generated summary.
    """
    if vector_store is None:
        return {}
    if summary_type not in ["bullet points", "paragraph"]:
        raise ValueError("Invalid summary_type. Accepted values are 'bullet points' or 'paragraph'.")
    llm = ChatOpenAI(temperature=0, model=model_name)
    system_template = f"""
        You are tasked with generating a concise, clear, and relevant summary based on a provided discussion transcript. Please pay meticulous attention to the following guidelines to ensure the highest quality summary:

        - **Word Limit:** Your summary must strictly adhere to a maximum of 100 words. Any excess will render the summary invalid. Be succinct and choose words that convey maximum meaning.
        - **Format:** Employ {summary_type} format, focusing on clarity and succinctness to relay explicit details, distinctive insights, and specific points from the transcript.
        - **Relevance:** Only include pertinent and topic-specific information and insights. Eliminate any generalized, unrelated, or superfluous content.
        - **References & Names:** Should there be any references, list them briefly by title and author. Use participants’ known names if provided.

        **Mandatory Compliance:**
        - Confine your summary to information explicitly stated in the transcript, excluding any form of assumptions, inferences, or embellishments.
        - Do not allude to the presence or absence of references or speculate on further exploration unless specific resources are explicitly cited in the transcript.

        ```{{context}}```

        **Urgent Reminder:** Precision, focus, and adherence to the 100-word limit are imperative. Every word should significantly contribute to the understanding of the topic. Non-compliance with the guidelines and word limit will lead to summary rejection. Please review your summary carefully to ensure strict adherence to the word limit and guidelines.
    """


    messages = [SystemMessagePromptTemplate.from_template(system_template), HumanMessagePromptTemplate.from_template("{question}")]
    chat_prompt = ChatPromptTemplate.from_messages(messages)
    qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=vector_store.as_retriever(), chain_type_kwargs={'prompt': chat_prompt})
    expanded_topic = qa.run(chapter)
    return {'chapter': chapter, 'summary': expanded_topic}


if __name__ == "__main__":
    None






    
