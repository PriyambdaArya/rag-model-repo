from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings


load_dotenv()

pdf_file_path = "data/test_resume.pdf"

text_split_config = RecursiveCharacterTextSplitter(
    chunk_size = 1000,
    chunk_overlap = 200,
    add_start_index = True
)

print("Loading pdf file...")
loader = PyPDFLoader(pdf_file_path)
pages = loader.load()

print("Begining the split process...")
splits = text_split_config.split_documents(pages)

print("Splitting completed! Total number of chunks created are", len(splits))

print("Saving splits to chroma DB...")
vectorstores = Chroma.from_documents(
    documents = splits,
    embedding = OpenAIEmbeddings(),
    persist_directory = "./chroma_db" 
)