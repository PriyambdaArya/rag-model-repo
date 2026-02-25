from dotenv import load_dotenv
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_openai import ChatOpenAI

load_dotenv()

db = Chroma(persist_directory='./chroma_db', embedding_function=OpenAIEmbeddings())
question = input("Ask me: \n\n")
results = db.similarity_search(question, k =6)

context_text = ""
for doc in results:
    context_text += doc.page_content + "\n\n"

llm = ChatOpenAI()
response = llm.invoke(f"Answer the question {question} based on the following context {context_text}")
print(response.content)