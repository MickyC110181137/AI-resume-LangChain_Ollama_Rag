from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama.llms import OllamaLLM
from langchain_community.chat_models import ChatOllama
llm = ChatOllama(model="llama3.1")

from langchain_community.vectorstores import Chroma
from langchain_text_splitters import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
import ollama
import chromadb

# try:
with open(r"D:\高科\專題\面試官\flask\rag\crew.txt", encoding='UTF-8') as f:
    last_question = f.read()
# except Exception as e:
#     print(f"An error occurred: {e}")
# with open("crew.txt", "r") as file:
#     last_question = file.readlines()

text_splitter = CharacterTextSplitter(
    separator="\n\n",
    chunk_size = 1000,
    chunk_overlap=200,
    length_function = len,
    is_separator_regex=False,

)
texts = text_splitter.create_documents([last_question])

print(texts)

vector_store = Chroma.from_documents(texts, OpenAIEmbeddings())

retriever = vector_store.as_retriever(
    search_type = "similarity_score_threshold",
    search_kwargs = {
        "k":3,
        "score_threshold":0.5,
    },
)

from langchain.prompts import PromptTemplate
#Create the prompte from the template.
prompt = PromptTemplate.from_template(
    """Answer the question as precise as possible using the provided context. If the answer is
    not contained in the context, say "answer not available in context" \n\n
    Context: {context}
    Question: {question}
    Answer:

     """
) 

from langchain.schema.runnable import RunnablePassthrough
from langchain.schema.output_parser import StrOutputParser

chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)


# 提问问题并获取回答
response = chain.invoke("船員資格有哪些?")
print(response)
