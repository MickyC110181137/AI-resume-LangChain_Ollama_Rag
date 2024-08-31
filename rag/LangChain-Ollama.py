from langchain_community.document_loaders import TextLoader

# -------------載入文件----------------
loader = TextLoader(file_path='./rag/crew.txt',encoding="UTF-8")
docs = loader.load()
# print(docs[0])

# -------------分割文件-----------------
from langchain.indexes import VectorstoreIndexCreator
from langchain_ollama import OllamaEmbeddings
from langchain_ollama import ChatOllama
from langchain_huggingface import HuggingFaceEmbeddings
model_name = "sentence-transformers/all-MiniLM-L6-v2"
model_kwargs = {'device': 'cpu'}
encode_kwargs = {'normalize_embeddings': False}
embeddings_model = HuggingFaceEmbeddings(
    model_name=model_name,
    model_kwargs=model_kwargs,
    encode_kwargs=encode_kwargs
)

# embeddings_model = OllamaEmbeddings(
#     model="llama3.1"
# )

chatmodel = ChatOllama(
    model = "llama3.1",
    temperature = 0.8,
    num_predict = 1024
)
index = VectorstoreIndexCreator(embedding=embeddings_model).from_loaders([loader])
# query = "船員資格有哪些?"
# response = index.query(llm=chatmodel, question=query)
# print(response)

# test_doc = docs[0].page_content[:500]
# test_doc

from langchain_text_splitters import(CharacterTextSplitter, RecursiveCharacterTextSplitter)
text_splitter = RecursiveCharacterTextSplitter(
    separators=' \n',
    chunk_size=10,
    chunk_overlap=2
)
chunks = text_splitter.split_documents(docs)
print(chunks)

# -----------------------文字轉向量---------------------------
from langchain_community.vectorstores import Chroma
Chroma.from_documents(
    documents=chunks,
    embedding=embeddings_model,
    persist_directory='./chroma_db/db',
    collection_metadata={"hnsw:space":"cosine"}
)
db = Chroma(
    persist_directory='./chroma_db/db',
    embedding_function=embeddings_model
)

# print(db.search('船員資格',k=2,search_type="similarity"))
# print(db.similarity_search_with_relevance_scores('二等船長適任證書',k=2))


retriever = db.as_retriever(search_type="similarity",
                            search_kwargs={"k": 6})
retrieved_docs = retriever.invoke("船員資格")
print(f'傳回 {len(retrieved_docs)} 筆資料')

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough

# from langchain.tools.retriever import create_retriever_tool
# tool = create_retriever_tool(
#     retriever=retriever,
#     name="retriever_by_car_regulations",
#     description="搜尋並返回船員資格包含哪些資料",
# )
# tools = [tool]

from langchain_core.prompts import MessagesPlaceholder
from langchain_core.prompts import ChatPromptTemplate
prompt = ChatPromptTemplate.from_messages([
    ('system','你是一位善用工具的好助理, '
              '請自己判斷上下文來回答問題, 不要盲目地使用工具'),
    MessagesPlaceholder(variable_name="chat_history"),
    ('human','{input}'),
    MessagesPlaceholder(variable_name="agent_scratchpad")
])

str_parser = StrOutputParser()
template = (
    "請根據以下內容加上自身判斷回答問題:\n"
    "{context}\n"
    "問題: {question}"
    )
prompt = ChatPromptTemplate.from_template(template)
chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | prompt
    | chatmodel
    | str_parser
)

print(chain.invoke("曾任漁船普通船員資歷滿一年以上。請問是否具有船員資格。"))
# while True:
#     msg = input("我說：")
#     if not msg.strip():
#         break
#     for chunk in memory_chain.stream(
#         {"input": msg},
#         config={"configurable": {"session_id": "test_id"}}):
#         if 'output' in chunk:
#             print(f"AI 回覆：{chunk['output']}", end="", flush=True)
#     print('\n')

