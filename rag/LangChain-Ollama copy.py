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

from langchain_core.prompts import MessagesPlaceholder
from langchain_core.prompts import ChatPromptTemplate


str_parser = StrOutputParser()

from langchain.memory import ChatMessageHistory

from langchain_core.runnables.history import RunnableWithMessageHistory # 用於將 Runnable 物件與訊息記憶功能結合，從而能夠處理包含對話歷史的對話流程
from langchain.tools.retriever import create_retriever_tool
from langchain.agents import AgentExecutor, create_react_agent

tool = create_retriever_tool(
    retriever=retriever,
    name="retriever_by_crew_regulations",
    description="搜尋並返回船員資格內容",
)
tools = [tool]
promptHistory = ChatPromptTemplate.from_messages([
    ('system','你是一位善用工具的好助理, '
              '請自己判斷上下文來回答問題, 不要盲目地使用工具'),
    MessagesPlaceholder(variable_name="chat_history"),
    ('human','{question}'),
    MessagesPlaceholder(variable_name="agent_scratchpad"),
])

agent = create_react_agent(chatmodel, tools, promptHistory)
agent_executor = AgentExecutor(agent=agent, tools=tools)
# memory = ChatMessageHistory()
# memories = {'0': memory, '1': ChatMessageHistory()} 
# # 定義了一個字典 memories，其中包含了不同會話的記憶體。每個記憶體對應一個 session_id，例如 '0' 和 '1'

memory = ChatMessageHistory(
    session_id="test_id"
)

def window_messages(chain_input):
    if len(memory.messages) > 6:
        cur_messages = memory.messages
        memory.clear()
        for message in cur_messages[-6:]:
            memory.add_message(message)
    return
def add_history(agent_executor):
    agent_with_chat_history = RunnableWithMessageHistory(
        agent_executor,
        lambda session_id: memory,
        input_messages_key="input",
        history_messages_key="chat_history",
    )
    memory_chain = (
        RunnablePassthrough.assign(messages=window_messages)
        | agent_with_chat_history
    )
    return memory_chain
memory_chain = add_history(agent_executor)

# chainHistory = (
#     # {"context": retriever, "question": RunnablePassthrough()}
#     promptHistory
#     | chatmodel
#     | str_parser
# )
# '0': memory 表示一個預先定義的 memory 物件、'1': ChatMessageHistory() 表示一個新的對話歷史記錄物件，專門用於存儲會話歷史
# 所以要記錄同一個訊息歷史就用0，想要用新的對話紀錄就用1
while True:
    msg = input("我說：")
    if not msg.strip():
        break
    for chunk in memory_chain.stream(
        {"input": msg},
        config={"configurable": {"session_id": "test_id"}}):
        if 'output' in chunk:
            print(f"AI 回覆：{chunk['output']}", end="", flush=True)
    print('\n')


# print(
#     chat_with_history.invoke(
#         {"question": "船員資格有哪些?"},
#         config={"configurable": {"session_id": "foo"}} # 指定會話 ID (session_id) 為 '0'，所以這次對話會使用 memories 中 '0' 對應的記憶體物件來存儲和檢索對話歷史
#     ) # 取這個結果中的文本內容
# )

# while True:
#     msg = input("我說:")
#     if not msg.strip():
#         break
#     for chunk in chat_history.stream(
#             {"question":msg},
#             config={"configurable": {"session_id": "0"}}):
#         if 'output' in chunk:
#             print(f"AI回復: {chunk['output']}",end="",flush=True)
#     print('\n')
