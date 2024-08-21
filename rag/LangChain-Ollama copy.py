import ollama
import chromadb
from langchain.chains import create_retrieval_chain
from langchain.embeddings import OllamaEmbeddings
from langchain.vectorstores import Chroma
from langchain_ollama.llms import OllamaLLM

# Read documents from an external .txt file
with open("./rag/crew.txt", "r") as file:
    documents = file.readlines()

client = chromadb.Client()
collection = client.create_collection(name="docs")
# print(documents)

# 使用 enumerate 函數遍歷 documents 列表。enumerate 函數會返回一個包含索引 i 和對應文檔 d 的元組。
for i, d in enumerate(documents):
    # 調用 ollama.embeddings 函數，使用模型 mxbai-embed-large 生成文檔 d 的嵌入。d.strip() 用於去除文檔中的換行符和多餘的空白字符。
    response = ollama.embeddings(model="mxbai-embed-large", prompt=d.strip()) #去除換行
    # 這行代碼從 response 中提取嵌入向量，並將其存儲在變量 embedding 中。
    embedding = response["embedding"]
    # 代碼將文檔的嵌入向量添加到集合 collection 中。ids 是文檔的唯一標識符，
    # 使用的是文檔的索引 i。embeddings 是嵌入向量，documents 是去除換行符後的原始文檔內容。
    collection.add(
        ids=[str(i)],
        embeddings=[embedding],
        documents=[d.strip()]  
    )

# An example prompt
prompt = "船員資格有哪些?"
# 创建 Langchain 的嵌入
embeddings = OllamaEmbeddings(model="mxbai-embed-large")

# 创建 Langchain 的向量数据库
docsearch = Chroma(
    embedding_function=embeddings,
    persist_directory="chroma_db"  # 存储向量数据库的路径
)
docsearch.add_documents(documents)

# 创建检索问答链
qa = create_retrieval_chain(
    llm=OllamaLLM(model_name="llama3.1"),
    chain_type="stuff",
    retriever=docsearch.as_retriever()
)

# 使用链进行查询
result = qa({"query": prompt})
print(result["answer"])

