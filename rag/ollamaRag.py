import ollama
import chromadb

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

# Generate an embedding for the prompt and retrieve the most relevant doc
response = ollama.embeddings(
    prompt=prompt,
    model="mxbai-embed-large"
)
# print(response)

# 這段代碼調用 collection.query 方法來查詢集合 collection。query_embeddings 
# 是一個包含查詢嵌入向量的列表，這裡使用的是之前生成的嵌入向量 response["embedding"]。
# n_results=1 表示我們希望檢索到最相關的一個文檔。
results = collection.query(
    query_embeddings=[response["embedding"]],
    n_results=1
)
# 這行代碼從查詢結果 results 中提取最相關的文檔內容。results['documents'] 是一個包含文檔的列表，
# [0][0] 表示我們取第一個結果中的第一個文檔
data = results['documents'][0][0]


print(data)
print(prompt)
# Generate a response combining the prompt and data we retrieved
output = ollama.generate(
    model='llama3.1',
    prompt=f"Using this data: {data}. Respond to this prompt: {prompt}，並盡可能列出所有選項",
    stream=True,
)

for chunk in output:
  print(chunk['response'], end='', flush=True)
  
# print(output['response'])



