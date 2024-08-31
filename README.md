運用Ollama與Flask架設LLM服務
--------------------------
8/21  
目前進度是本地Flask.py與Ollama.py初版完成  
ollamaRag.py是目前完成度最高的RAG,但是目前有一個問題是向量搜尋,他會搜尋最相近的句子,  
但是缺少了上下文關係,導致雖然有回答道我的問題,但是只回答了一點   
我試著使用langchain的chunk_size與chunk_overlap,因為LangChain迭代太快,之前的程式需要更新網上的資料太混亂,還在找怎麼做比較快速正確  
(chunk_size是分割的大小，chunk_overlap是重疊的部分)  

8/31
pip uninstall torch torchvision torchaudio
pip install torch==2.3.0 torchvision==0.18.0 torchaudio==2.3.0
Initially, I got this problem after the installation of torchvision, torch was working fine before that.
fbgemm.dll
完成LangChain-Ollama完全本地的RAG初版
使用Ollama3.1作為聊天模型
使用sentence-transformers/all-MiniLM-L6-v2作為文字分割模型
使用RecursiveCharacterTextSplitter文字分割
使用Chroma設定向量資料庫(from langchain_community.vectorstores import Chroma)
使用retriever = db.as_retriever(search_type="similarity",
                            search_kwargs={"k": 6})設定檢索資料
使用from langchain_core.prompts import ChatPromptTemplate設定聊天提示模板
最後使用
chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | prompt
    | chatmodel
    | str_parser
)問答
print(chain.invoke("曾任漁船普通船員資歷滿一年以上。請問是否具有船員資格。"))
