運用Ollama與Flask架設LLM服務
--------------------------
8/21
目前進度是本地Flask.py與Ollama.py初版完成
ollamaRag.py是目前完成度最高的RAG,但是目前有一個問題是向量搜尋,他會搜尋最相近的句子,但是缺少了上下文關係,導致雖然有回答道我的問題,但是只回答了一點 
我試著使用langchain的chunk_size與chunk_overlap,因為LangChain迭代太快,之前的程式需要更新網上的資料太混亂,還在找怎麼做比較快速正確
(chunk_size是分割的大小，chunk_overlap是重疊的部分)

