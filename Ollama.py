
from rag.LangChainOllama import Initialize_LLM, chatLLM

# Ollama------------------------------------------------------
# 逐行顯示
# def chunk():
#     for chunk in stream:
#         print(chunk['message']['content'], end='', flush=True)

# # 限制為五次對話
# for i in range(0,5):
#     #input()是python在terminal輸入的功能，str()是改成字串
#     userinput = str(input())
#     stream = ollama.chat(
#         model='llama3.1',
#         #content才是對話的輸入要放的地方
#         messages=[{'role': 'user', 'content': f'{userinput}'}],
#         stream=True,
#     )
#     chunk()


# Ollama------------------------------------------------------
if __name__ == "__main__":
    LLM = Initialize_LLM()
    chatmodel = LLM[0]
    retriever= LLM[1]
    retrieved_docs = retriever.invoke("船員資格")
    print(f'傳回 {len(retrieved_docs)} 筆資料')
    UserQuestiion = "曾任漁船普通船員資歷滿一年以上。請問是否具有船員資格。"
    print(chatLLM(UserQuestiion,chatmodel,retriever))
    