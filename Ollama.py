import ollama

# Ollama------------------------------------------------------
# 逐行顯示
def chunk():
    for chunk in stream:
        print(chunk['message']['content'], end='', flush=True)

# 限制為五次對話
for i in range(0,5):
    #input()是python在terminal輸入的功能，str()是改成字串
    userinput = str(input())
    stream = ollama.chat(
        model='llama3.1',
        #content才是對話的輸入要放的地方
        messages=[{'role': 'user', 'content': f'{userinput}'}],
        stream=True,
    )
    chunk()


# Ollama------------------------------------------------------

