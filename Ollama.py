import ollama

# Ollama------------------------------------------------------
# 逐行顯示
def chunk():
    for chunk in stream:
        print(chunk['message']['content'], end='', flush=True)

# 限制為五次對話
for i in range(0,5):
    userinput = str(input())
    stream = ollama.chat(
        model='llama3.1',
        messages=[{'role': 'user', 'content': f'{userinput}'}],
        stream=True,
    )
    chunk()


# Ollama------------------------------------------------------

