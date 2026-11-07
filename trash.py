import ollama

response = ollama.chat(
    model="gemma4:e2b",
    messages=[
        {
            "role":"system",
            "content":"your are an AI/Ml mentor. you are supposed to explain concepts in simple language with practical examples.And do not just directly give the entire code unless the user asks when the user is solving something and only provide show the entire code when required like for teching the working of a code."
        },
        {
            "role": "user",
            "content": "Explain machine learning."
        }
    ]
)

print(response["message"]["content"])