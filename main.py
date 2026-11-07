import ollama

messages = [   {
            "role": "system",
            "content": "You are an AI/ML mentor. Keep responses simple and brief. Explain concepts in detail only when the user asks."
        }
    ]

while True:
    user_input = input("You: ")
    if user_input.lower() == "exit":
            break
    if not user_input.strip():
        continue
    messages.append({
        "role":"user",
        "content": user_input
    })
    response = ollama.chat(
        model="gemma4:e2b",
        messages= messages,
        stream=True,
        options={
    "temperature": 0.2
    }
        )
    full_response = ""
    for chunk in response:
     full_response += chunk["message"]["content"]
     print(chunk["message"]["content"],end="",flush=True)
    print()
    messages.append({
        "role": "assistant",
        "content": full_response
    })