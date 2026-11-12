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


import ollama
import json
import os 

messages = [   {
            "role": "system",
            "content": "You are an AI/ML mentor, Your name is Aryan AI. Keep responses simple and brief. Explain concepts in detail only when the user asks."
        }
    ]
if os.path.exists("chat.json"):
 try :
      chat =open("chat.json","r")
      messages = json.load(chat)
 except json.JSONDecodeError:
    print("Invalid chat history. Starting a new conversation.")
while True:
    user_input = input("You: ")
    if user_input.lower() == "exit":
            break
    if not user_input.strip():
        continue
    if user_input.lower() == "clear":
        messages[:] = messages[0:1]
        print("AryanAI: Conversation history cleared!")
        continue
    messages.append({
        "role":"user",
        "content": user_input
    })
    try:
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
     chat.seek(0)
     chat.truncate()
     json.dump(messages,chat)
     chat.flush()
    except Exception as e:
      messages.pop()
      print("AryanAI: Unable to connect to Ollama. Please check if Ollama is running.")
chat.close()