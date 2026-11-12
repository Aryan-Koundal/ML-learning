import ollama
import json
import os 

system_message = {
            "role": "system",
            "content": "You are an AI/ML mentor, Your name is Aryan AI. Keep responses simple and brief. Explain concepts in detail only when the user asks."
        }
messages = [system_message]
if os.path.exists("chat.json"):
 try :
      with open("chat.json", "r") as chat:
       messages = json.load(chat)
       if (not isinstance(messages, list) 
           or not messages
           or not all(isinstance(msg, dict) and "role" in msg and "content" in msg and msg["role"] in ("system", "user", "assistant")and isinstance(msg["content"], str)for msg in messages) 
           or messages[0].get("role") != "system"):
          messages = [system_message]
 except json.JSONDecodeError:
    messages = [system_message]
    print("Invalid chat history. Starting a new conversation.")
while True:
    user_input = input("You: ")
    if user_input.lower() == "exit":
            break
    if not user_input.strip():
        continue
    if user_input.lower() == "clear":
        messages[:] = messages[0:1]
        with open("chat.json","w") as chat:
           json.dump(messages,chat,indent=2)
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
     print("AryanAI: ", end="")
     for chunk in response:
      full_response += chunk["message"]["content"]
      print(chunk["message"]["content"],end="",flush=True)
     print()
     messages.append({
        "role": "assistant",
        "content": full_response
     })
     with open("chat.json", "w") as chat:
      json.dump(messages, chat, indent=2)
    except Exception as e:
      messages.pop()
      print("AryanAI: Unable to connect to Ollama. Please check if Ollama is running.")