import os
from groq import Groq

# Initialize the Groq client
client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

try:
    response = client.chat.completions.create(
        messages=[
            {
                "role": "user",
                "content": "Hello, this is a test. Please reply with a short greeting."
            }
        ],
        model="llama-3.1-8b-instant" # This is a very fast, free model on Groq
    )
    print("Success! Groq says:", response.choices[0].message.content)
except Exception as e:
    print("Error:", e)