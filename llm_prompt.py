from dotenv import load_dotenv
load_dotenv()

from openai import OpenAI
import os

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def generate_response(context, question, user_profile):
    profile_prefix = (
        "Explain in a simple tone as the student struggles with Physics."
        if user_profile == "weak"
        else "Explain in a concise and technical tone."
    )

    prompt = f"""{profile_prefix}
Context:
{context}

Question:
{question}

Answer:"""

    response = client.chat.completions.create(
        model="gpt-4",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7
    )

    return response.choices[0].message.content
