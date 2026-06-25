import os
from dotenv import load_dotenv
from groq import Groq
from ingest import retrieve

load_dotenv()  # reads your .env file into the environment

client = Groq(api_key=os.getenv("GROQ_API_KEY"))


def ask_groq(query, context_chunks):
    context = "\n\n".join(context_chunks)
    prompt = f"""Use the following context to answer the question. If the answer isn't in the context, say so.

Context:
{context}

Question: {query}"""

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=500
    )
    return response.choices[0].message.content


if __name__ == "__main__":
    query = "What projects has this person worked on?"
    context_chunks = retrieve(query)
    answer = ask_groq(query, context_chunks)
    print(answer)