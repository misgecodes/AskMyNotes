from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()
llm = OpenAI()

with open("faq.txt", "r") as file:
    faq = file.read()

question = input("Ask a question: ")

response = llm.responses.create(
    model="gpt-4.1-mini",
    input=f"""
You are a customer support assistant.

Use the following FAQ to answer the user's question.

FAQ:
{faq}

User question:
{question}
"""
)

print(response.output_text)