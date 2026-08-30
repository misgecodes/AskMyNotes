


from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_text_splitters import RecursiveCharacterTextSplitter
load_dotenv()
llm = ChatOpenAI(model="gpt-4.1-mini")










with open("faq.txt", "r") as file:
    faq = file.read()

# question = input("Ask a question: ")

# response = llm.invoke(
#     f"""
# You are a customer support assistant.

# Use the following FAQ to answer the user's question.

# FAQ:
# {faq}

# User question:
# {question}
# """
# )


splitter = RecursiveCharacterTextSplitter(
    chunk_size=80, 
    chunk_overlap=0, 
    separators=["\n\n", "\n", " ", ""])

chunks = splitter.create_documents([faq])

print(len(chunks))

# print(response.content)

for i in range(len(chunks)):
    print(f"Chunk {i+1}:")
    print(chunks[i].page_content)
    print("\n")