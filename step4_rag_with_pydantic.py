from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from model.answer import Answer



load_dotenv()
llm = ChatOpenAI(model="gpt-4.1-mini")



with open("faq.txt", "r") as file:
    faq = file.read()

question = input("Ask a question: ")

splitter = RecursiveCharacterTextSplitter(
    chunk_size=80, 
    chunk_overlap=0, 
    separators=["\n\n", "\n", " ", ""])

chunks = splitter.create_documents([faq])

embeddings = OpenAIEmbeddings()
vectorestore = Chroma.from_documents(chunks, embeddings)

# question = "whats your mom's name?"
results = vectorestore.similarity_search(question, k = 1)

structured_llm = llm.with_structured_output(Answer)
response = structured_llm.invoke(
    f"""
You are a customer support assistant.

Use the following FAQ to answer the user's question.

FAQ:
{results[0].page_content}

User question:
{question}
"""
)

print("\nAnswer:", response.answer, "\nSource Snippet:", results[0].page_content, "\nConfidence:", response.confidence)

# print(len(chunks))

# # print(response.content)

# for i in range(len(chunks)):
#     print(f"Chunk {i+1}:")
#     print(chunks[i].page_content)
#     print("\n")



# print("Results:", results[0].page_content)


