from type.graph_state import GraphState
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from model.answer import Answer
from helper_functions.retrieve_function import retrieve




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

if response.confidence < 0.5:
    graph_state = GraphState(question=question, answer=response.answer, retries=2)
    response = retrieve(graph_state, vectorestore)

print("\nAnswer:", response.answer, "\nSource Snippet:", results[0].page_content, "\nConfidence:", response.confidence)




# print("Results:", results[0].page_content)


