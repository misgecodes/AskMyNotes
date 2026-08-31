from type.graph_state import GraphState
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from schemas.answer import Answer
from node_functions.generate import generate, make_generate
from node_functions.reformuate import reformulate
from node_functions.route import route

from langgraph.graph import StateGraph, END


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

graph_state = GraphState(question=question, answer="", confidence=0.0, retries=0)
new_state  = generate(graph_state, vectorestore)

# while route(new_state) == "reformulate":
#     new_state = reformulate(new_state)
#     new_state  = generate(new_state, vectorestore)

builder = StateGraph(GraphState)
builder.add_node("generate", make_generate(vectorestore))
builder.add_node("reformulate", reformulate)

builder.set_entry_point("generate")
builder.add_conditional_edges("generate", route, {"end": END, "reformulate": "reformulate"})
builder.add_edge("reformulate", "generate")

graph = builder.compile()



result = graph.invoke({"question": question, "chunks": [], "answer": "", "confidence": 0.0, "retries": 0})
print(result["answer"], "Confidence:", result["confidence"], "Retries:", result["retries"], "last question:", result["question"])





# print("Results:", results[0].page_content)


