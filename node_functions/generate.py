from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from schemas.answer import Answer
from type.graph_state import GraphState
from langchain_community.vectorstores import Chroma


def generate(state: GraphState, vectorstore: Chroma) -> dict:

    load_dotenv()
    llm = ChatOpenAI(model="gpt-4.1-mini")

    results = vectorstore.similarity_search(state["question"], k=1)
    structured_llm = llm.with_structured_output(Answer)
    response = structured_llm.invoke(f"""
    You are a customer support assistant.
    
    Use the following FAQ to answer the user's question.
    
    FAQ:
    {results[0].page_content}
    
    User question:
    {state["question"]}
    """
    )
    return {"chunks": results, "answer": response.answer, "confidence": response.confidence}


def make_generate(vectorstore: Chroma):
    def generate_node(state: GraphState) -> dict:
        return generate(state, vectorstore)
    return generate_node