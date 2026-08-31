from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

from model.answer import Answer
from type.graph_state import GraphState
from langchain_community.vectorstores import Chroma




def retrieve(state: GraphState, vectorstore: Chroma) -> dict:
    
    load_dotenv()
    llm = ChatOpenAI(model="gpt-4.1-mini")


    if state["retries"] > 0:
     structured_llm = llm.with_structured_output(Answer)
     response = structured_llm.invoke(
        f"""
          The user asked {state['question']}. you previously answered {state['answer']} 
          with low confidence. 
          Please reformulate the question to help me find a better answer. 

    """
    )
    # if this is a retry (retries > 0), reformulate the query
    # using the previous low-confidence answer before searching again
    

    results = vectorstore.similarity_search(response.answer, k=1)
    state["retries"] -= 1
    state["answer"] = response.answer
    state["confidence"] = response.confidence
    return {"chunks": results}
