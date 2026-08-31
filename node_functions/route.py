

from type.graph_state import GraphState


def route(state: GraphState) -> str:
    if state["confidence"] >= 0.9 or state["retries"] >= 2:
        return "end"
    return "reformulate"