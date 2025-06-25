from typing import Annotated,Literal
from langgraph.graph import StateGraph,START,END
from langgraph.graph.message import add_messages
from pydantic import BaseModel,Field
from typing_extensions import TypedDict

class MessageClassifier(BaseModel):
    message_type: Literal["multiple","prediction"] = Field(
        ...,
        description="Classify if the message is multiple or prediction",
    )

class State(TypedDict):
    message: Annotated[list,add_messages]
    message_type: str | None
    next: str

def classify_message(state: State) -> State:
    messsage = state["message"][-1].content
    message_type = classify_xlm(messsage)
    validated_type = MessageClassifier(message_type=message_type) 
    return {"message_type":validated_type.message_type}

def router(state: State) -> State:
    message_type = state.get("message_type")
    return {"next":message_type}

def prediction_agent(state: State ,prompt : str) -> State:
    message = state["message"][-1].content
    respond = multiple_answer(text = message,system_prompt=prompt)
    return {"message":respond}

def multiple_agent(state: State ,prompt : str) -> State:
    message = state["message"][-1].content
    respond = multiple_answer(text = message,system_prompt=prompt)
    return {"message":respond}

def graph_init():
    graph_builder = StateGraph(State)

    graph_builder.add_node("classifier",classify_message)
    graph_builder.add_node("router",router)
    graph_builder.add_node("prediction_agent",prediction_agent)
    graph_builder.add_node("multiple_agent",multiple_agent)

    graph_builder.add_edge(START,"classifier")
    graph_builder.add_edge("classifier","router")

    graph_builder.add_conditional_edges(
        "router",
        lambda state: state.get("next"),
        {
            "prediction": "prediction_agent",
            "multiple": "multiple_agent"
        }
    )
    graph_builder.add_edge("prediction_agent",END)
    graph_builder.add_edge("multiple_agent",END)
    return graph_builder.compile()