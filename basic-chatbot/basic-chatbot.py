from typing import Annotated, TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage

# 1. State — a list of BaseMessage, with a built-in reducer that appends
class State(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]

# 2. LLM
llm = ChatAnthropic(model="claude-sonnet-4-5")

# 3. Node — takes current state, calls the LLM, returns a partial update
def chatbot(state: State) -> dict:
    response: AIMessage = llm.invoke(state["messages"])
    return {"messages": [response]}   # only the new AIMessage is returned

# 4. Build the graph
graph = StateGraph(State)
graph.add_node("chatbot", chatbot)
graph.add_edge(START, "chatbot")
graph.add_edge("chatbot", END)

# 5. Compile with a checkpointer so state persists across turns
memory = MemorySaver()
app = graph.compile(checkpointer=memory)

# 6. The iterative loop — this is what makes it a real conversation
config = {"configurable": {"thread_id": "chat-1"}}

print("Chatbot ready. Type 'quit' to exit.")
while True:
    user_input = input("You: ")
    if user_input.lower() in ("quit", "exit"):
        break

    result = app.invoke(
        {"messages": [HumanMessage(content=user_input)]},
        config=config
    )
    print("Bot:", result["messages"][-1].content)