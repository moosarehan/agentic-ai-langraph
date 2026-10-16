# Resume Chat in LangGraph

## 1. What is the resume chat feature?

**Resume chat means opening an existing conversation, displaying its previous messages, and continuing that same conversation with its saved context.**

Imagine you have two chats:

- In one chat, you are learning Python.
- In another chat, you are planning a trip.

When you click the Python chat in the sidebar, its messages should appear. Your next question should continue the Python discussion. When you switch to the travel chat, that chat's messages and context should become active instead.

The application therefore needs to remember two things: **which conversation is selected** and **what state belongs to that conversation**.

```text
Sidebar                         Main chat area
----------------------          ------------------------------
Python questions  <selected>    User: What is a Python list?
Trip planning                   Assistant: A list stores items...
                                User: How do I add another item?
```

Here, "resume chat" means continuing a saved conversation after a completed turn. LangGraph also supports resuming interrupted workflows, but that is a separate flow involving interrupts and resume commands.

## 2. The main idea: one conversation has one thread ID

A **thread ID** is the identifier used to select a conversation's saved graph state.

Think of a thread ID as a notebook label. Each notebook contains a different conversation. To continue writing in a particular notebook, you must open that notebook again.

| Conversation | Example thread ID | Saved context |
| --- | --- | --- |
| Python questions | `python-chat-a` | Python discussion |
| Trip planning | `travel-chat-b` | Travel discussion |
| Interview preparation | `interview-chat-c` | Interview discussion |

These short IDs are for explanation. In the implementation, we can generate UUID strings dynamically.

The rule is simple:

```text
New conversation       -> new thread ID
Another message        -> same thread ID
Open an old conversation -> that old conversation's thread ID
```

Creating a new ID for every message would start a fresh conversation each time. Reusing one fixed ID for every sidebar chat would mix them into the same thread.

## 3. What we need to implement it

| Part | Responsibility |
| --- | --- |
| UUID generator | Creates an ID when the user starts a new conversation. |
| Current thread ID | Records which conversation is currently active. |
| Chat list / registry | Provides the sidebar entries and their IDs. |
| LangGraph checkpointer | Saves graph state for each thread. |
| `get_state(config)` | Reads the selected thread's latest saved state. |
| `invoke()` or `stream()` | Runs the next turn using the selected thread's configuration. |
| Frontend message history | Provides the messages currently rendered on screen. |

The sidebar list and the saved graph state have different jobs. The list helps the user **find a chat**; checkpoints let the application **restore its state**.

LangGraph supports thread-scoped state through checkpointers. The `thread_id` is supplied inside the graph configuration. [Official persistence documentation](https://docs.langchain.com/oss/python/langgraph/persistence).

## 4. Step 1: enable checkpointing in the backend

A thread ID identifies the state, but does not store it by itself. Compile the graph with a checkpointer:

```python
from langgraph.checkpoint.memory import InMemorySaver

# builder is your StateGraph after adding its nodes and edges.
checkpointer = InMemorySaver()
chatbot = builder.compile(checkpointer=checkpointer)
```

Your message state should also use a message reducer so that a new turn updates the conversation rather than replacing its message list:

```python
from typing import Annotated, TypedDict
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages


class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
```

With this design, the model node reads `state["messages"]` and returns its new assistant message. Existing messages remain available through the thread's checkpointed state. The reducer merges messages, including updates by message ID; it is more than plain list concatenation.

The existing backend in this repository, `chatbot-in-langraph/langraph_backend.py`, already uses `InMemorySaver` and `add_messages`.

**Important storage detail:** `InMemorySaver` keeps data in the running process. Its contents disappear when that process restarts. To reopen chats after an application restart, use a durable checkpointer, such as a file-backed SQLite implementation for local development or an appropriate database-backed implementation for deployment. [Persistence and restart behavior](https://docs.langchain.com/oss/python/langgraph/persistence).

Keep the checkpointer and compiled graph alive across frontend reruns. Recreating an empty in-memory checkpointer on every rerun loses the earlier state.

## 5. Step 2: create a new thread ID when New Chat is clicked

Python provides UUID generation through its standard library:

```python
from uuid import uuid4

new_thread_id = str(uuid4())
```

An example result is:

```text
4fa38eab-43f2-4437-a1e8-bc42af40329d
```

Convert the UUID to a string so it is straightforward to store and pass in configuration.

The New Chat action should:

1. Generate a new thread ID.
2. Register the chat in the sidebar's chat list.
3. Set that ID as the current thread ID.
4. Clear the displayed message history.

For example, inside a Streamlit app:

```python
import streamlit as st
from uuid import uuid4


def start_new_chat():
    thread_id = str(uuid4())
    st.session_state["chat_threads"].append(thread_id)
    st.session_state["current_thread_id"] = thread_id
    st.session_state["message_history"] = []
```

This function assumes `chat_threads` has already been initialized. A complete initialization example appears later.

Clearing the display does not delete the old conversation. It only prepares the screen for the newly selected empty chat. Also, generating an ID does not itself create a LangGraph checkpoint; the chat may remain empty until its first graph execution.

## 6. Step 3: pass the current thread ID on every turn

Create configuration from the currently selected ID:

```python
current_thread_id = st.session_state["current_thread_id"]
config = {"configurable": {"thread_id": current_thread_id}}
```

### Using `invoke()`

```python
from langchain_core.messages import HumanMessage

result = chatbot.invoke(
    {"messages": [HumanMessage(content=user_input)]},
    config=config,
)
```

With a checkpointer and message reducer, the graph loads the selected thread's saved state and incorporates the new user message. Your model node receives that accumulated state and can use it as conversation context.

Send only the new user message for the new turn in this pattern. Resending the entire displayed history as newly created messages can duplicate conversation content.

### Using `stream()`

The same configuration rule applies when streaming:

```python
for update in chatbot.stream(
    {"messages": [HumanMessage(content=user_input)]},
    config=config,
    stream_mode="updates",
):
    print(update)
```

This example uses node updates, not token-by-token text. The choice of streaming mode changes what the frontend receives; the selected thread ID still determines which conversation is continued.

Build the configuration after selecting the chat. A configuration created once with a hard-coded ID will keep targeting that ID even if the sidebar display changes.

## 7. Step 4: load the chat list into the sidebar

When the frontend opens, it needs a list of conversations to show. For a small demo, keep IDs in session state. For a restart-safe application, fetch saved chat records from durable application storage.

A useful chat record might look like this:

```json
{
  "thread_id": "4fa38eab-43f2-4437-a1e8-bc42af40329d",
  "title": "Python questions",
  "user_id": "current-user",
  "created_at": "2026-10-16T10:00:00Z"
}
```

The title is a readable label; the thread ID is the stable key. Two chats can have the same title without sharing the same ID.

A conceptual startup flow is:

```text
Frontend opens
    -> fetch this user's saved chat records
    -> display titles in the sidebar
    -> keep each title connected to its thread ID
    -> restore the last selected chat, or select/create a chat
```

`load_chat_records(user_id)` would be an application-defined database function, not a built-in method on a compiled LangGraph workflow. Similarly, a local workflow does not automatically populate a sidebar for you.

Some checkpointer APIs can enumerate checkpoint records, but checkpoints are not the same as a polished chat list. A single conversation has many checkpoints, so such a listing needs deduplication and appropriate filtering. A separate chat registry makes titles, ordering, ownership, and empty chats easier to manage.

## 8. Step 5: clicking a chat selects its thread ID

Each sidebar item must carry the ID of the conversation it represents:

```python
for thread_id in reversed(st.session_state["chat_threads"]):
    if st.sidebar.button(
        f"Chat {thread_id[:8]}",
        key=f"select_{thread_id}",
    ):
        st.session_state["current_thread_id"] = thread_id
```

The unique widget key identifies the sidebar button. The full thread ID selects the actual conversation. A shortened ID can be shown as a label, but pass the full value to LangGraph.

Selecting the ID is only the first half of switching chats. The frontend must also replace the displayed messages with messages from that thread.

## 9. Step 6: restore messages with `get_state()`

The exact Python method is **`get_state`**, with an underscore. It receives a configuration dictionary, rather than a bare thread ID:

```python
config = {"configurable": {"thread_id": selected_thread_id}}
snapshot = chatbot.get_state(config)
messages = snapshot.values.get("messages", [])
```

If your compiled graph is named `workflow`, the same call is:

```python
snapshot = workflow.get_state(
    {"configurable": {"thread_id": selected_thread_id}}
)
messages = snapshot.values.get("messages", [])
```

`chatbot` and `workflow` are variable names, not different APIs. `get_state()` reads the selected thread's state; it does not call the model to generate another answer. [Inspecting thread state](https://docs.langchain.com/oss/python/langgraph/add-memory).

The returned snapshot contains state values. If your application keeps conversation messages under `messages`, that is where you read them. With trimming or summarization, the latest state may contain only retained messages, so a complete transcript may require separate storage.

For a simple text chatbot, convert the stored messages into display records:

```python
from langchain_core.messages import AIMessage, HumanMessage


def load_messages(thread_id):
    config = {"configurable": {"thread_id": thread_id}}
    snapshot = chatbot.get_state(config)
    history = []

    for message in snapshot.values.get("messages", []):
        if isinstance(message, HumanMessage):
            history.append({"role": "user", "content": message.content})
        elif isinstance(message, AIMessage) and isinstance(message.content, str):
            if message.content:
                history.append({"role": "assistant", "content": message.content})

    return history
```

This deliberately simple renderer handles text chat. Tool messages, structured content, and media need their own display rules instead of treating every stored message as an ordinary assistant reply.

## 10. Step 7: continue the selected conversation

After clicking a chat, perform both operations together:

```python
st.session_state["current_thread_id"] = selected_thread_id
st.session_state["message_history"] = load_messages(selected_thread_id)
```

Now the screen and the backend target refer to the same conversation.

Example:

```text
Select Python chat -> current_thread_id = A -> load A's messages
Send "Show an example" -> invoke using A -> continue Python discussion

Select Travel chat -> current_thread_id = B -> load B's messages
Send "What about hotels?" -> invoke using B -> continue travel discussion
```

Loading A's messages while sending the next request to B is a common bug. The UI looks correct, but the model receives the wrong context. Always derive both actions from the selected thread ID.

## 11. Putting the frontend flow together

The following is an illustrative Streamlit frontend for a **simple text-only chatbot**. It expects a backend module that exports a compiled `chatbot` with checkpointing and a `messages` state key. Adapt the backend import to your project layout.

It keeps the chat registry in Streamlit session state, so it demonstrates switching within that session. It is not a durable chat-list implementation across browser sessions or server restarts.

```python
from uuid import uuid4

import streamlit as st
from langchain_core.messages import AIMessage, HumanMessage
from langraph_backend import chatbot


def thread_config(thread_id):
    return {"configurable": {"thread_id": thread_id}}


def load_messages(thread_id):
    snapshot = chatbot.get_state(thread_config(thread_id))
    history = []
    for message in snapshot.values.get("messages", []):
        if isinstance(message, HumanMessage):
            history.append({"role": "user", "content": message.content})
        elif isinstance(message, AIMessage) and isinstance(message.content, str):
            if message.content:
                history.append({"role": "assistant", "content": message.content})
    return history


def new_chat():
    thread_id = str(uuid4())
    st.session_state["chat_threads"].append(thread_id)
    st.session_state["current_thread_id"] = thread_id
    st.session_state["message_history"] = []


def select_chat(thread_id):
    st.session_state["current_thread_id"] = thread_id
    st.session_state["message_history"] = load_messages(thread_id)


# Initialize once per frontend session, not once per rerun.
if "chat_threads" not in st.session_state:
    st.session_state["chat_threads"] = []

if "current_thread_id" not in st.session_state:
    new_chat()

if "message_history" not in st.session_state:
    st.session_state["message_history"] = load_messages(
        st.session_state["current_thread_id"]
    )

st.sidebar.button("New Chat", on_click=new_chat)
st.sidebar.subheader("Conversations")

for thread_id in reversed(st.session_state["chat_threads"]):
    st.sidebar.button(
        f"Chat {thread_id[:8]}",
        key=f"chat_{thread_id}",
        on_click=select_chat,
        args=(thread_id,),
    )

st.caption(f"Current thread: {st.session_state['current_thread_id']}")

for message in st.session_state["message_history"]:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

user_input = st.chat_input("Type a message")

if user_input:
    # Capture the selected ID for this turn.
    thread_id = st.session_state["current_thread_id"]
    try:
        chatbot.invoke(
            {"messages": [HumanMessage(content=user_input)]},
            config=thread_config(thread_id),
        )
    except Exception:
        st.error("This turn did not finish. Please check the backend.")
    else:
        # Reload the saved state so the UI matches the conversation.
        st.session_state["message_history"] = load_messages(thread_id)
        st.rerun()
```

Callbacks run before Streamlit redraws the main interface. This allows a sidebar click to update the selected ID and message history before the chat area is rendered.

For a streaming variation, consume `chatbot.stream(...)` with `thread_config(thread_id)`, render the desired events, then reload saved messages after the stream completes. A partial or failed turn needs explicit handling; do not assume every interrupted generation produced a complete saved assistant response.

## 12. Making the feature work after reopening the app

There are three different lifetimes to consider:

| Storage | What it preserves | Limitation |
| --- | --- | --- |
| Frontend session state | Current selection, sidebar IDs, displayed messages | Not a durable database across sessions. |
| In-memory checkpointer | Thread state in the live backend | Lost on process restart. |
| Durable chat registry and checkpointer | Discoverable conversations and their saved state | Must be configured and kept consistent. |

For a durable implementation, save the chat record when New Chat is clicked, query those records on startup, and use a persistent checkpointer for graph state. Store the last selected ID if you want the app to reopen the same chat automatically. Ensure that ID still belongs to an available chat before restoring it.

In a multi-user application, load only the signed-in user's chats and verify ownership when reading or writing a thread. A UUID identifies a conversation; it does not grant permission to access it.

## 13. Common mistakes

| Mistake | Result | Correct approach |
| --- | --- | --- |
| Use the same hard-coded ID for every chat | Conversations share state. | Generate one ID per new conversation. |
| Generate an ID for every message or rerun | Previous context seems lost. | Retain the selected ID across turns. |
| Change the sidebar selection but keep old config | The next message goes to another chat. | Build config from the current selection. |
| Call `workflow.getstate(thread_id)` | Incorrect API usage. | Use `workflow.get_state(config)`. |
| Load messages without changing the active ID | Display and backend context disagree. | Update selection and history together. |
| Keep only a sidebar list without checkpoints | Chats have IDs but no restorable state. | Enable checkpointing. |
| Keep checkpoints without a saved chat list | State exists but chats are hard to discover. | Maintain/query a chat registry. |
| Resend the entire history as new messages | Messages may be duplicated. | Send the new turn and let saved state provide history. |

## 14. The complete flow to remember

```text
New Chat clicked
    -> generate UUID
    -> register the chat
    -> select its thread ID
    -> show an empty conversation

Message submitted
    -> read the current thread ID
    -> put it in config["configurable"]["thread_id"]
    -> call invoke() or stream()
    -> graph uses and updates that thread's saved state

Frontend opened
    -> load available chat records
    -> display them in the sidebar

Existing chat clicked
    -> select its saved thread ID
    -> call get_state(config)
    -> show snapshot.values["messages"]
    -> send future turns using that same thread ID
```

**Remember:** the thread ID chooses the conversation, the checkpointer stores its state, and the frontend loads and displays the selected conversation.

## References

- [LangGraph persistence](https://docs.langchain.com/oss/python/langgraph/persistence)
- [LangGraph memory and inspecting state](https://docs.langchain.com/oss/python/langgraph/add-memory)
- [LangGraph streaming](https://docs.langchain.com/oss/python/langgraph/streaming)

The snippets are learning examples; they do not modify the existing chatbot application. The combined frontend assumes a configured backend and has not been run as an application here.
