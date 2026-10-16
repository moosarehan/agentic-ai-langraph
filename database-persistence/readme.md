# Database Persistence in LangGraph

## 1. What is persistence?

**Persistence means keeping application data or state so it can be used again later.** In a chatbot, that state may include conversation messages, retrieved information, intermediate results, and the workflow's progress.

Imagine this conversation:

```text
User: My name is Musa.
Assistant: Hello, Musa!

User: What is my name?
Assistant: Your name is Musa.
```

The second answer requires access to information from the first turn. The application must retain that information and provide it to the model. A model call does not automatically remember earlier calls.

There are two useful questions:

- Can the application retain context between turns while it is running?
- Can it restore that context after its process has stopped and restarted?

In-memory checkpointing addresses the first question within the lifetime of its saver instance. Database checkpointing can address both, provided the underlying database storage remains available.

**Example:** you close a chatbot server in the evening and restart it the next day. With durable checkpoint storage and the same thread ID, the application can reopen a saved conversation instead of starting from an empty state.

In LangGraph, checkpointers provide thread-scoped persistence. Cross-thread knowledge, such as a preference shared across different conversations, is a separate concern commonly handled through a store. [Persistence overview](https://docs.langchain.com/oss/python/langgraph/persistence).

## 2. What are checkpoints?

### 2.1 A checkpoint is a saved point in graph execution

A **checkpoint** records a thread's graph state at a particular point. Think of saving your progress in a game: you can return to a saved position instead of beginning again.

For a chatbot, saved state might contain the messages accumulated so far. For a larger workflow, it might include a search result, draft response, validation outcome, or other state fields.

```text
User input
    -> retrieve documents
    -> generate answer
    -> validate answer
    -> finish

Saved state lets the application inspect its progress at checkpoint boundaries.
```

LangGraph creates full checkpoints at **super-step boundaries**. A super-step is a unit of execution in which scheduled nodes can run, potentially in parallel. A full checkpoint is therefore not necessarily created after every individual node in a parallel branch. [Checkpointer concepts](https://docs.langchain.com/oss/python/langgraph/checkpointers).

### 2.2 Checkpoint, checkpointer, and database are different things

| Term | Meaning | Simple analogy |
| --- | --- | --- |
| State | The application's current working information | The document you are editing. |
| Checkpoint | A saved version of state and execution information | A saved version of that document. |
| Checkpointer | The component that writes and reads checkpoints | The save/load mechanism. |
| Storage backend | Where checkpoint data is held | RAM, a SQLite file, or a PostgreSQL database. |
| Thread ID | Identifies the conversation or execution thread | The label on the conversation's folder. |
| Checkpoint ID | Identifies a particular saved checkpoint | The label on one saved version inside that folder. |

A thread can have many checkpoints. The thread ID does not change for every checkpoint or every message in the same conversation.

### 2.3 What can we read from a saved state?

The public state snapshot exposes state values and execution information such as upcoming work and checkpoint metadata. For a conversation whose state contains a `messages` key:

```python
config = {"configurable": {"thread_id": "chat-123"}}
snapshot = chatbot.get_state(config)
messages = snapshot.values.get("messages", [])
```

To inspect historical snapshots:

```python
for snapshot in chatbot.get_state_history(config):
    print(snapshot.values)
```

These APIs operate through the configured checkpointer. Your frontend does not need to decode checkpoint tables directly. [Reading state and checkpoint history](https://docs.langchain.com/oss/python/langgraph/checkpointers).

### 2.4 Why are checkpoints useful?

For a resume-chat application, checkpoints provide the saved state used to restore a conversation. In other workflows, saved execution state can support inspection, approval steps, recovery, and replay.

A checkpoint does not freeze the entire Python process. Local variables outside graph state, open sockets, and arbitrary external side effects are not automatically restored. If a tool sends an email or charges a card, checkpoint recovery alone does not guarantee that the action can never be repeated. Such operations need an appropriate retry and idempotency design.

## 3. Two checkpoint storage approaches

For these notes, we group checkpoint storage into two practical categories: **in-memory storage** and **database-backed storage**. These are implementation choices for checkpointing, rather than an exhaustive list of every persistence mechanism.

### 3.1 InMemorySaver — one-paragraph explanation

`InMemorySaver` stores checkpoints in the Python process's RAM. It is convenient for learning, experiments, and tests because it requires no database setup: create the saver, compile the graph with it, and pass a thread ID on each invocation. The same saver instance can retain separate conversations while it remains alive, but restarting the process or replacing that saver with a new instance loses its saved checkpoints. Independent app processes also do not automatically share that memory. Use it when temporary state is sufficient; choose durable database storage when chats must survive restarts. [In-memory persistence limitations](https://docs.langchain.com/oss/python/langgraph/persistence).

### 3.2 Database persistence

A database-backed checkpointer serializes and saves checkpoint information in a database. The application can reconnect later and restore saved state using the same thread ID.

```text
Application runs
    -> checkpointer saves thread state in database
    -> application stops
    -> database storage remains
    -> application restarts and reconnects
    -> same thread ID selects the saved state
```

SQLite and PostgreSQL are two available database-backed options. They use different storage architectures, but your graph still follows the familiar pattern:

```python
chatbot = builder.compile(checkpointer=checkpointer)
config = {"configurable": {"thread_id": "chat-123"}}
```

The checkpointer changes where data is stored. The thread ID still chooses which conversation is being used.

## 4. SQLite checkpointer

### 4.1 What is SQLite?

SQLite is an embedded database engine. For durable local checkpointing, it stores data in a file accessed by the application, without requiring a separate database server.

```text
Python application -> SqliteSaver -> local database file
```

A path such as `chat_checkpoints.sqlite` identifies the database file. Reconnecting to that same preserved file lets a later process access the saved checkpoints.

### 4.2 Basic setup

Install the checkpoint integration in your Python environment:

```powershell
python -m pip install langgraph langgraph-checkpoint-sqlite
```

The following integration fragment assumes `builder` is an existing graph builder with a `messages` state field and an appropriate message reducer:

```python
from langgraph.checkpoint.sqlite import SqliteSaver

config = {"configurable": {"thread_id": "sqlite-chat-123"}}

with SqliteSaver.from_conn_string("chat_checkpoints.sqlite") as checkpointer:
    chatbot = builder.compile(checkpointer=checkpointer)

    chatbot.invoke(
        {"messages": [{"role": "user", "content": "My name is Musa."}]},
        config=config,
    )

    snapshot = chatbot.get_state(config)
    print(snapshot.values.get("messages", []))
```

`from_conn_string()` is used as a context manager. Keep graph operations inside the connection's lifetime. A server or frontend resource should keep its connection alive for as long as its compiled graph needs it. [SQLite connection examples](https://reference.langchain.com/python/langgraph.checkpoint.sqlite/SqliteSaver/from_conn_string).

Use a stable file path. If a relative path resolves to a different directory after restart, the application may open a different database and appear to have lost history.

### 4.3 File-backed SQLite versus `:memory:`

```python
# File-backed: saved data can survive process restarts.
SqliteSaver.from_conn_string("chat_checkpoints.sqlite")

# In-memory SQLite: temporary data, not restart-safe file storage.
SqliteSaver.from_conn_string(":memory:")
```

These lines illustrate connection targets; use the context-manager pattern above for actual operations. Choosing SQLite does not guarantee durability if the database itself is configured to live only in memory.

### 4.4 Advantages over InMemorySaver

For a local learning project, the key improvement is that saved state can outlive the Python process. You can stop the chatbot, reopen the same file, and continue a known thread without rebuilding its history manually. You also gain a tangible local database that can be backed up using SQLite-aware procedures.

**Illustrative scenario:** you are testing your resume-chat UI. With an in-memory saver, a backend restart clears the old thread state. With a persistent SQLite file, the backend can recover it after restarting, assuming the sidebar still knows the thread ID.

### 4.5 Limitations and when to use SQLite

SQLite is a strong fit for local projects, prototypes, single-machine tools, and workloads with modest write contention. It supports multiple readers, but only one writer at a time per database file. For heavier simultaneous writes or shared access from multiple app servers, a client/server database is often a better fit. Avoid treating a shared network filesystem as a substitute for a database server. [SQLite's deployment guidance](https://www2.sqlite.org/whentouse.html).

Choose SQLite when you want a simple durable setup on one machine and can preserve the database file. A small production tool may also be suitable if its workload and operational requirements fit; the decision is not determined by whether the project is called a prototype.

An application deployed with an ephemeral filesystem can still lose its SQLite database when the container is replaced. Use persistent storage when those files must survive deployment changes.

## 5. PostgreSQL checkpointer

### 5.1 What is PostgreSQL?

PostgreSQL is a client/server database. The application connects to a running database service rather than directly managing a local SQLite file.

```text
Application instance A --\
                         -> PostgreSQL database
Application instance B --/
```

A shared database allows multiple application instances to access the same checkpoint storage. This is particularly useful when requests for a conversation may reach different workers.

### 5.2 Basic setup

Install the integration and database driver:

```powershell
python -m pip install langgraph langgraph-checkpoint-postgres "psycopg[binary,pool]"
```

You also need an accessible PostgreSQL database and its connection configuration. Installing the Python package does not create a database server.

This fragment assumes `builder` is already defined and `DATABASE_URL` is configured in the environment:

```python
import os
from langgraph.checkpoint.postgres import PostgresSaver

config = {"configurable": {"thread_id": "postgres-chat-123"}}

with PostgresSaver.from_conn_string(os.environ["DATABASE_URL"]) as checkpointer:
    # Initialize the required checkpoint schema on first use.
    checkpointer.setup()

    chatbot = builder.compile(checkpointer=checkpointer)
    chatbot.invoke(
        {"messages": [{"role": "user", "content": "My name is Musa."}]},
        config=config,
    )

    snapshot = chatbot.get_state(config)
    print(snapshot.values.get("messages", []))
```

The checkpoint tables need initialization through `setup()` when using this integration. In a deployed application, manage schema initialization and migration as part of the application's setup process. Keep connection resources alive while graph operations use them. [PostgreSQL checkpointer setup](https://docs.langchain.com/oss/python/langgraph/add-memory).

Async applications should use the corresponding async saver and async graph methods. Do not mix a synchronous example into an async server without considering blocking database work and connection management.

### 5.3 Advantages over InMemorySaver

PostgreSQL separates saved state from the app worker's RAM. Restarting or replacing an application worker does not inherently delete checkpoints in the database. Multiple workers can use shared storage, and the database can be operated with centralized access controls, backups, monitoring, and recovery procedures.

Those operational benefits require configuration. Merely choosing PostgreSQL does not automatically enable backups, replication, or high availability. The database service and its storage must also remain durable.

### 5.4 When to use PostgreSQL

Choose PostgreSQL when your application needs several backend workers or app instances, a shared remote database, substantial simultaneous writes, or established database operations. It is also reasonable for a smaller project if PostgreSQL is already part of its infrastructure.

**Illustrative scenario:** a customer-support chatbot serves users through several backend instances. Each request should find the same saved conversation regardless of which instance receives it. Shared PostgreSQL checkpoints fit that requirement better than separate in-memory savers on each worker.

A shared database does not make concurrent turns in the same conversation automatically safe. Coordinate overlapping requests for the same thread so their state updates follow the intended order.

## 6. SQLite versus PostgreSQL versus InMemorySaver

| Consideration | InMemorySaver | File-backed SQLite | PostgreSQL |
| --- | --- | --- | --- |
| Storage location | Process RAM | Database file | Database service |
| Survives app restart | No | Yes, if the file remains | Yes, if database storage remains |
| Separate server required | No | No | Yes, local or hosted |
| Setup effort | Minimal | Low | Higher |
| Several independent app servers | No automatic shared state | Awkward fit for a shared file | Well suited to shared access |
| Concurrent writes | Limited to the process's design | One writer per file at a time | Better fit for substantial write concurrency |
| Operational responsibility | Keep the process alive for temporary state | Preserve and back up the file correctly | Operate database access, capacity, backups, and recovery |
| Typical fit | Learning and tests | Local apps and modest single-machine workloads | Shared, multi-worker, or larger services |

**Choose based on your workload:** temporary experiment -> in-memory saver; simple durable local chatbot -> SQLite; shared service with multiple workers or heavier writes -> PostgreSQL.

## 7. How database persistence supports resume chat

Database checkpointing is one part of the resume-chat feature. The application also needs a way to discover the chats, usually through a chat registry containing IDs, titles, and ownership.

```text
New Chat
    -> generate a UUID
    -> save a chat record
    -> use that thread ID for graph calls
    -> database checkpointer saves state

Application restarts
    -> load saved chat records into sidebar
    -> user selects a chat
    -> get_state(config) restores its state
    -> next invoke()/stream() uses that same thread ID
```

A database of checkpoints does not automatically create the frontend's sidebar. Conversely, a saved list of chat IDs is insufficient if the corresponding checkpoint state lived only in RAM and has been lost.

When continuing a checkpointed message-based conversation, send the new message and use the proper reducer to merge it with existing state. Resending the full displayed history as new messages can introduce duplicates.

## 8. A simple restart check

To check your chosen persistence setup:

1. Save a turn using a known thread ID.
2. Confirm `get_state(config)` contains the expected state.
3. Stop the application process.
4. Restart it with the same graph definition and preserved database storage.
5. Reconnect and read the same thread ID before sending another message.
6. Confirm that another thread ID does not load that conversation.

This verifies more than clicking a sidebar button while the same process is still running. A generated UUID must be saved somewhere discoverable; generating a fresh UUID after restart will select a different thread.

The code fragments in this document illustrate integration and have not been run here. They require a configured graph; PostgreSQL additionally requires an accessible database.

## References

- [LangGraph persistence](https://docs.langchain.com/oss/python/langgraph/persistence)
- [LangGraph checkpointers and state inspection](https://docs.langchain.com/oss/python/langgraph/checkpointers)
- [LangGraph memory and PostgreSQL setup](https://docs.langchain.com/oss/python/langgraph/add-memory)
- [SqliteSaver connection examples](https://reference.langchain.com/python/langgraph.checkpoint.sqlite/SqliteSaver/from_conn_string)
- [Appropriate uses for SQLite](https://www2.sqlite.org/whentouse.html)
