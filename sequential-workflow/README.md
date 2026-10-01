# Sequential Workflows in LangGraph

## What Are Sequential Workflows?

A sequential workflow is the simplest type of workflow — tasks execute **one after another in a fixed, linear order**. There is no branching, no looping, no conditional logic. Step 1 runs, then Step 2, then Step 3, and so on, in a straight line from start to finish.

```
START → Step 1 → Step 2 → Step 3 → END
```

In LangGraph, this means the graph has **no conditional edges** — every node connects to exactly one next node via a normal `add_edge()`. The execution path is fully determined at build time.

---

## Why LangGraph Is NOT Ideal for Sequential Workflows

Here's the honest truth: **LangGraph is overkill for sequential workflows.**

LangGraph was designed for **complex, non-linear workflows** — workflows with loops, conditional branching, human-in-the-loop steps, and persistent state across long-running processes. That's where it shines. A simple linear pipeline like "take input → call LLM → return output" does **not** need a graph.

**LangChain already handles sequential workflows perfectly well.** A simple LangChain chain or even a plain Python function can do this with far less boilerplate. When you use LangGraph for a purely sequential task, you're essentially:

1. Defining a `TypedDict` state class (unnecessary for a simple pipeline)
2. Creating a `StateGraph` object (unnecessary overhead)
3. Adding nodes one by one (when you could just call functions in order)
4. Adding edges between every consecutive node (just to say "go to the next step")
5. Compiling the graph (extra step that adds nothing for linear flows)

That's a lot of ceremony for something that could be a 5-line Python script.

### So Why Are We Doing It?

**To learn the LangGraph execution model.** Before building complex workflows with conditional branching, loops, and parallel execution, you need to understand the fundamentals — how state flows through the graph, how nodes read from and write to state, and how the graph compiles and executes. Sequential workflows strip away all the complexity so you can focus purely on the execution model itself.

Think of it like learning to drive in an empty parking lot before hitting the highway.

---

## The LangGraph Execution Model

Every LangGraph workflow — whether simple or complex — follows the same pattern:

### Step 1: Define the State

The state is a `TypedDict` that acts as the **shared data container** for the entire workflow. Every node in the graph reads from and writes to this state. It's how data flows from one step to the next.

```python
class MyState(TypedDict):
    input_field: str
    output_field: str
```

### Step 2: Define Node Functions

Each node is a regular Python function that takes the current state as input, does some work (call an LLM, process data, etc.), updates the state, and returns it.

```python
def my_node(state: MyState) -> MyState:
    # Read from state
    data = state['input_field']
    # Do work
    result = process(data)
    # Write to state
    state['output_field'] = result
    return state
```

### Step 3: Build the Graph

Create a `StateGraph`, add your nodes, connect them with edges, and compile.

```python
graph = StateGraph(MyState)
graph.add_node('my_node', my_node)
graph.add_edge(START, 'my_node')
graph.add_edge('my_node', END)
workflow = graph.compile()
```

### Step 4: Execute

Pass an initial state and invoke the workflow. LangGraph walks the graph from `START` to `END`, executing each node in order and passing the state along.

```python
result = workflow.invoke({'input_field': 'some data'})
```

This same pattern scales from the simplest one-node graph to complex multi-agent systems with dozens of nodes and conditional edges.

---

## Code 1: Simple LLM Workflow

📄 **File:** `2_simple_llm_workflow.ipynb`

This is the most minimal LangGraph workflow possible — a single node that takes a question, sends it to an LLM, and returns the answer.

### The State

```python
class LLMState(TypedDict):
    question: str
    answer: str
```

Two fields: `question` (what we put in) and `answer` (what we get out). That's it.

### The Node Function

```python
def llm_qa(state: LLMState) -> LLMState:
    # extract the question from state
    question = state['question']

    # form a prompt
    prompt = f'Answer the following question {question}'

    # ask that question to the LLM
    answer = model.invoke(prompt).content

    # update the answer in the state
    state['answer'] = answer

    return state
```

This function follows the exact execution model described above:
1. **Read** the `question` from state
2. **Process** it by calling the LLM
3. **Write** the `answer` back to state
4. **Return** the updated state

### The Graph

```python
graph = StateGraph(LLMState)

# add nodes
graph.add_node('llm_qa', llm_qa)

# add edges
graph.add_edge(START, 'llm_qa')
graph.add_edge('llm_qa', END)

# compile
workflow = graph.compile()
```

The graph structure is as simple as it gets:

```
START → llm_qa → END
```

### Execution

```python
initial_state = {'question': 'How far is moon from the earth?'}
final_state = workflow.invoke(initial_state)
print(final_state['answer'])
```

**Output:**
```
The average distance between the Moon and Earth is about 384,400 kilometers (238,855 miles).
```

### Could This Be Done Without LangGraph?

Absolutely. This single line does the exact same thing:

```python
model.invoke('How far is moon from the earth?').content
```

The notebook even demonstrates this comparison directly. The point is **not** that LangGraph is better here — it's that you now understand how state, nodes, edges, and execution work together.

---

## Common Workflow Patterns in LangGraph

LangGraph is designed for building all kinds of workflows, and every application will have its own unique workflow structure based on its specific requirements. There is no one-size-fits-all graph — a customer support agent's workflow looks completely different from a research assistant's workflow or a code review pipeline.

However, certain **workflow patterns appear again and again** across different applications. These are common building blocks that almost every LangGraph application uses in some form. One of the most fundamental of these is **Prompt Chaining**.

---

## Code 2: Prompt Chaining Workflow

📄 **File:** `3_prompt_chaining.ipynb`

Prompt chaining is a pattern where the **output of one LLM call becomes the input to the next LLM call**. Instead of asking the LLM to do everything in a single prompt (which often produces lower quality results), you break the task into multiple focused steps, where each step builds on the result of the previous one.

In this example, we build a **blog writer** that works in two steps:
1. **Generate an outline** from a blog title
2. **Write the full blog** using both the title and the generated outline

### The State

```python
class BlogState(TypedDict):
    title: str
    outline: str
    content: str
```

Notice how the state has grown compared to the simple example. We now have three fields because data needs to flow across **two** LLM calls:
- `title` — the input (provided by the user)
- `outline` — intermediate result (generated by the first node, consumed by the second)
- `content` — final output (generated by the second node)

### Node 1: Create the Outline

```python
def create_outline(state: BlogState) -> BlogState:
    # fetch title
    title = state['title']

    # call llm gen outline
    prompt = f'Generate a detailed outline for a blog on the topic - {title}'
    outline = model.invoke(prompt).content

    # update state
    state['outline'] = outline

    return state
```

This node **reads** the `title` from state, generates an outline using the LLM, and **writes** the `outline` back to state. The key insight: it doesn't need to know about the next node or what happens after it — it just does its job and updates the shared state.

### Node 2: Write the Blog

```python
def create_blog(state: BlogState) -> BlogState:
    title = state['title']
    outline = state['outline']

    prompt = f'Write a detailed blog on the title - {title} using the following outline \n {outline}'

    content = model.invoke(prompt).content

    state['content'] = content

    return state
```

This is where the **chaining** happens. This node reads **both** the original `title` and the `outline` that was generated by the previous node. It uses both to produce a much higher quality blog post than if we had just asked the LLM to "write a blog about X" in a single prompt.

### The Graph

```python
graph = StateGraph(BlogState)

# nodes
graph.add_node('create_outline', create_outline)
graph.add_node('create_blog', create_blog)

# edges
graph.add_edge(START, 'create_outline')
graph.add_edge('create_outline', 'create_blog')
graph.add_edge('create_blog', END)

workflow = graph.compile()
```

The graph structure:

```
START → create_outline → create_blog → END
```

Still sequential, but now we have **two nodes** and the state carries data forward between them.

### Execution

```python
initial_state = {'title': 'Rise of AI in India'}
final_state = workflow.invoke(initial_state)

print(final_state['outline'])  # The generated outline
print(final_state['content'])  # The full blog post
```

The LLM first generates a structured outline with sections like Introduction, Historical Context, Current State, Challenges, Future Outlook, and Conclusion. Then the second node uses that outline to write a complete, well-structured blog post that follows the outline's organization.

### Why Prompt Chaining Works Better Than a Single Prompt

| Single Prompt | Prompt Chaining |
|---|---|
| "Write a blog about AI in India" | Step 1: "Generate an outline for a blog about AI in India" |
| LLM tries to do everything at once | Step 2: "Write a blog using this title AND this outline" |
| Often produces unstructured, rambling output | Produces structured, organized output |
| No intermediate results to inspect or modify | You can inspect/modify the outline before writing |

The quality difference comes from **decomposition** — breaking a complex task into simpler, focused sub-tasks. Each LLM call does one thing well, rather than trying to do everything at once.

---

## Key Takeaways

1. **Sequential workflows are the simplest LangGraph pattern** — they follow the same execution model (State → Nodes → Graph → Execute) but with no branching or loops.

2. **LangGraph is overkill for simple sequential tasks** — LangChain or plain Python is better for linear pipelines. We use sequential workflows here to learn the execution model.

3. **The LangGraph execution model is always the same** — Define state, write node functions that read/write state, build the graph, compile, and execute. This pattern applies to every LangGraph workflow.

4. **Prompt chaining is a common pattern** — Breaking a complex LLM task into multiple focused steps produces better results than a single monolithic prompt.

5. **State is the backbone** — The `TypedDict` state is how data flows between nodes. As workflows get more complex, the state grows to carry more information.
