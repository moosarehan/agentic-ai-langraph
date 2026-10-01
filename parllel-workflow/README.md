# Parallel Workflows in LangGraph

## What Are Parallel Workflows?

In a sequential workflow, nodes execute one after another in a fixed, linear order. In a **parallel workflow**, multiple nodes execute **at the same time within a single step** — called a **super-step** in LangGraph.

```
                ┌─── Node A ───┐
START ──────────┼─── Node B ───┼──── Next Node ──── END
                └─── Node C ───┘
```

Parallelism is achieved purely through how you wire the **edges**. When multiple edges fan out from the same source (like `START`), LangGraph schedules all of those target nodes to run concurrently in the same super-step. When all parallel nodes finish, the next node downstream (that all of them feed into) executes.

This is powerful because:
- **Independent computations run simultaneously** — no node is waiting for another to finish if they don't depend on each other.
- **The graph structure itself defines the parallelism** — you don't need threads, async code, or any concurrency primitives. Just add edges.

---

## Code 1: Batsman Stats Workflow (No LLM)

📄 **File:** `4_batsman_workflow.ipynb`

This is a clean, non-LLM example that shows the mechanics of parallel execution in LangGraph. Given a batsman's raw cricket stats (runs, balls, fours, sixes), the workflow computes three derived statistics **in parallel**, then combines them into a summary.

### The State

```python
class BatsmanState(TypedDict):

    runs: int
    balls: int
    fours: int
    sixes: int

    sr: float
    bpb: float
    boundary_percent: float
    summary: str
```

The state has two layers:
- **Inputs** — `runs`, `balls`, `fours`, `sixes` (provided by the user)
- **Computed outputs** — `sr` (strike rate), `bpb` (balls per boundary), `boundary_percent`, and `summary`

The three computed stats are **independent of each other** — each one only needs the raw inputs. This is exactly the kind of situation where parallel execution makes sense.

### The Parallel Nodes

```python
def calculate_sr(state: BatsmanState):

    sr = (state['runs']/state['balls'])*100
    
    return {'sr': sr}
```

```python
def calculate_bpb(state: BatsmanState):

    bpb = state['balls']/(state['fours'] + state['sixes'])

    return {'bpb': bpb}
```

```python
def calculate_boundary_percent(state: BatsmanState):

    boundary_percent = (((state['fours'] * 4) + (state['sixes'] * 6))/state['runs'])*100

    return {'boundary_percent': boundary_percent}
```

Each node reads from the raw inputs in the state, computes one stat, and returns **only the key it changed** — a partial state update.

### The Summary Node

```python
def summary(state: BatsmanState):

    summary = f"""
Strike Rate - {state['sr']} \n
Balls per boundary - {state['bpb']} \n
Boundary percent - {state['boundary_percent']}
"""
    
    return {'summary': summary}
```

This node runs **after** all three parallel nodes finish. By that point, `sr`, `bpb`, and `boundary_percent` are all populated in the state, so it can read all three and produce the summary.

### The Graph

```python
graph = StateGraph(BatsmanState)

graph.add_node('calculate_sr', calculate_sr)
graph.add_node('calculate_bpb', calculate_bpb)
graph.add_node('calculate_boundary_percent', calculate_boundary_percent)
graph.add_node('summary', summary)

# edges

graph.add_edge(START, 'calculate_sr')
graph.add_edge(START, 'calculate_bpb')
graph.add_edge(START, 'calculate_boundary_percent')

graph.add_edge('calculate_sr', 'summary')
graph.add_edge('calculate_bpb', 'summary')
graph.add_edge('calculate_boundary_percent', 'summary')

graph.add_edge('summary', END)

workflow = graph.compile()
```

The parallelism is right there in the edges:
- Three edges fan out from `START` → all three stat nodes run in the **same super-step**.
- Three edges converge into `summary` → the summary node waits until all three are done.

```
                ┌─── calculate_sr ──────────────┐
START ──────────┼─── calculate_bpb ─────────────┼──── summary ──── END
                └─── calculate_boundary_percent ┘
```

### Execution

```python
initial_state = {
    'runs': 100,
    'balls': 50,
    'fours': 6,
    'sixes': 4
}

workflow.invoke(initial_state)
```

**Output:**
```python
{'runs': 100,
 'balls': 50,
 'fours': 6,
 'sixes': 4,
 'sr': 200.0,
 'bpb': 5.0,
 'boundary_percent': 48.0,
 'summary': '\nStrike Rate - 200.0 \n\nBalls per boundary - 5.0 \n\nBoundary percent - 48.0\n'}
```

All three stats were computed in parallel, and the summary was generated only after all three completed.

---

## Partial State Updates and the Conflict Problem

Notice something important in the code above. Each node does **not** return the full state. Instead, each node returns only a dictionary with the key(s) it changed:

```python
# Only returns {'sr': sr} — NOT the entire state
def calculate_sr(state: BatsmanState):
    sr = (state['runs']/state['balls'])*100
    return {'sr': sr}
```

In LangGraph, **every node returns a partial update to the state** — just a dictionary containing the key(s) it actually changed — rather than the entire state object. This is the default behavior for all nodes, whether they run sequentially or in parallel.

The need for this becomes critical in parallel workflows, where multiple nodes can be active and run within the **same super-step**. If two of these parallel nodes happen to write to the **same key** in the state, LangGraph has no way of knowing whose update should "win," since both writes are arriving at the same time with equal priority — **this is the conflict**.

Simply returning a partial update doesn't resolve this on its own, since even a one-key partial update still collides if another node in the same super-step writes to that same key. **But** — if the updated keys are **different** across the parallel nodes, there's no conflict at all, since each node is only claiming ownership of its own key.

In the batsman example, there is **no conflict** because:
- `calculate_sr` writes to `sr`
- `calculate_bpb` writes to `bpb`
- `calculate_boundary_percent` writes to `boundary_percent`

Three different keys. No overlap. No problem.

**But what happens when parallel nodes DO need to write to the same key?** That's where **reducers** come in — and that's exactly what the next example demonstrates.

---

## Code 2: UPSC Essay Evaluation Workflow (LLM-Based)

📄 **File:** `5_UPSC_essay_workflow.ipynb`

This is a real-world LLM-powered workflow. Given a UPSC essay, three evaluator nodes run **in parallel** — each assessing the essay on a different criterion (language quality, depth of analysis, clarity of thought). Each evaluator produces a feedback string and a score. Then a final node combines all three feedbacks into an overall summary and computes the average score.

### Setup

```python
from langgraph.graph import StateGraph, START, END
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from typing import TypedDict, Annotated
from pydantic import BaseModel, Field
import operator

load_dotenv()

model = ChatOpenAI(model='gpt-4o-mini')
```

### Structured Output Schema

```python
class EvaluationSchema(BaseModel):

    feedback: str = Field(description='Detailed feedback for the essay')
    score: int = Field(description='Score out of 10', ge=0, le=10)
```

This Pydantic model ensures every LLM evaluator returns a structured object with exactly two fields: a `feedback` string and a `score` integer (0–10). The `model.with_structured_output(EvaluationSchema)` call binds this schema to the model so every invocation is guaranteed to return this structure.

```python
structured_model = model.with_structured_output(EvaluationSchema)
```

### The State — And Why `Annotated` + `operator.add` Matters

```python
class UPSCState(TypedDict):

    essay: str
    language_feedback: str
    analysis_feedback: str
    clarity_feedback: str
    overall_feedback: str
    individual_scores: Annotated[list[int], operator.add]
    avg_score: float
```

Most of this state looks familiar — each evaluator writes to its own feedback key (`language_feedback`, `analysis_feedback`, `clarity_feedback`), and the final node writes `overall_feedback` and `avg_score`. No conflicts there.

But look at `individual_scores`:

```python
individual_scores: Annotated[list[int], operator.add]
```

This is the key line. The `Annotated[list[int], operator.add]` syntax attaches a **reducer** to this key. Here's why that's necessary:

All three evaluator nodes need to write a score. But they all run in **the same super-step** (in parallel). If they all tried to write to a plain `individual_scores: list[int]` key, LangGraph would have **three simultaneous writes to the same key** — and no way to decide which one wins. This is the conflict we discussed earlier.

**Without a reducer, LangGraph throws an `InvalidUpdateError`.**

The `operator.add` reducer tells LangGraph: **"When multiple writes arrive for this key at the same time, don't overwrite — combine them by adding (concatenating) the lists together."**

So when the three evaluators return:
- `{'individual_scores': [7]}` (language)
- `{'individual_scores': [8]}` (analysis)
- `{'individual_scores': [7]}` (clarity)

Instead of one overwriting the others, the reducer **merges** them:

```python
individual_scores = [7] + [8] + [7]  # → [7, 8, 7]
```

This is why each evaluator wraps its score in a **list** (`[output.score]` not `output.score`) — because `operator.add` on lists means concatenation. The reducer concatenates all the lists into one, preserving every evaluator's score.

### The Three Parallel Evaluator Nodes

```python
def evaluate_language(state: UPSCState):

    prompt = f'Evaluate the language quality of the following essay and provide a feedback and assign a score out of 10 \n {state["essay"]}'
    output = structured_model.invoke(prompt)

    return {'language_feedback': output.feedback, 'individual_scores': [output.score]}
```

```python
def evaluate_analysis(state: UPSCState):

    prompt = f'Evaluate the depth of analysis of the following essay and provide a feedback and assign a score out of 10 \n {state["essay"]}'
    output = structured_model.invoke(prompt)

    return {'analysis_feedback': output.feedback, 'individual_scores': [output.score]}
```

```python
def evaluate_thought(state: UPSCState):

    prompt = f'Evaluate the clarity of thought of the following essay and provide a feedback and assign a score out of 10 \n {state["essay"]}'
    output = structured_model.invoke(prompt)

    return {'clarity_feedback': output.feedback, 'individual_scores': [output.score]}
```

Each evaluator:
1. **Reads** the essay from state
2. **Calls the LLM** with a criterion-specific prompt
3. **Returns a partial update** with its own feedback key AND a score wrapped in a list

The feedback keys (`language_feedback`, `analysis_feedback`, `clarity_feedback`) are all different — no conflict. The `individual_scores` key is shared — but the reducer handles that.

### The Final Evaluation Node

```python
def final_evaluation(state: UPSCState):

    # summary feedback
    prompt = f'Based on the following feedbacks create a summarized feedback \n language feedback - {state["language_feedback"]} \n depth of analysis feedback - {state["analysis_feedback"]} \n clarity of thought feedback - {state["clarity_feedback"]}'
    overall_feedback = model.invoke(prompt).content

    # avg calculate
    avg_score = sum(state['individual_scores'])/len(state['individual_scores'])

    return {'overall_feedback': overall_feedback, 'avg_score': avg_score}
```

By the time this node runs, the state has:
- All three feedback strings (from three different keys — no conflict)
- `individual_scores` as `[7, 8, 7]` (three scores merged by the reducer)

It calls the LLM one more time to produce a combined summary from all three feedbacks, then computes the average score with simple arithmetic.

### The Graph

```python
graph = StateGraph(UPSCState)

graph.add_node('evaluate_language', evaluate_language)
graph.add_node('evaluate_analysis', evaluate_analysis)
graph.add_node('evaluate_thought', evaluate_thought)
graph.add_node('final_evaluation', final_evaluation)

# edges
graph.add_edge(START, 'evaluate_language')
graph.add_edge(START, 'evaluate_analysis')
graph.add_edge(START, 'evaluate_thought')

graph.add_edge('evaluate_language', 'final_evaluation')
graph.add_edge('evaluate_analysis', 'final_evaluation')
graph.add_edge('evaluate_thought', 'final_evaluation')

graph.add_edge('final_evaluation', END)

workflow = graph.compile()
```

Same fan-out / fan-in pattern as the batsman example:

```
                ┌─── evaluate_language ───┐
START ──────────┼─── evaluate_analysis ───┼──── final_evaluation ──── END
                └─── evaluate_thought ────┘
```

### Execution

```python
initial_state = {
    'essay': essay2
}

workflow.invoke(initial_state)
```

The workflow:
1. All three evaluators run **in parallel** in the same super-step
2. Each produces feedback + a score
3. Feedbacks go to separate keys (no conflict)
4. Scores go to the **same key** (`individual_scores`) — the `operator.add` reducer merges them into a single list
5. `final_evaluation` reads all feedbacks and scores, produces the overall summary and average

**Output (for a poorly written essay):**
```python
{
    'language_feedback': "The essay attempts to address a significant topic... however, the language quality is quite poor...",
    'analysis_feedback': "The essay presents a basic overview... However, the depth of analysis is shallow...",
    'clarity_feedback': "The essay presents a clear and simple overview... However, the clarity of thought is hindered...",
    'overall_feedback': "**Summarized Feedback:**\n\nThe essay addresses an important topic... but it suffers from significant language and structural issues...",
    'individual_scores': [4, 3, 4],
    'avg_score': 3.6666666666666665
}
```

---

## Reducers: The Full Picture

Let's bring together everything about reducers so the concept is crystal clear.

### The Problem

In a parallel workflow, multiple nodes run in the **same super-step**. Each node returns a partial state update. If two or more nodes try to update the **same key**, there's a conflict — LangGraph doesn't know which value should win.

### What Doesn't Work

**No reducer + same key + parallel nodes = `InvalidUpdateError`**

If `individual_scores` was just `list[int]` (no `Annotated`, no reducer), and three nodes all returned `{'individual_scores': [score]}` in the same super-step, LangGraph would crash. It refuses to silently drop data.

### What the Reducer Does

A reducer is a function attached to a state key via `Annotated[type, reducer_function]`. It tells LangGraph **how to combine multiple writes** to the same key that arrive in the same super-step.

```python
individual_scores: Annotated[list[int], operator.add]
```

`operator.add` for lists means **concatenation**. So:

```
[7] + [8] + [7]  →  [7, 8, 7]
```

### Why Each Score is Wrapped in a List

Each evaluator returns `[output.score]` — a single-element list — not `output.score` directly. This is because the reducer (`operator.add`) operates on lists. You can't "add" integers to a list; you concatenate lists with lists. So each node wraps its score in `[...]` to make it a list, and the reducer concatenates them all.

### When You Don't Need a Reducer

If parallel nodes write to **different keys**, there's no conflict, and no reducer is needed. In both examples:
- The batsman workflow: `sr`, `bpb`, `boundary_percent` — all different keys. No reducer needed.
- The UPSC workflow: `language_feedback`, `analysis_feedback`, `clarity_feedback` — all different keys. No reducer needed for these. Only `individual_scores` needed a reducer because all three evaluators write to it.

### The Key Takeaway

> Reducers exist because **parallel execution creates write conflicts**. When nodes run sequentially, each node sees the previous node's full output — there's no ambiguity about what the "current" value is. But when nodes run in parallel, multiple "current" values arrive at the same time. The reducer is the rule that resolves this ambiguity — it defines how to **merge** concurrent writes instead of having one overwrite the others.

---

## Key Takeaways

1. **Parallel workflows use fan-out / fan-in edges** — multiple edges from one source to many targets create parallelism; multiple edges from many sources to one target create synchronization.

2. **Parallelism is structural, not code-based** — you don't write async code or manage threads. The graph topology defines what runs in parallel.

3. **Every node returns a partial state update** — just the keys it changed, not the full state. This is true for both sequential and parallel workflows.

4. **Different keys = no conflict** — if parallel nodes all write to different state keys, everything just works.

5. **Same key + no reducer = error** — if parallel nodes write to the same key without a reducer, LangGraph throws an `InvalidUpdateError`.

6. **Reducers resolve write conflicts** — `Annotated[type, reducer_function]` tells LangGraph how to merge concurrent writes. `operator.add` concatenates lists; other reducers could sum numbers, merge dicts, etc.

7. **Wrap values in the reducer's expected type** — if the reducer is `operator.add` on lists, each node must return a list (e.g., `[score]` not `score`).
