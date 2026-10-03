# Iterative Workflows in LangGraph

## What Are Iterative Workflows?

In traditional pipelines (sequential or parallel), data moves strictly in one direction — forward from start to finish. Once a step executes, it never runs again. These are known as Directed Acyclic Graphs (**DAGs**).

An **iterative workflow** breaks this linear constraint by introducing **cycles (loops)**. In an iterative graph, the output of a downstream step is sent back as input to an earlier step, allowing the workflow to repeatedly refine, polish, or self-correct until a specific quality standard or exit condition is satisfied:

```
          ┌─────────────────────────────────────────────────┐
          │                                                 │
          ▼                                                 │ (Needs Improvement + Feedback)
START ──► Generate ──► Evaluate ──► [Quality Gate Check] ───┘
                           │
                           ▼ (Approved or Max Iterations Reached)
                          END
```

### Why Iterative Workflows Matter in LLM Applications
Large Language Models rarely generate flawless results on their first attempt, especially for tasks requiring creativity, humor, strict constraints, or multi-step logic. Asking an LLM to "write and self-critique" in a single prompt often leads to blind spots because the model evaluates its own reasoning with the same bias that produced the draft.

Iterative workflows solve this by:
- **Separating generation from evaluation**: Giving different personas to different nodes (e.g., a creative writer vs. a ruthless critic).
- **Targeted feedback loops**: Rather than regenerating from scratch, subsequent passes fix only the flaws identified by the evaluator.
- **Measurable convergence**: Every loop brings the output closer to defined acceptance criteria.

---

## The Evaluator–Optimizer Pattern

One of the most effective iterative architectural patterns in LangGraph is the **Evaluator–Optimizer** workflow. It divides responsibility across three distinct roles:

1. **Generator**: Creates the initial candidate response based solely on the user's objective or topic.
2. **Evaluator**: Critically inspects the draft against explicit evaluation criteria (rubric, constraints, tone, format) and produces a verdict (`approved` vs. `needs_improvement`) along with concrete actionable feedback.
3. **Optimizer**: Ingests the previous draft along with the critic's feedback, directly addresses the criticisms to punch up the output, increments the retry counter, and passes the revised draft back to the Evaluator.

This cycle continues until either:
- The Evaluator approves the output, OR
- The safety threshold (`max_iteration`) is reached.

---

## Code Walkthrough: X (Twitter) Post Generator

📄 **File:** `8_X_post_generator.ipynb`

This workflow crafts viral, witty, and punchy posts for X (Twitter). Writing good tweets is notoriously difficult for a single prompt because it requires high originality, brevity (under 280 characters), meme logic, and scroll-stopping humor without falling into predictable setup-punchline clichés.

### 1. Structured Evaluator Schema

To eliminate parsing errors and guarantee reliable routing, the evaluator uses Pydantic with `with_structured_output()`:

```python
from pydantic import BaseModel, Field
from typing import Literal

class TweetEvaluation(BaseModel):
    evaluation: Literal["approved", "needs_improvement"] = Field(
        ..., description="Final evaluation result."
    )
    feedback: str = Field(
        ..., description="Feedback for the tweet."
    )

structured_evaluator_llm = evaluator_llm.with_structured_output(TweetEvaluation)
```

This guarantees the model returns an explicit programmatic verdict (`approved` or `needs_improvement`) along with written feedback.

---

### 2. State Design (`TweetState`)

The state manages both the current working variables and the historical audit trail across loops:

```python
from typing import TypedDict, Annotated
import operator

class TweetState(TypedDict):
    topic: str
    tweet: str
    evaluation: Literal["approved", "needs_improvement"]
    feedback: str
    iteration: int
    max_iteration: int

    # History preserved across cycles using reducers
    tweet_history: Annotated[list[str], operator.add]
    feedback_history: Annotated[list[str], operator.add]
```

- **`iteration` and `max_iteration`**: Track loop execution to guarantee a safe halting condition.
- **`tweet_history` and `feedback_history`**: Decorated with `Annotated[list[str], operator.add]`. Every iteration appends its revision and critique to these lists instead of overwriting them, providing a complete record of how the draft evolved.

---

### 3. The Workflow Nodes

#### Node 1: Initial Generator (`generate_tweet`)
Adopts a witty, clever influencer persona to write the first draft based on the topic:

```python
def generate_tweet(state: TweetState):
    messages = [
        SystemMessage(content="You are a funny and clever Twitter/X influencer."),
        HumanMessage(content=f"""
Write a short, original, and hilarious tweet on the topic: "{state['topic']}".

Rules:
- Do NOT use question-answer format.
- Max 280 characters.
- Use observational humor, irony, sarcasm, or cultural references.
- Think in meme logic, punchlines, or relatable takes.
- Use simple, day to day english
""")
    ]
    response = generator_llm.invoke(messages).content
    return {'tweet': response, 'tweet_history': [response]}
```

#### Node 2: Ruthless Critic (`evaluate_tweet`)
Acts as a strict evaluator checking originality, humor, punchiness, virality, and format restrictions:

```python
def evaluate_tweet(state: TweetState):
    messages = [
        SystemMessage(content="You are a ruthless, no-laugh-given Twitter critic. You evaluate tweets based on humor, originality, virality, and tweet format."),
        HumanMessage(content=f"""
Evaluate the following tweet:
Tweet: "{state['tweet']}"

Use the criteria below:
1. Originality – Is this fresh, or have you seen it a hundred times before?
2. Humor – Did it genuinely make you smile, laugh, or chuckle?
3. Punchiness – Is it short, sharp, and scroll-stopping?
4. Virality Potential – Would people retweet or share it?
5. Format – Under 280 characters, no Q&A joke format, no cliché setup-punchline.

### Respond ONLY in structured format:
- evaluation: "approved" or "needs_improvement"
- feedback: One paragraph explaining the strengths and weaknesses
""")
    ]
    response = structured_evaluator_llm.invoke(messages)
    return {
        'evaluation': response.evaluation,
        'feedback': response.feedback,
        'feedback_history': [response.feedback]
    }
```

#### Node 3: Optimizer (`optimize_tweet`)
Reads the previous tweet and the critic's exact feedback, punches up the lines, and increments the iteration count:

```python
def optimize_tweet(state: TweetState):
    messages = [
        SystemMessage(content="You punch up tweets for virality and humor based on given feedback."),
        HumanMessage(content=f"""
Improve the tweet based on this feedback:
"{state['feedback']}"

Topic: "{state['topic']}"
Original Tweet:
{state['tweet']}

Re-write it as a short, viral-worthy tweet. Avoid Q&A style and stay under 280 characters.
""")
    ]
    response = optimizer_llm.invoke(messages).content
    iteration = state['iteration'] + 1

    return {
        'tweet': response,
        'iteration': iteration,
        'tweet_history': [response]
    }
```

---

### 4. Router and Graph Construction

The decision to continue looping or exit is determined by `route_evaluation`:

```python
def route_evaluation(state: TweetState):
    if state['evaluation'] == 'approved' or state['iteration'] >= state['max_iteration']:
        return 'approved'
    else:
        return 'needs_improvement'
```

#### Graph Definition:
```python
graph = StateGraph(TweetState)

# 1. Register nodes
graph.add_node('generate', generate_tweet)
graph.add_node('evaluate', evaluate_tweet)
graph.add_node('optimize', optimize_tweet)

# 2. Sequential start
graph.add_edge(START, 'generate')
graph.add_edge('generate', 'evaluate')

# 3. Conditional exit or loop
graph.add_conditional_edges(
    'evaluate',
    route_evaluation,
    {
        'approved': END,
        'needs_improvement': 'optimize'
    }
)

# 4. Closing the cycle (Optimizer back to Evaluator)
graph.add_edge('optimize', 'evaluate')

workflow = graph.compile()
```

```
START ──► generate ──► evaluate ◄──────────┐
                          │                │
                          ▼                │
               [route_evaluation]          │ (Loop Back)
                 /             \           │
     'approved' /               \ 'needs_improvement'
               ▼                 ▼         │
              END             optimize ────┘
```

---

### 5. Execution Example

```python
initial_state = {
    "topic": "Adulting",
    "iteration": 1,
    "max_iteration": 5
}
result = workflow.invoke(initial_state)

for i, (tweet, feedback) in enumerate(zip(result['tweet_history'], result['feedback_history']), 1):
    print(f"--- Iteration {i} ---")
    print(f"Tweet: {tweet}")
    print(f"Feedback: {feedback}\n")
```

If the first tweet is generic, the evaluator flags it, passes actionable critique to `optimize`, and the optimizer revises it until the rubric is met or iteration reaches `max_iteration`.

---

## Why Loops in LangGraph REQUIRE Conditional Edges

In LangGraph, whenever you build a loop — an edge that sends a node's output back to a node that already executed, like the "Rejected + Feedback" cycle in Evaluator–Optimizer — you have to pair it with a conditional edge that checks the state and decides when the loop should stop. Without that check, the graph has no way of knowing when to exit, and it keeps re-activating the same node every round.

A plain, unconditional edge always fires. So if a node looped back to itself through a normal edge, it would get a new message every single super-step with no way out — meaning the halting condition (no active nodes, no messages in transit) would never be reached, since that node would always have something waiting for it in the next round.

A conditional edge solves this because it's a function that looks at the current state right after a node runs and chooses where to route next:

```python
def should_continue(state) -> str:
    if state["attempts"] >= 3 or state["accepted"]:
        return "exit"
    return "generator"   # loop back

graph.add_conditional_edges(
    "evaluator",
    should_continue,
    {"generator": "generator", "exit": END}
)
```

So the rule stands: any loop needs a conditional edge (or similar state-based branching) to act as its exit condition — usually tied to something like a retry counter or an "accepted" flag in the state. Leave that out, and you end up with an infinite loop, since nothing ever interrupts the cycle of messages being sent back to the same node.

---

## Summary of Patterns Learned

| Workflow Type | Graph Topology | Control Flow Primitive | Primary Use Case |
| :--- | :--- | :--- | :--- |
| **Sequential** | Linear ($A \rightarrow B \rightarrow C$) | `add_edge()` | Straightforward pipelines & prompt chains |
| **Parallel** | Fan-out / Fan-in ($A \rightarrow [B, C] \rightarrow D$) | Multiple edges + Reducers (`Annotated`) | Independent tasks, multiple evaluators |
| **Conditional** | Branching ($A \rightarrow B \text{ or } C$) | `add_conditional_edges()` | Triage, routing, intent classification |
| **Iterative** | Cyclic ($A \rightarrow B \rightarrow C \rightarrow B \dots$) | `add_edge()` + `add_conditional_edges()` | Evaluator-Optimizer, self-correction, refinement |
