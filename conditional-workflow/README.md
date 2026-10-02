# Conditional Workflows in LangGraph

## What Are Conditional Workflows?

In sequential workflows, execution moves linearly from one step to the next ($A \rightarrow B \rightarrow C$). In parallel workflows, multiple nodes execute simultaneously within a single super-step ($A \rightarrow [B, C] \rightarrow D$).

A **conditional workflow** introduces **dynamic decision-making** into the graph. Instead of following a predetermined path, the execution routes to different nodes depending on the state at runtime:

```
                          ┌─── Path A (e.g., Condition X) ───┐
START ─── Evaluation ────►├─── Path B (e.g., Condition Y) ───┼───► END
                          └─── Path C (e.g., Condition Z) ───┘
```

In LangGraph, conditional routing is achieved using **conditional edges** via `graph.add_conditional_edges()`. A router function inspects the current state and returns a value (such as a node name or a key mapped to a node name) indicating which node should execute next.

---

## Advantages of Conditional Workflows

1. **Dynamic Decision-Making**: Workflows adapt on the fly based on runtime evaluation, intermediate calculations, or LLM classifications.
2. **Cost and Token Efficiency**: Prevents unnecessary LLM calls. For example, simple requests can be resolved immediately without triggering expensive multi-step diagnostic or reasoning chains.
3. **Modularity and Specialization**: Instead of forcing a single monolithic prompt to handle every edge case, individual nodes specialize in specific scenarios (e.g., one node for refunds, another for technical bugs, another for praises).
4. **Asymmetric Execution Paths**: Different conditions can have entirely different path lengths and complexities. One path might terminate immediately, while another triggers a multi-step sub-workflow.
5. **Foundation for Agentic Behavior & Loops**: Conditional edges are the core building block for cycles, self-correction loops, critique mechanisms, and human-in-the-loop validation.

---

## Code 1: Quadratic Equation Workflow (Deterministic Logic)

📄 **File:** `6_quadratic_equation_workflow.ipynb`

This notebook demonstrates the foundational mechanics of conditional branching without external API dependencies. Given the coefficients of a quadratic equation ($ax^2 + bx + c = 0$), the workflow calculates the discriminant ($\Delta = b^2 - 4ac$) and routes execution to the appropriate formula depending on whether roots are distinct real, repeated, or complex.

### 1. State Definition

```python
from typing import TypedDict, Literal

class QuadState(TypedDict):
    a: int
    b: int
    c: int
    equation: str
    discriminant: float
    result: str
```

- **Inputs**: `a`, `b`, `c`
- **Intermediates**: `equation`, `discriminant`
- **Outputs**: `result`

### 2. Processing Nodes

- **`show_equation`**: Formats the string representation of the equation.
- **`calculate_discriminant`**: Computes $b^2 - 4ac$.
- **`real_roots`**: Executes quadratic formula for two distinct real roots ($\Delta > 0$).
- **`repeated_roots`**: Calculates the single repeated root ($\Delta = 0$).
- **`no_real_roots`**: Returns message indicating no real roots exist ($\Delta < 0$).

Each node only updates the key it is responsible for (partial state update):

```python
def calculate_discriminant(state: QuadState):
    discriminant = state["b"]**2 - (4 * state["a"] * state["c"])
    return {'discriminant': discriminant}

def real_roots(state: QuadState):
    root1 = (-state["b"] + state["discriminant"]**0.5) / (2 * state["a"])
    root2 = (-state["b"] - state["discriminant"]**0.5) / (2 * state["a"])
    return {'result': f'The roots are {root1} and {root2}'}

def repeated_roots(state: QuadState):
    root = (-state["b"]) / (2 * state["a"])
    return {'result': f'Only repeating root is {root}'}

def no_real_roots(state: QuadState):
    return {'result': 'No real roots'}
```

### 3. The Router Function

The routing decision is implemented in a separate python function that reads `state['discriminant']` and returns a `Literal` matching the target node name:

```python
def check_condition(state: QuadState) -> Literal["real_roots", "repeated_roots", "no_real_roots"]:
    if state['discriminant'] > 0:
        return "real_roots"
    elif state['discriminant'] == 0:
        return "repeated_roots"
    else:
        return "no_real_roots"
```

### 4. Graph Construction and Conditional Edges

```python
graph = StateGraph(QuadState)

# Add all nodes
graph.add_node('show_equation', show_equation)
graph.add_node('calculate_discriminant', calculate_discriminant)
graph.add_node('real_roots', real_roots)
graph.add_node('repeated_roots', repeated_roots)
graph.add_node('no_real_roots', no_real_roots)

# Fixed sequential edges
graph.add_edge(START, 'show_equation')
graph.add_edge('show_equation', 'calculate_discriminant')

# Conditional edge: inspects discriminant and branches
graph.add_conditional_edges('calculate_discriminant', check_condition)

# Terminal edges
graph.add_edge('real_roots', END)
graph.add_edge('repeated_roots', END)
graph.add_edge('no_real_roots', END)

workflow = graph.compile()
```

```
START ──► show_equation ──► calculate_discriminant
                                   │
                  ┌────────────────┼────────────────┐
                  ▼ (Δ > 0)        ▼ (Δ == 0)       ▼ (Δ < 0)
              real_roots     repeated_roots   no_real_roots
                  │                │                │
                  └────────────────┼────────────────┘
                                   ▼
                                  END
```

### 5. Execution Example

```python
initial_state = {'a': 2, 'b': 4, 'c': 2}
workflow.invoke(initial_state)
```

**Output:**
```python
{
  'a': 2,
  'b': 4,
  'c': 2,
  'equation': '2x24x2',
  'discriminant': 0,
  'result': 'Only repeating root is -1.0'
}
```

Since $4^2 - 4(2)(2) = 0$, `check_condition` routed execution directly to `repeated_roots`, skipping `real_roots` and `no_real_roots`.

---

## Code 2: Review Reply Workflow (LLM-Based Routing)

📄 **File:** `7_review_reply_workflow.ipynb`

This notebook demonstrates an intelligent, multi-step customer support triage system. Incoming customer reviews are dynamically routed based on sentiment:
- **Positive Reviews**: Routed to a single-step path that writes an appreciative thank-you response.
- **Negative Reviews**: Routed to a two-step diagnostic path that analyzes the root cause (issue type, tone, urgency) and then generates an empathetic support resolution.

### 1. Pydantic Structured Outputs

To make routing deterministic and free of LLM hallucinations, structured output schemas are defined using Pydantic:

```python
from pydantic import BaseModel, Field

class SentimentSchema(BaseModel):
    sentiment: Literal["positive", "negative"] = Field(description='Sentiment of the review')

class DiagnosisSchema(BaseModel):
    issue_type: Literal["UX", "Performance", "Bug", "Support", "Other"] = Field(
        description='The category of issue mentioned in the review'
    )
    tone: Literal["angry", "frustrated", "disappointed", "calm"] = Field(
        description='The emotional tone expressed by the user'
    )
    urgency: Literal["low", "medium", "high"] = Field(
        description='How urgent or critical the issue appears to be'
    )

structured_model = model.with_structured_output(SentimentSchema)
structured_model2 = model.with_structured_output(DiagnosisSchema)
```

### 2. State Definition

```python
class ReviewState(TypedDict):
    review: str
    sentiment: Literal["positive", "negative"]
    diagnosis: dict
    response: str
```

### 3. Nodes and Conditional Routing

#### Classifier Node & Router Function
```python
def find_sentiment(state: ReviewState):
    prompt = f'For the following review find out the sentiment \n {state["review"]}'
    sentiment = structured_model.invoke(prompt).sentiment
    return {'sentiment': sentiment}

def check_sentiment(state: ReviewState) -> Literal["positive_response", "run_diagnosis"]:
    if state['sentiment'] == 'positive':
        return 'positive_response'
    else:
        return 'run_diagnosis'
```

#### Positive Branch
```python
def positive_response(state: ReviewState):
    prompt = f"""Write a warm thank-you message in response to this review:
    \"{state['review']}\"
Also, kindly ask the user to leave feedback on our website."""
    response = model.invoke(prompt).content
    return {'response': response}
```

#### Negative Branch (Two-Step Chain)
```python
def run_diagnosis(state: ReviewState):
    prompt = f"""Diagnose this negative review:\n\n{state['review']}\nReturn issue_type, tone, and urgency."""
    response = structured_model2.invoke(prompt)
    return {'diagnosis': response.model_dump()}

def negative_response(state: ReviewState):
    diagnosis = state['diagnosis']
    prompt = f"""You are a support assistant.
The user had a '{diagnosis['issue_type']}' issue, sounded '{diagnosis['tone']}', and marked urgency as '{diagnosis['urgency']}'.
Write an empathetic, helpful resolution message."""
    response = model.invoke(prompt).content
    return {'response': response}
```

### 4. Graph Architecture: Asymmetric Branching

Notice the difference in path length:
- `find_sentiment` $\rightarrow$ `positive_response` $\rightarrow$ `END` (1 downstream step)
- `find_sentiment` $\rightarrow$ `run_diagnosis` $\rightarrow$ `negative_response` $\rightarrow$ `END` (2 downstream steps)

```python
graph = StateGraph(ReviewState)

graph.add_node('find_sentiment', find_sentiment)
graph.add_node('positive_response', positive_response)
graph.add_node('run_diagnosis', run_diagnosis)
graph.add_node('negative_response', negative_response)

# Start by classifying sentiment
graph.add_edge(START, 'find_sentiment')

# Branch dynamically based on sentiment
graph.add_conditional_edges('find_sentiment', check_sentiment)

# Branch 1 terminates immediately after thank you
graph.add_edge('positive_response', END)

# Branch 2 chains diagnosis into response generation
graph.add_edge('run_diagnosis', 'negative_response')
graph.add_edge('negative_response', END)

workflow = graph.compile()
```

```
                                  check_sentiment
                                         │
                    ┌────────────────────┴────────────────────┐
                    ▼ ("positive")                            ▼ ("negative")
            positive_response                          run_diagnosis
                    │                                         │
                    │                                         ▼
                    │                                 negative_response
                    │                                         │
                    └────────────────────┬────────────────────┘
                                         ▼
                                        END
```

### 5. Execution Example

When given a critical bug report:
```python
initial_state = {
    'review': "I’ve been trying to log in for over an hour now, and the app keeps freezing on the authentication screen. I even tried reinstalling it, but no luck. This kind of bug is unacceptable, especially when it affects basic functionality."
}
workflow.invoke(initial_state)
```

**Output State:**
```python
{
  'review': 'I’ve been trying to log in for over an hour now...',
  'sentiment': 'negative',
  'diagnosis': {
      'issue_type': 'Bug',
      'tone': 'frustrated',
      'urgency': 'high'
  },
  'response': "Subject: We're Here to Help You with the Bug Issue\n\nHi [User's Name],\n\nThank you for reaching out and bringing this issue to our attention. I understand how frustrating it can be to deal with a bug, especially when it feels urgent. I'm here to help you resolve this as quickly as possible..."
}
```

The system bypassed `positive_response`, diagnosed the problem as a high-urgency bug with a frustrated tone, and generated an empathetic, context-aware support reply.

---

## Key Takeaways

1. **`add_conditional_edges(source, router_function)`**: Connects a source node to dynamically chosen downstream nodes based on the output of `router_function`.
2. **Type Safety with `Literal`**: Annotating router return values with `typing.Literal["node_a", "node_b"]` ensures clarity and helps LangGraph validate graph connectivity.
3. **Structured Outputs for Reliable Routing**: When an LLM is the router, pairing it with Pydantic structured schemas (`with_structured_output`) guarantees reliable output values matching the expected branch names.
4. **Asymmetric Workflow Paths**: Conditional workflows allow simple cases to finish fast while complex cases pass through multi-step diagnostic or refinement pipelines.
