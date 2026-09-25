# Agentic AI

## 1. What is Agentic AI?

**Agentic AI** is a type of AI that can take on a task or goal from a user and then work toward completing it on its own, with minimal human guidance.

It has four core behaviors:

- **Plans** — breaks the goal into a sequence of steps
- **Takes action** — executes those steps using available tools
- **Adapts to changes** — adjusts its plan and actions when conditions shift
- **Seeks help only when necessary** — escalates to a human only at meaningful checkpoints, not for every micro-decision

Unlike a simple chatbot that responds to a single prompt and stops, an agentic system keeps working toward a persistent objective across multiple steps and tool calls until the goal is achieved or it needs help.

---

## 2. Key Characteristics of Agentic AI

Agentic AI is built on six pillars:

- **Autonomous** — acts on its own with minimal step-by-step human instruction
- **Goal Oriented** — keeps a persistent objective and works continuously toward it
- **Planning** — breaks a high-level goal into a structured sequence of actions
- **Reasoning** — interprets information, draws conclusions, and makes decisions
- **Adaptability** — modifies plans and actions in response to unexpected conditions
- **Context Awareness** — understands, retains, and uses relevant information across a multi-step process

A practical example is an AI recruiter agent hiring a backend engineer.

---

## 3. Autonomy

**Definition:** Autonomy refers to the AI system's ability to make decisions and take actions on its own to achieve a given goal, without needing step-by-step human instructions.

**Example:** The AI recruiter is autonomous because it is proactive, not just reactive.

### Autonomy operates in multiple facets

1. **Execution** — carrying out tasks
2. **Decision making** — choosing between options
3. **Tool usage** — deciding which tools to invoke

### Autonomy can and should be controlled

- **Permission Scope** — limits what tools or actions the agent can perform independently
- **Human-in-the-Loop (HITL)** — requests approval before high-risk actions
- **Override Controls** — allows users to stop, pause, or change behavior at any time
- **Guardrails / Policies** — hard rules or ethical boundaries the agent must follow

### Autonomy can be dangerous if uncontrolled

- Sending job offers with incorrect salary or terms
- Shortlisting candidates using prohibited criteria such as age or nationality
- Spending excess budget on ads without approval

> Autonomy is powerful but needs boundaries — permission scopes, HITL checkpoints, override controls, and guardrails keep it safe.

---

## 4. Goal Oriented

**Definition:** Being goal-oriented means the AI system operates with a persistent objective in mind and continuously directs its actions to achieve that objective, rather than responding to isolated prompts.

### Key ideas

1. **Goals act as a compass** — every autonomous action is checked against the objective
2. **Goals can have constraints** — boundaries the agent must respect while pursuing the goal
3. **Goals are stored in memory** — persisted as structured data that the agent can reference and update over time
4. **Goals can be altered** — by the user or by the agent adapting to new information

### Example goal stored in memory

```json
{
  "main_goal": "Hire a backend engineer",
  "constraints": {
    "experience": "2-4 years",
    "remote": true,
    "stack": ["Python", "Django", "Cloud"]
  },
  "status": "active",
  "created_at": "2025-06-27",
  "progress": {
    "JD_created": true,
    "posted_on": ["LinkedIn", "AngelList"],
    "applications_received": 8,
    "interviews_scheduled": 2
  }
}
```

This lets the agent and user always know: what the goal is, what constraints bound it, and what progress has been made.

---

## 5. Planning

**Definition:** Planning is the agent's ability to break down a high-level goal into a structured sequence of actions or subgoals and decide the best path toward the desired outcome.

### Step 1: Generate multiple candidate plans

- **Plan A:** Post the JD on LinkedIn, GitHub Jobs, and AngelList
- **Plan B:** Use internal referrals and hiring agencies

### Step 2: Evaluate each plan against criteria

- **Efficiency** — which is faster?
- **Tool availability** — which tools are actually available?
- **Cost** — does it require premium tools?
- **Risk** — will it fail if we get no applicants?
- **Alignment** — does it fit constraints such as remote-only and budget?

### Step 3: Select the best plan

- With **Human-in-the-loop input** — for example, “Which option do you prefer?”
- Or with a **pre-programmed policy** — for example, “Favor low-cost channels first.”

---

## 6. Reasoning — and How Planning Fits Into It

**Definition:** Reasoning is the cognitive process through which an agentic AI system interprets information, draws conclusions, and makes decisions — both while planning ahead and while executing actions in real time.

This is the key link between planning and reasoning: **planning is not a separate, one-time step — it is one of the two phases in which reasoning operates.** Reasoning happens continuously, but it shows up differently depending on when it is being applied: before action starts (planning) versus while action is underway (execution).

### A. Reasoning during planning

This is the “thinking before doing” phase — reasoning is used to construct the plan itself:

1. **Goal decomposition** — breaking the abstract goal (“Hire a backend engineer”) into concrete steps such as posting the JD, screening resumes, and scheduling interviews
2. **Tool selection** — reasoning about which tools are needed for each step
3. **Resource estimation** — reasoning about time, dependencies, and risks

At this stage, reasoning allows the agent to generate and evaluate candidate plans such as Plan A versus Plan B.

### B. Reasoning during execution

This is the “thinking while doing” phase — reasoning is used once the plan is being carried out and reality does not always match the plan:

1. **Decision-making** — choosing between live options
2. **HITL handling** — knowing when to pause and ask a human for help
3. **Error handling** — interpreting failures and recovering from them

### Putting it together

| Phase | Role of Reasoning | Role of Planning |
| --- | --- | --- |
| Before action (Planning) | Reasoning generates and evaluates candidate paths | Planning is the output — the structured sequence chosen |
| During action (Execution) | Reasoning makes live decisions and handles errors/HITL | The plan is the reference point being followed or adjusted |

So: **Planning is reasoning applied upfront to produce a roadmap; execution-time reasoning is what keeps the agent making good decisions as it walks that roadmap and the real world pushes back.** When execution-time reasoning detects a meaningful change, it hands off to **Adaptability** to revise the plan itself.

---

## 7. Adaptability

**Definition:** Adaptability is the agent's ability to modify its plans, strategies, or actions in response to unexpected conditions while staying aligned with the goal.

### Triggers for adaptation

1. **Failures** — for example, the Calendar API goes down
2. **External feedback** — for example, fewer applications than expected
3. **Changing goals** — for example, the user decides to hire a freelancer instead of a full-time employee

> Adaptability is what keeps the agent's execution-time reasoning useful in the real world — it is the mechanism that actually rewrites or reroutes the plan when the original path no longer fits.

---

## 8. Context Awareness

**Definition:** Context awareness is the agent's ability to understand, retain, and utilize relevant information from the ongoing task, past interactions, user preferences, and environmental cues to make better decisions throughout a multi-step process.

### Types of context

- **The original goal** — “Hire a backend engineer”
- **Progress to date and interaction history** — the job description was finalized and posted
- **Environment state** — the number of applicants and time remaining in a campaign
- **Tool responses** — resume parser results or calendar availability data
- **User-specific preferences** — remote-first candidates, specific interview formats
- **Policies and guardrails** — approval rules and platform constraints

### How context awareness is implemented

Context awareness is implemented through **memory**, split into two types:

1. **Short-term memory** — holds immediate, task-specific context such as the current step and recent tool outputs
2. **Long-term memory** — persists across sessions and tasks, including user preferences and recurring policies

---

## 9. How It All Connects

```text
Goal (Goal Oriented)
   │
   ▼
Reasoning (Planning phase) → Planning → Candidate Plans → Best Plan Selected
   │
   ▼
Autonomy executes the plan (within Permission Scope / Guardrails)
   │
   ▼
Reasoning (Execution phase): Decision-making, HITL handling, Error handling
   │
   ├── If an unexpected condition arises → Adaptability revises the plan/action
   │
   ▼
Context Awareness (memory) informs every step above throughout the process
```

**In short:** A goal is set → reasoning drives planning to produce a structured plan → autonomy executes it within safe boundaries → reasoning continues in real time to make decisions and catch errors → adaptability reroutes the plan when needed → and context awareness, via short-term and long-term memory, feeds relevant information into every step so decisions stay informed and consistent.

---

## 10. Components of an Agentic AI System

While Sections 3–8 describe the *behaviors* an agentic system exhibits, this section describes the actual *building blocks* that implement those behaviors. An agentic AI system is made up of five core components:

### 10.1 Brain (LLM)

The **Brain** is the LLM at the core of the agent — the part that actually “thinks.”

| Function | What it does |
| --- | --- |
| Goal Interpretation | Understands user instructions and translates them into objectives |
| Planning | Breaks down high-level goals into subgoals and ordered steps |
| Reasoning | Makes decisions, resolves ambiguity, and evaluates trade-offs |
| Tool Selection | Chooses which tool or tools to use at a given step |
| Communication | Generates natural language outputs for humans or other agents |

> This is where the reasoning and planning behaviors from Sections 5–6 are actually carried out.

### 10.2 Orchestrator

The **Orchestrator** is the “conductor” — it does not invent the plan, but it controls how the plan actually flows and executes.

| Function | What it does |
| --- | --- |
| Task Sequencing | Determines the order of actions |
| Conditional Routing | Directs flow based on context and conditions |
| Retry Logic | Handles failed tool calls or reasoning attempts with backoff |
| Looping & Iteration | Repeats steps until a condition is met |
| Delegation | Decides whether to hand off work to tools, the LLM, or a human |

### 10.3 Tools

**Tools** let the agent reach outside itself and act on or retrieve information from the real world.

| Function | What it does |
| --- | --- |
| External Actions | Perform API calls such as posting a job, sending an email, or triggering onboarding |
| Knowledge Base Access | Retrieve factual or domain-specific information using RAG or search tools |

### 10.4 Memory

**Memory** is the concrete implementation behind the context-awareness behavior described in Section 8.

| Function | What it does |
| --- | --- |
| Short-Term Memory | Maintains the active session’s context, recent user messages, tool calls, and immediate decisions |
| Long-Term Memory | Persists goals, past interactions, user preferences, and decisions across sessions |
| State Tracking | Monitors progress such as “JD posted” or “Offer sent” |

### 10.5 Supervisor

The **Supervisor** is the human-oversight layer — the piece that keeps autonomy safe and controlled.

| Function | What it does |
| --- | --- |
| Approval Requests (HITL) | Checks with a human before high-risk actions |
| Guardrails Enforcement | Blocks unsafe or non-compliant behavior |
| Edge Case Escalation | Alerts humans when uncertainty or conflict arises |

### Mapping components to behaviors

| Component | Primarily implements |
| --- | --- |
| Brain | Reasoning, planning, goal interpretation |
| Orchestrator | Execution flow and plan sequencing |
| Tools | Autonomy and real-world action |
| Memory | Context awareness |
| Supervisor | Safe autonomy, HITL, guardrails |

---

## 11. Generative AI vs. Agentic AI

It is easy to confuse the two, but they solve different problems and sit at different layers:

| Generative AI | Agentic AI |
| --- | --- |
| Is about creating content | Is about solving a goal |
| Is reactive — responds to a single prompt | Is proactive — pursues a goal across multiple steps with minimal prompting |
| Produces an output and stops | Keeps working, planning, and adapting until the goal is achieved |

> **Key relationship:** Generative AI is a building block of agentic AI. The “Brain” of an agentic system is typically an LLM — i.e., a generative AI model. Agentic AI wraps that generative capability with planning, memory, tools, orchestration, and supervision so it can go beyond generating a single response and actually pursue and complete a goal.

---

## Summary

Agentic AI combines:

- a clear goal,
- reasoning and planning,
- tool use and autonomy,
- memory and context awareness,
- and human oversight.

That combination allows an AI system to act more like a problem-solving assistant or teammate than a static chatbot.
