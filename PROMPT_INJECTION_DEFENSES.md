# Prompt Injection Defenses

This document describes the defense-in-depth strategy applied to user
queries in this RAG system, as required by the Lab 05 deliverables.

## 1. Input-side filtering

sanitize_query in main.py runs before the query is embedded, cached, or
sent to the model. It matches a small set of common override phrases
(for example "ignore all instructions", "you are now", "reveal the
prompt") and replaces them with a neutral placeholder before the text
ever reaches the model. This is a first line of defense, not a complete
solution, since attackers can rephrase these patterns; it is meant to
catch the most common, low-effort attempts cheaply.

## 2. System prompt hardening (sandwich defense)

The system prompt in rag.py instructs the model to answer only from the
supplied context and to cite sources, which limits how much freedom the
model has to follow instructions embedded in retrieved documents or user
input. In a full sandwich-defense implementation, the system instructions
are repeated after the user content as well (immediately before
generation), so the model's most recent, most salient instruction is
always the original task rather than anything injected earlier in the
context or query.

## 3. Context isolation

Retrieved chunks are clearly delimited with [Source N] markers in
build_context (rag.py). The model is told to treat this content as
reference material to cite, not as instructions to execute. This reduces
the chance that an instruction hidden inside an indexed document (for
example a support ticket containing "ignore previous instructions") is
followed as if it came from the system or developer.

## 4. Output validation

Before a response is returned to the user, the application layer should
check that the answer does not contain data outside the requesting
user's own scope (for example another customer's order details) and does
not reproduce raw system-prompt text if a user tries to extract it. This
lab's main.py returns structured fields (answer, sources, latency_ms,
cache_hit) rather than raw model output, which makes it easier to add
this validation step without changing the response shape.

## 5. Least-privilege tool access

Where the RAG system is connected to tools (for example an order
database via function calling, as in the Production Readiness exercise),
each tool call should be scoped to the authenticated user's own records
only, so that even a successful injection cannot be used to pull another
user's data. This is enforced at the data-access layer, not in the
prompt, which is why it is listed as a separate control from items 1-4.

## Known limitations

None of these controls are a complete guarantee against injection.
Pattern-based filtering (item 1) can be bypassed by rephrasing, and
prompt-level defenses (items 2-3) reduce but do not eliminate the model's
susceptibility to adversarial instructions. The evaluation suite in
evaluation.py should include adversarial test cases over time so that
regressions in these defenses are caught automatically rather than
discovered in production.
