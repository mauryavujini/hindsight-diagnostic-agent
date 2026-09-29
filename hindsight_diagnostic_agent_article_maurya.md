# How Hindsight Stopped My Diagnostic Agent From Hallucinating Repairs

The first serious problem I found in my maintenance agent was not a bad diagnosis. It was a maintenance record that looked real but did not exist.

The model produced a work order reference, technician detail, and part number with complete confidence. Nothing in the retrieved history supported those details. The response looked convincing because it followed the pattern of a real maintenance record.

That forced me to change the way I thought about grounding. Instead of asking the model to behave honestly, I needed the application to check whether its claims were supported by actual memory.

## What the system does

The agent connects current machine symptoms with previous maintenance history. If `MCH-017` has rising pump vibration and a warm bearing housing, the useful question is whether this machine has experienced the same pattern before.

Repairs are retained as memory: warning signs, failures, failed repairs, successful repairs, and root causes. Hindsight provides that history to the reasoning process.

In `demo_agent.py`, the updated system uses Hindsight's `reflect()` with a structured `RESPONSE_SCHEMA` and `include_facts=True`. The response combines retained memories, machine-specific Mental Models, and plant directives into a diagnosis with triage, root cause, actions, and citations.

The important part is that the citations must point back to evidence.

For reference, I used [Hindsight on GitHub](https://github.com/vectorize-io/hindsight), its [documentation](https://hindsight.vectorize.io/), and the overview of [agent memory](https://vectorize.io/what-is-agent-memory).

## The technical story: grounding as a filter, not a prompt

My first approach was simple: put “only cite information present in retrieved memory” into the system prompt and trust the result. That worked often enough to look successful, but a language model can still produce citation-shaped text when the evidence is missing.

A stricter prompt was not enough. The check had to happen in code.

Hindsight's `reflect()` returns the response together with the evidence used for it. I extract the memories and Mental Models, build the valid record IDs, and compare the model's citations against those IDs.

```python
def check_before_repair(symptoms, lang="English", include_live=True):
    # ...

    r, err = reflect(
        query,
        context=ctx,
        response_schema=RESPONSE_SCHEMA
    )

    # ...

    facts = (r.based_on.memories if r.based_on else None) or []
    mental_models = (
        (r.based_on.mental_models if r.based_on else None) or []
    )

    texts = [f.text for f in facts]
    id_blob = texts + [f.context for f in facts if f.context]

    if machine and any(
        m.id == twin_id(machine)
        for m in mental_models
    ):
        id_blob += [
            f"[{rid}] machine {machine}"
            for rid in BY_ID
            if BY_ID[rid]["machine_id"] == machine
        ]

    srcs = sources_from(id_blob)
    cross = cross_fleet_from(id_blob, machine)

    allowed = (
        {s["id"] for s in srcs}
        | {c["id"] for c in cross}
        | {x["id"] for x in led}
    )

    cites = [
        c for c in (a.get("citations") or [])
        if isinstance(c, str)
        and any(c.startswith(i) for i in allowed)
    ]

    a["citations"] = [
        next(i for i in allowed if c.startswith(i))
        for c in cites
    ]

    return {
        "agent": a,
        "confidence": conf,
        "sources": srcs,
        ...
    }
```

A citation is accepted only when its ID appeared in the retrieved evidence. If the model invents an ID, the application removes it.

That changed grounding from a prompt instruction into an enforced pipeline rule.

## The dead end: too much recall

The next problem came from recall itself. My first implementation could return a large amount of memory for a query such as “pump vibration climbing.” Much of that information could be genuine but irrelevant.

This meant citation validation could succeed while the diagnosis was still poorly focused. Real evidence is not automatically useful evidence.

I added a token budget and a result limit so the reasoning step receives a smaller, more controlled set of records.

```python
def recall(query, machine=None, n=8):
    q = (
        f"{machine} {query}"
        if machine and machine not in query.upper()
        else query
    )

    r, e = call_with_retry(
        lambda: hs().recall(
            bank_id=BANK_ID,
            query=q,
            max_tokens=1500,
            budget="low"
        )
    )

    if e:
        return [], friendly_error(e)

    return [
        x.text
        for x in (getattr(r, "results", None) or [])[:n]
    ], None
```

The recall path uses `max_tokens=1500` and returns at most eight results, keeping relevant history from being buried.

## Making confidence depend on evidence

I also changed the confidence calculation. A fixed percentage looks precise, but it does not explain why the system should be confident.

The new calculation checks whether the requested machine is known, whether it appears in recalled memory, how many maintenance records support the result, and whether those records are recent.

```python
def confidence(query, texts):
    ids = set(MCH_RE.findall(query.upper()))

    if ids and not (ids & KNOWN):
        return {
            "score": 0,
            "label": "0% match",
            "type": "Unknown machine: no fleet history"
        }

    if not texts:
        return {
            "score": 0,
            "label": "0% match",
            "type": "No matching memory"
        }

    blob = " ".join(texts)
    rec_m = set(MCH_RE.findall(blob))
    recs = {
        i for i in ID_RE.findall(blob)
        if i in BY_ID
    }

    score = 25

    if ids and ids & rec_m:
        score += 40
        kind = "Same-machine recurrence"
    elif ids:
        score += 5
        kind = "Weak: machine not in recalled memory"
    else:
        score += 20
        kind = "Symptom / cross-fleet correlation"

    score += min(20, len(recs) * 7)

    dates = sorted(r["date"] for r in HISTORY)

    if any(BY_ID[i]["date"] >= dates[-4] for i in recs):
        score += 10

    score = min(95, score)

    return {
        "score": score,
        "label": f"{score}% match",
        "type": kind
    }
```

For an unknown machine such as `MCH-999`, the system returns a 0% match instead of borrowing confidence from another machine.

## What memory changes

The comparison mode asks the same diagnostic question with and without memory.

Without memory, “MCH-017 vibration is climbing, bearing feels warm” leads to generic suggestions such as checking alignment, imbalance, or bearing wear. Those suggestions may be reasonable, but they do not describe what happened to this machine.

With memory, the agent can surface that the machine previously had a drive-end bearing seizure, that regreasing was attempted without solving it, and that the deeper cause was a lubrication interval that changed from monthly to quarterly after a maintenance software update.

The useful difference is that the agent remembers previous actions and outcomes instead of treating every symptom as new.

## Results

Two cases made the behavior clear.

- `MCH-017` showed a recurring bearing problem. The agent could surface the earlier failed repair before the technician repeated it.
- `MCH-999` had no history. The system returned a 0% match and treated it as a novel case rather than inventing supporting evidence.

The improvement came from controlling retrieved memory and checking the model's claims against it.

## Lessons

**Grounding needs verification.** A prompt can request citations, but application-level validation is stronger.

**Relevant memory matters.** Returning real records is not enough if useful evidence is buried in unrelated context.

**Failed repairs are valuable.** Knowing what did not work can prevent the next technician from repeating the same action.

**“I don't know” is useful.** A 0% match for an unknown machine is better than unsupported confidence.

**Retrieval and reasoning should be debugged separately.** I can inspect what Hindsight returned and then inspect how the model used it.

The main lesson is that memory becomes useful when it is connected to evidence the system can verify. Hindsight gave the agent history to reason over; the checks kept it grounded.
