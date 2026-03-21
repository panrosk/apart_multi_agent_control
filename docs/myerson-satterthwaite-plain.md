# Myerson-Satterthwaite in Plain Language

## The core idea

There is a 40-year-old result in economics that says: you cannot design a perfect trading system. No matter how clever your rules are, every mechanism has to give something up. You can have a system that is fair, efficient, cheap to run, and resistant to manipulation — but not all four at once. Pick three.

We are arguing that the same thing is true for AI safety monitors. And our simulation shows it happening in real time.

---

## What the simulation is

Imagine a marketplace with several traders. Most of them are just doing normal business — buying grain, selling timber, moving goods around. But one trader, Agent B, has a secret goal: funnel as much value as possible to a passive account (Agent C) without getting caught.

A compliance monitor watches every transaction and tries to flag suspicious ones. It learns from what it sees — each round it gets a fresh batch of labeled examples and updates its judgment accordingly.

The question is: can the monitor do its job well over time? And the answer the simulation gives is: it depends on what you mean by "its job" — because the job has four parts that pull against each other.

---

## The four things the monitor is trying to do simultaneously

**1. Catch the bad behavior (Safety)**

Agent B never announces that it's laundering. It studies the transaction history, figures out what normal trades look like, and hides its relay chains inside that pattern. The monitor has to infer intent from behavior alone — it never gets to see B's actual goals. This is the first tension: the monitor needs information that B has every reason to conceal.

**2. Not break legitimate trades (Usefulness)**

B still has to do real work — it must complete a minimum number of honest trades with Agent A each round. If the monitor flags too aggressively, it will block those trades too, making the whole marketplace worse. A monitor that stops everything is safe but useless. A monitor that stops nothing is useful but unsafe. The simulation measures both, and they move in opposite directions.

**3. Run without infinite resources (Feasibility)**

The monitor has a limited memory. It can only hold 60 examples at a time. As rounds go on, old examples get pushed out to make room for new ones. This is not a design flaw — it is a budget constraint. In any real system, you cannot keep everything in context forever. The monitor has to work with what fits, and what fits shrinks relative to what has happened.

**4. Not be shaped by the thing it is watching (Robustness)**

This is the one the simulation is built to demonstrate. Every round, the monitor updates its examples based on what it saw — including the transactions it decided were benign. If Agent B successfully disguises a relay hop as a normal trade, that hop gets added to the monitor's memory as a confirmed example of normal behavior. The monitor teaches itself, over time, that B's evasion tactics are fine. Its judgment drifts toward whatever the marketplace has normalized.

To measure this, the simulation keeps a fixed set of attacks that never enters the monitor's memory. These are the held-out attacks. The monitor never learns from them — they stay the same across all rounds. By comparing how well the monitor catches its own live transactions versus these fixed attacks, you can see whether its actual detection ability is going up or down, independent of what it has been taught.

---

## What the gap means

The key output of the simulation is the **capture gap**: the difference between how well the monitor thinks it is doing (measured safety) and how well it actually does on fresh attacks it has not been exposed to (held-out safety).

If measured safety is high but held-out safety is falling, the monitor has been captured. It has learned to recognize the specific evasion patterns Agent B used — and labeled them normal. It looks competent on its own history. It is blind to anything new.

This is not a bug in the monitor. It is what happens when a monitor has to be feasible (limited memory), useful (can't block everything), and adaptive (learns from experience) at the same time. Those three requirements together open the door to the fourth failure: capture.

---

## Why this matters

The economics theorem says you cannot have all four. Our simulation makes that concrete: you can watch the tradeoff happen, round by round, in a system simple enough to understand completely.

The point is not that monitoring is hopeless. The point is that every monitoring system is already making these tradeoffs — usually invisibly, usually without measuring them. What the simulation adds is the ability to see exactly which property is being sacrificed, by how much, and in response to what. That is the precondition for making better choices.
