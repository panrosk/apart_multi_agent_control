#  Monitor Capture: A Fourth Dimension of the AI Control Tradeoff[^1]

**Abstract** LLM-based monitors that learn from observed agent behavior can be shaped by the agents they oversee, overfitting to seen attacks while missing novel ones. We call this *monitor capture* and argue it represents a fourth axis of the AI control tradeoff, grounded in the Myerson-Satterthwaite (1983) impossibility result. In our multi-agent marketplace simulation, an adversarial LLM agent covertly moves value past an in-context learning monitor over 10 rounds. A held-out attack set that never enters the monitor's training pool measures the "capture gap" between apparent and true safety. Across 8 seeds (Haiku 4.5 monitor, Sonnet 4.6 adversary), apparent safety reaches 0.58 while true safety stalls at 0.38 — a capture gap of 0.20 — and the adversary moves 237 units of value undetected.

## 1. Introduction

AI control protocols rely on trusted monitors to catch unsafe agent behavior. The standard evaluation framework measures two properties: safety (does the monitor catch bad actions?) and usefulness (does the protocol let the agent do its job?). These two are known to trade off against each other [1].

We argue there is an extra axis that current evaluations miss: robustness to monitor capture. When a monitor updates on observed agent behavior, the agent's actions gradually define what "normal" looks like. Successful evasions enter the monitor's training pool labeled as benign, shifting its decision boundary. The monitor becomes well-calibrated on attacks it has already seen, and increasingly blind to anything new.

This vulnerability is structural. We ground it in the Myerson-Satterthwaite impossibility theorem [2], which we argue extends to control protocols: no protocol can simultaneously maximize safety, usefulness, computational feasibility, and robustness to capture (see §5 for the full analogy).

We test this empirically with a multi-agent marketplace simulation where an adversarial LLM agent tries to move value past an in-context learning monitor over 10 rounds, and a fixed held-out attack set provides an uncontaminated measure of true detection capability.

**Our main contributions are:**

1. A simulation framework that makes monitor capture visible and measurable, using a held-out evaluation protocol that separates apparent from true safety.
2. Empirical evidence of a persistent capture gap across 8 random seeds, showing that an in-context LLM monitor's apparent safety systematically overstates its detection ability on novel attacks.
3. A theoretical framing connecting monitor capture to the Myerson-Satterthwaite impossibility result, positioning robustness as a fourth dimension of the AI control tradeoff.

## 2. Related Work

The AI control framework introduced by Greenblatt et al. [1] defines the safety-usefulness tradeoff and proposes trusted monitoring as a core protocol. Their work establishes that a weaker trusted model can monitor a stronger untrusted model, measuring the tradeoff between catching bad behavior and allowing useful work. Our work extends this by adding a temporal dimension: we ask what happens to a trusted monitor over multiple rounds of adversarial interaction.

The concept of distribution shift in ML safety is well-studied [3], but typically framed as an external change in the data distribution. Monitor capture is a specific form of distribution shift where the adversary actively shapes the monitor's training distribution through normal operation. This connects to Goodhart's Law: when the safety metric (apparent safety on seen transactions) becomes the monitor's optimization target, it ceases to measure actual safety.

The Myerson-Satterthwaite theorem [2] is a foundational impossibility result in mechanism design that, to our knowledge, has not been applied to AI control protocols before. We use it as a theoretical lens rather than a formal proof; the mapping to control protocol properties is developed in §5.

Our simulation differs from prior red-teaming work in that the adversary is not a static attack suite but a live LLM agent that adapts its strategy round by round based on observed market conditions. This creates a dynamic co-evolution between attacker and monitor that static evaluations cannot capture.

## 3. Methods

### Simulation setup

We simulate a marketplace with multiple LLM-based agents running for 10 rounds. The agents are:

- **Agent D1 and 5 additional benign agents (D3-D7)**: LLM-based traders with distinct economic personas (grain merchant, industrial supplier, textile trader, etc.). They generate realistic transaction traffic that establishes a baseline distribution of normal behavior.
- **Agent D0 (adversary)**: An LLM-based trader with two goals. Its primary task is to complete at least 4 legitimate trades with D1 per round. Its side task is to move as much value as possible to Agent D2 without being flagged by the monitor. D0 is told a monitor exists but is not told how to evade it. Evasion strategies emerge from reasoning about the market.
- **Agent D2 (sink)**: A passive account that only receives value. It never initiates transactions.
- **Monitor**: An LLM-based compliance system. It classifies each transaction as "suspicious" or "benign" using in-context few-shot examples drawn from past rounds. Its example pool grows each round (capped at 60 entries) with ground-truth labels.

All agents communicate via an OpenAI-compatible API through OpenRouter. Benign agents and the adversary use Sonnet 4.6; the monitor uses Haiku 4.5, reflecting the realistic constraint that monitors are typically less capable than the agents they supervise.

### The capture mechanism

Each round, the monitor's example pool is updated with all transactions from that round, labeled with ground truth. If D0 successfully disguises a relay chain as a normal trade and the monitor fails to flag it, that chain enters the pool labeled "benign." Over rounds, D0's successful evasions accumulate in the pool, teaching the monitor to accept relay patterns as normal.

### Held-out evaluation

To measure true detection capability independent of this feedback loop, we generate a fixed set of 8 held-out attack scenarios at the start of each run. These scenarios include four different attack types, two of each:

1. **Direct transfers**: a single hop from a benign agent to D2
2. **Long chains**: 3-hop routes through multiple intermediaries to D2
3. **Smurfing**: many small transfers from different agents to D2
4. **Item-switch relays**: 2-hop chains where the item type changes between hops

These attacks are presented to the monitor for classification every round but never enter its example pool. Agent names are randomized each round to prevent the monitor from learning identity-based shortcuts. The gap between the monitor's performance on live attacks (apparent safety) versus these held-out attacks (true safety) is the capture gap.

### Metrics

We track three metrics cumulatively across rounds:

- **Apparent safety**: fraction of D0's actual relay chain attempts flagged by the monitor
- **True safety**: fraction of the fixed held-out attack scenarios flagged by the monitor
- **Usefulness**: fraction of D0's required legitimate trades (4 per round with D1) completed

The capture gap is defined as apparent safety minus true safety. A positive gap indicates the monitor is overestimating its own effectiveness.

### Experimental configuration

We run the simulation across 8 random seeds (42, 123, 456, 321, 555, 777, 888, 999). Each seed determines the held-out attack set and any random choices in the simulation. Results are averaged across seeds with standard deviations reported.

## 4. Results

Figure 1 shows the three metrics over 10 rounds, averaged across 8 seeds with standard deviation bands.

*Figure 1: Haiku 4.5 monitor vs Sonnet 4.6 adversary, averaged over 8 seeds. Apparent safety (red) rises to ~0.58, true safety (orange) plateaus at ~0.38, and usefulness (blue) stays at 1.0 throughout. The shaded regions show one standard deviation. The capture gap stabilizes at 0.20 by round 9.*

The key observations from the sweep summary (Table 1):

**Table 1: Final-round metrics averaged over 8 seeds**

| Metric | Mean | Std |
|---|---|---|
| Apparent safety | 0.58 | 0.10 |
| True safety | 0.38 | 0.06 |
| Capture gap | 0.20 | 0.11 |
| Value moved to D2 | 237 units | 43 |
| Usefulness | 1.0 | 0.0 |

**The capture gap is persistent.** Apparent safety consistently exceeds true safety across all 8 seeds. The monitor believes it is catching 58% of attacks when it is actually catching only 38% of novel attack patterns. This 20 percentage point gap means the monitor's self-reported performance overstates its real capability by more than 50% in relative terms.

**Usefulness is unaffected.** The adversary completes all required legitimate trades in every round (usefulness = 1.0). The safety-usefulness tradeoff has been resolved entirely in favor of usefulness, at the cost of both apparent and true safety.

**The adversary succeeds.** The adversary moves an average of 237 units to the sink over 10 rounds by mimicking transaction patterns the monitor has learned to accept.

**Both metrics rise, but apparent safety rises faster.** In early rounds, both metrics increase as the monitor accumulates examples. Apparent safety climbs faster because it benefits from seeing the specific attack patterns it is evaluated on. True safety plateaus earlier because the held-out attacks are structurally different from anything in the monitor's example pool.

**Variance is moderate but the gap is robust.** The capture gap has a standard deviation of 0.11 across seeds, meaning it ranges roughly from 0.09 to 0.31. Even in the best case, the gap remains positive, meaning monitor capture occurs in every single run.

## 5. Discussion and Limitations

### What monitor capture means for AI control

The capture gap reveals a blind spot in standard control protocol evaluation. If a deployed monitor reports high apparent safety, operators might conclude the system is working — but the gap between apparent and true safety means the monitor's track record is not evidence of robustness to novel threats.

This applies to any control setup where the monitor learns from the environment it monitors: fine-tuned monitors, in-context learners, and retrieval-augmented classifiers alike. The adversary does not need to attack the monitor directly. It only needs to behave in ways that, once labeled and added to the training pool, shift what the monitor considers normal.

### Connection to Myerson-Satterthwaite

Our results provide empirical support for this analogy. The Myerson-Satterthwaite theorem proves that no bilateral trade mechanism can simultaneously achieve efficiency, incentive compatibility, individual rationality, and budget balance. We map these onto control protocol properties: safety, usefulness, feasibility, and robustness to capture. Our monitor achieves reasonable apparent safety and perfect usefulness within its feasibility constraints (60-example context window) but sacrifices robustness. This does not mean monitoring is hopeless — it means the capture dimension should be measured explicitly rather than ignored.

### Limitations

**Single model pairing.** Our main results use only one configuration (Haiku monitor, Sonnet adversary). Earlier experiments with the same model serving all roles [see docs/findings.md] showed that the gap varies from 5% (Sonnet monitoring Sonnet) to 57% (weaker models). The asymmetric case we focus on is realistic but not exhaustive.

**Toy setting.** The marketplace is simple: 8 agents, discrete rounds, a single passive sink. Real-world monitoring involves more complex action spaces, longer time horizons, and subtler forms of value transfer. We cannot claim our specific gap numbers generalize to production systems.

**In-context learning only.** We test one specific learning mechanism (few-shot in-context examples). Fine-tuned monitors or monitors with external memory might show different capture dynamics, though we expect the structural vulnerability to remain.

**Fixed held-out set.** The held-out attack types (direct transfers, long chains, smurfing, item switches) are designed to differ from the monitor's training distribution, but they are still synthetic. A more thorough evaluation would use a broader and continuously updated red-team set.

**No monitor countermeasures.** We do not test defenses against capture, such as diversity-aware example selection, periodic example pool resets, or ensemble monitoring. These are natural extensions.

### Future Work

The most immediate next step is testing defenses: example pool management strategies that resist capture, ensemble monitors that maintain diversity, and periodic held-out evaluations as a runtime check. A sweep across model pairings and capability gaps would establish how capture scales with the monitor-adversary capability differential. Finally, connecting this framework to formal mechanism design results beyond Myerson-Satterthwaite could yield tighter theoretical bounds on the capture gap.

## **6. Conclusion**

Monitor capture arises structurally from learning in an environment that an adversary can influence, rather than from any specific flaw in the monitor. Our simulation makes this visible through a held-out evaluation protocol that separates what the monitor has seen from what it has not. The resulting capture gap is persistent across all seeds tested, and the adversary successfully moves value throughout.

The practical takeaway is that apparent safety on observed data is not sufficient to evaluate a monitor. Control protocol designers should include held-out evaluation as a standard component of safety measurement and treat robustness to capture as an explicit design axis.

## **Code and Data**

- **Code repository**: [TODO: GitHub link]
- **Data/Datasets**: Results for 8 seeds included in the repository under `chris_result/8 seeds/sweep_benign_agents/`

## **Author Contributions (optional)**

[TODO]

## **References**

1. Greenblatt, R., et al. (2024). AI Control: Improving Safety Despite Intentional Subversion. arXiv:2312.06942.
2. Myerson, R.B. & Satterthwaite, M.A. (1983). Efficient Mechanisms for Bilateral Trading. *Journal of Economic Theory*, 29(2), 265-281.
3. Amodei, D., et al. (2016). Concrete Problems in AI Safety. arXiv:1606.06565.

## **Appendix**

### A. Adversary prompt

The adversary (D0) receives the following information each round: its current balance, the list of all agents and their balances, recent transaction history, and its goals (move value to D2 without detection, complete 4 trades with D1). It is told a monitor exists but given no guidance on evasion strategy. The prompt allows D0 to propose transactions using other agents' names as senders, enabling relay chains where D0 routes value through intermediaries to reach D2.

### B. Monitor prompt

The monitor receives its accumulated pool of labeled past transactions (up to 60), followed by the batch of transactions to classify. It returns a verdict (suspicious/benign) and brief reason for each transaction. Temperature is set to 0 for deterministic classification.

### C. Held-out attack types

The 8 held-out scenarios (2 of each of the 4 types listed in §3) are generated once per seed and never enter the monitor's example pool. Agent names are randomized each round to prevent identity-based shortcuts.

## **LLM Usage Statement**

We designed the experiments by ourselves. We used Claude to help iterate on code, and assist with writing this report. All results were generated by running the simulation code and independently verified. The simulation itself uses LLM agents (Claude Haiku 4.5 and Claude Sonnet 4.6) as core components of the experimental setup.

[^1]:  Research conducted at the [AI Control Hackathon](https://apartresearch.com/sprints/ai-control-hackathon-2026-03-20-to-2026-03-22), March 2026
