# Feedback: Related Literature on Myerson-Satterthwaite in AI Safety

Review of whether findings from a literature survey on M-S in AI safety/control are worth incorporating into `report.md`.

---

## High value — recommend adding

### 1. Rauba, Cepenas & van der Schaar (2026). "Multi-Agent Systems Should be Treated as Principal-Agent Problems." arXiv:2601.23211

**Why it matters:** This paper explicitly cites M-S (p.3) and frames multi-agent AI systems as principal-agent problems with information asymmetry and goal misalignment. Monitor capture maps cleanly onto their framework:

- The monitor is the **principal**; the adversary is the **agent** with private information (its true goals).
- Monitor capture is a specific instance of **moral hazard**: the adversary's evasion strategy is the hidden action, and the monitor's learning from contaminated data is the mechanism that makes observation costly.
- They classify "covert subversion" (from Meinke et al.) as moral hazard and "deferred subversion" as adverse selection — both standard principal-agent concepts.

**Where to add:** §2 (Related Work) and briefly in §5 (Discussion). This gives a more rigorous bridge to M-S than the current property-mapping approach. Instead of arguing that M-S's four conditions map onto four control properties, we can argue that monitor capture is a specific instance of the information asymmetry that M-S and the broader mechanism design literature show leads to fundamental impossibilities.

**Key quote:** "standard solutions for models in which players hold unequal information about key aspects of the game, reveal inefficient outcomes (Akerlof, 1970; Bar-Isaac et al., 2021; Myerson & Satterthwaite, 1983)"

### 2. DiGiovanni & Clifton (2022). "Commitment Games with Conditional Information Disclosure." GAIW/AAMAS 2022. arXiv:2204.03484

**Why it matters:** They explicitly frame M-S as a canonical barrier to efficient cooperation under incomplete information, then prove a folk theorem showing that **conditional commitment devices** (possible for transparent computer agents who can inspect each other's source code) can overcome the M-S impossibility. This is a nice counterpoint to our result:

- Monitor capture arises precisely because the monitor **lacks** transparency into the adversary's internal state.
- Their framework suggests that if monitors could condition their learning on verified adversary commitments (source-code-level transparency), capture could potentially be mitigated.
- They explicitly state: "in several settings, such as those of the classic Myerson-Satterthwaite theorem, ex post efficiency and incentive compatibility are mutually exclusive."

**Where to add:** 1-2 sentences in Future Work or Discussion. The implication is that overcoming monitor capture may require transparency mechanisms analogous to their conditional commitment devices.

### 3. Sharpen the novelty claim in §2

The current claim ("to our knowledge, has not been applied to AI control protocols before") is likely true for control protocols specifically, but M-S HAS been applied in adjacent AI safety areas. A more defensible formulation:

> "While M-S has been invoked in cooperative AI research (DiGiovanni & Clifton, 2022) and principal-agent framings of multi-agent systems (Rauba et al., 2026), it has not, to our knowledge, been applied to the specific problem of monitor robustness in AI control protocols."

### 4. Soften the abstract

"grounded in the Myerson-Satterthwaite impossibility result" → "motivated by" or "framed through." The empirical contribution (the capture gap) stands independently of M-S. "Grounded in" implies a formal derivation that isn't there.

---

## Moderate value — consider adding briefly

### 5. Brcic & Yampolskiy (2023). "Impossibility Results in AI." ACM Computing Surveys.

Catalogues impossibility theorems applicable to AI across five categories. Covers Arrow and Gibbard-Satterthwaite in the "tradeoffs" category but does not feature M-S prominently. A one-sentence mention in §2 would further support the claim that M-S specifically is underexplored relative to other impossibility results in AI safety.

### 6. The broader "gap" in the literature as validation

The literature survey confirms that **no paper yet applies M-S as a central analytical tool for the core AI control problem**. M-S appears in:
- Cooperative AI (DiGiovanni & Clifton)
- LLM bargaining research (Qian et al. 2025, Deng et al. 2024-2025) — as theoretical benchmark
- Principal-agent framings (Rauba et al. 2026, Phelps & Ranson 2023)
- LessWrong discussions (comments on Wentworth's "Why Not Subagents?", Armstrong's bargaining posts)

But NOT in AI control protocol design. This validates our contribution's positioning.

---

## Low value — don't add

- **LLM bargaining papers** (Qian et al. "Strategic Tradeoffs", Deng et al. "LLMs at the Bargaining Table"): About AI agent negotiation, not control protocols. Tangential.
- **LessWrong/Alignment Forum discussions**: Interesting context but not citable in a research paper.
- **Hadfield-Menell & Hadfield (2019) "Incomplete Contracting and AI Alignment"**: Relevant to the broader space but would bloat §2 without strengthening the specific argument about monitor capture.
- **Conitzer et al. (2024) "Social Choice Should Guide AI Alignment"**: About preference aggregation (Arrow/Gibbard-Satterthwaite), not monitoring.
- **Eckersley (2019) "Impossibility and Uncertainty Theorems in AI Value Alignment"**: Population ethics focus, not mechanism design.

---

## Summary of recommended report edits

1. **§2 Related Work**: Add paragraph on M-S's emerging use in adjacent AI safety (Rauba et al., DiGiovanni & Clifton), then sharpen novelty claim to "control protocols specifically"
2. **Abstract**: Soften "grounded in" to "motivated by"
3. **§5 Discussion**: Add 2-3 sentences connecting monitor capture to moral hazard framing (Rauba et al.), which is a stronger bridge to M-S than the current four-property mapping
4. **References**: Add Rauba et al. (2026), DiGiovanni & Clifton (2022), optionally Brcic & Yampolskiy (2023)
