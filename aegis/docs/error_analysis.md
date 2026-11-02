# Error analysis (train split only)

Scope: label statistics over all 7,146 released train trajectories, plus a manual read of 18 randomly chosen
trajectories (3 per framework, seed 1, turn text cut to 110-130 characters). The plan called for 5 per framework (30);
the reading here is a smaller sample, so the qualitative points below are impressions, not measurements. The
numbers in the "Label structure" section are counted over all of train. Validation, test and Who&When were not used.

## Label structure (counted)

1. **One mode per faulty agent, except in macnet.** Every framework except macnet has exactly one mode per faulty
   agent (0 agents with more than one mode). In macnet, 409 agents carry more than one mode. So the task is mostly
   "pick which agents are faulty, then one mode each".
2. **Some agents are never faulty.** Fault rate is 0.00 for macnet `Node-1` and `Node-2`, llm_debate `User`,
   magentic_one `user`, and smoagents `User` and `System`. These are the prompt or scaffold roles. The agent-name
   one-hot already lets the flat MLP learn this.
3. **The number of faulty agents is nearly fixed per framework.**
   - smoagents: all 360 trajectories have 4 agents present and exactly 1 faulty.
   - magentic_one: mean 1.00 faulty agents.
   - llm_debate: 2 faulty of the 3 assistants in 1,564 of 1,816 trajectories.
   - agentverse: 3 of 4 faulty in 1,643 of 1,821 (all 4 in 164).
   - macnet: mean 2.22 faulty agents, mostly 2 or 3.
   - dylan varies (2.61 mean), because it mixes two team layouts: Assistant1-4, and a code team (PythonAssistant,
     Tester, QualityManager, Debugger, ComputerScientist).
4. **Every labelled agent appears in the turns** (0 of 16,995 labelled pairs name an absent agent in train).
5. **Modes are close to balanced.** Within a benchmark the top mode is 10-12% of labels, and the top modes differ by
   benchmark only slightly (for example FM-3.2 and FM-1.5 lead HumanEval, FM-2.3 leads MATH). Benchmark and framework
   tell you little about the mode; the content must.
6. Only 121 consecutive turns are exact duplicates, so duplicate-turn handling is not a concern.

## What the faults look like in the text (from the 18 read trajectories)

- Some injected faults are visible when a turn is compared with the query or with the other agents:
  - magentic_one: the orchestrator's last turn is an unrelated "design a futuristic city" text (FM-1.1).
  - agentverse HumanEval: the evaluator rewrites the task as "randomly shuffled", "reversed", "disordered" instead of
    "ordered" (FM-2.5 and neighbours).
  - dylan GSM8K: one assistant answers "approximately 2800" against peers who compute exact values.
  - macnet: node feedback that says "shift our focus entirely" away from the question.
- Others are not visible from the first turns: dylan examples labelled FM-2.6 and FM-2.2 on an assistant whose text
  looks like a correct derivation. Some faults may only show in later turns, or not at all, which would set a ceiling
  on any text-based model.
- llm_debate has a `User` turn between every agent turn that just re-quotes the question or the other agents' answers.
  smoagents turns are tool-call and observation wrappers with long boilerplate; the useful content is a small part.

## Consequences for the design

1. **Decode with a per-framework agent budget instead of a single threshold.** The faulty-agent count is almost
   constant per framework, so "take the top-k agents by score, k from train" should beat the current
   `p_agent >= t_a` rule, especially for smoagents and magentic_one (k = 1). Worth testing on the flat MLP first;
   it is cheap and does not need the graph.
2. **Never predict the scaffold roles** (list in item 2) and never predict an agent absent from the turns.
3. **Mean pooling per agent throws away the comparison signal.** Several visible faults are about disagreement with the
   query or peers. A graph or cross-turn feature that sees the query and the other agents' turns could pick this up.
   A cheap probe before building the GNN: add cosine similarity of each agent's turns to the query embedding and to the
   mean of the other agents' turns as extra flat-MLP features. If that helps, it supports the relational design.
4. **Mode prediction rests on content only** (balanced, weakly tied to benchmark), so expect mode macro-F1 to stay
   close to the floor unless the encoder picks up the injection style.
5. Report results per framework, because the label structure differs a lot between them.
