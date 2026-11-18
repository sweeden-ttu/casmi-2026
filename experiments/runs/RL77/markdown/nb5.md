<!-- intro -->
# CASMI 2026 · RL77 Notebook 5 — Q-Learning / RL Reward Shaping for the ChemFormer Decoder

*Assembled from **CASMI 2026 Colab Rubric.pdf** (pages 48–59). Python cells are the reviewed `experiments/runs/RL77/nb5/cell_NN.py` files; text cells are taken from the PDF.*

> **User prompt:** Write a PyTorch Q-learning / RL reward-shaping wrapper for the ChemFormer decoder that optimizes generated SMILES toward exact precursor mass and Mass2SMILES functional group consistency.

Standard teacher-forced cross-entropy ($L_{gen}$) trains the **ChemFormer** decoder to maximize token-level likelihood, which frequently produces syntactically valid SMILES that drift by a methyl or hydroxyl group from the true precursor mass or omit functional groups evidenced by MS/MS neutral losses. Wrapping the Stage 2 decoder in a **Token-Level Dueling Soft Q-Learning and Potential-Based Reward Shaping** module (`ChemFormerQDecoderWrapper`) directly optimizes generation toward exact monoisotopic precursor mass, **Mass2SMILES** 71-D functional group counts, and **MIST-CF** molecular formula priors while anchoring to the supervised Stage 2 policy via a Kullback-Leibler (KL) divergence penalty to prevent mode collapse.

<!-- before:cell_01 -->
## Cell 1: Dense Prefix Potential & Terminal Chemical Reward Engine

To overcome the sparse-reward problem (where RDKit can only parse a SMILES string after the `[EOS]` token is emitted), `CASMI26ChemicalRewardEngine` pairs a **dense heavy-atom mass potential** $\Phi(s_t)$ at every token step $t$ with a **multi-objective terminal reward** $R_{\text{term}}(y)$ at `[EOS]`. By Ng et al.'s potential-based reward shaping theorem, intermediate shaping rewards $F(s_t, s_{t+1}) = \gamma\Phi(s_{t+1}) - \Phi(s_t)$ accelerate Q-value convergence without altering the optimal chemical policy.

> **Review fix:** the notebook-1 tokenizer is character-level, so the `"Cl"` / `"Br"` keys never matched: chlorine counted as 12 Da (its `C`) and bromine as 0. Per-character increments (`l`, `B`, `r`) keep the heavy-atom lower bound exact, and explicit `H` is counted. All reward and potential coefficients come from the Evidently weights (`nb5.reward.*`, `nb5.potential.*`).

<!-- before:cell_02 -->
## Cell 2: Dueling Soft Q-Learning Wrapper (`ChemFormerQDecoderWrapper`)

To make the decoder explicitly mass-aware at every generation step, `ChemFormerQDecoderWrapper` extracts the causal Transformer hidden state $\mathbf{h}_t \in \mathbb{R}^{256}$, concatenates an encoding of the **remaining precursor mass budget** $(m_{\text{target}} - m_{\text{heavy}}(s_t))$, and passes the fused state through a **Dueling Q-Network** $\left(Q(s_t, a_t) = V(s_t) + A(s_t, a_t) - \overline{A}(s_t)\right)$.

<!-- before:cell_03 -->
## Cell 3: Stage 2.5 Hybrid Soft Q-Learning + KL-Anchored Actor-Critic Trainer

Training an autoregressive chemical language model purely with policy gradients or unconstrained Q-learning quickly collapses syntactic SMILES validity. This trainer combines three complementary objectives:

1. **Shaped Soft Bellman TD Loss ($\mathcal{L}_Q$):** Trains $Q_\theta(s_t, a_t)$ on both teacher-forced ground-truth trajectories and on-policy sampled trajectories using dense potential shaping $r_t = \gamma\Phi(s_{t+1}) - \Phi(s_t)$ for $t < T$ and $r_T = R_{\text{term}}(y) - \Phi(s_T)$ at `[EOS]`.
2. **Conservative Q-Regularization ($\mathcal{L}_{\text{CQL}}$):** Penalizes out-of-distribution chemical token actions so unvisited invalid SMILES transitions do not receive overestimated Q-values.
3. **KL-Anchored Advantage Actor Loss ($\mathcal{L}_\pi$):** Updates the `ChemFormerMolecularDecoder` weights toward high-advantage mass- and functional-group-consistent tokens while penalizing divergence $\mathrm{KL}(\pi_\phi \,\|\, \pi_{\text{ref}})$ from the frozen Stage 2 checkpoint.

> **Review fix:** the bootstrap target $Q_{\text{targ}}(s_{t+1}, \cdot)$ was computed by feeding `act_ids` (the trajectory without `[BOS]`) as a sequence, which shifted every position embedding and dropped the BOS context; it is now the full-trajectory target-network output at position $t+1$. Neutral mass uses `neutral_mass_from_precursor()`, and γ, α, CQL, KL, anchor, Polyak, rollout temperature, and the expert reward come from the Evidently weights.

> **Review fix (after the nb5 Kaggle run):** the actor's advantage was the critic's $Q - V$. While the dueling heads are immature, that signal is uncorrelated with the actual outcome. In the first full run the policy drifted and rollouts collapsed to invalid SMILES: the mean terminal reward went from −0.28 (best −0.11 around step 250) to −0.96 by step 4,885, and Q-guided beams were 49% valid with 0% inside the mass window. The actor now uses REINFORCE with a batch baseline, $\hat A = (R(y) - \bar R)/\sigma_R$ shared by every token of $y$, so an all-invalid batch gives no push. The Q-heads still learn by the shaped soft Bellman TD loss and still guide beam search.

<!-- before:cell_04 -->
## Cell 4: Mass-Constrained Q-Guided Beam Search

During inference on `test.parquet`, `q_guided_beam_search_decode` replaces standard log-probability beam search. At every step $t$, candidate token extensions are scored by the joint **Actor-Critic objective**:

$$\text{Score}(s_t, a_t) = \log \pi_\phi(a_t \mid s_t, \mathbf{H}^{ca}) + \beta_Q\, Q_\theta(s_t, a_t, m_{\text{rem}})$$

Any partial beam whose accumulated heavy-atom mass exceeds `target_neutral_mass + 2.0 Da` is immediately pruned, and all completed beams are re-ranked by `CASMI26ChemicalRewardEngine` so the top valid de novo isomers can both anchor GLMR Stage 2 unimodal similarity and populate novel scaffold slots (Tier 3).

> **Review fix:** beam width, β, pruning margin, and the reward mix come from the Evidently weights, and per-beam rewards are returned for the centroid below. The chrysin/paracetamol smoke test is replaced by the real run in Cell 5.

<!-- before:cell_05 -->
## Cell 5: Stage 2.5 Training on the Competition Data, Q-Guided Validation, Test Submission

The full run: fine-tune notebook 1's decoder with the Stage 2.5 trainer on notebook 1's training split within a GPU time budget, then compare on the same held-out timsTOF molecules notebook 1's pipeline (with its adopted re-rank weights) against Q-guided beams with the reward-weighted centroid below (MRR@25, validity, mass agreement, exact top-1). Candidates matching notebook 4's top formula priors rank first when notebook 4 adopted formula tiers (`nb4.priors.rank_formulas` > 0). The centroid blend α and the Q-beam insert rank are notebook 5's own weights (`nb5.rerank.*`), cross-fitted on the held-out molecules and written to `fitted_weights.json`; the test submission uses the fit only when it beats the prior on the held-out halves (the rule `tools/evidently_weights.py adopt` applies). A best-checkpoint guard keeps the decoder and Q-heads with the highest 100-step moving-average rollout reward and restores them before validation. Outputs become the `casmi26-rl77-nb5-qrl-outputs` dataset.

> **Review fix (after the notebook-1 Kaggle run):** this cell first reused `nb1.rerank.alpha`. Notebook 1's cross-fit set that to 0 because plain beam search hurt MRR@25, and α = 0 would silently remove the Q-centroid from the ranking, so notebook 5 now has its own keys.

Instead of conditioning Stage 2 unimodal similarity on a single unconstrained beam sequence, average the `ChemFormerMolecularEncoder` embeddings of the top-3 mass-verified Q-beam structures:

$$\mathbf{E}^m_+ = \text{Normalize}\left(\sum_{j=1}^{3} w_j\, f_m(y_j^+; \gamma)\right), \qquad w_j = \frac{\exp(R_{\text{term}}(y_j^+))}{\sum_{k=1}^{3}\exp(R_{\text{term}}(y_k^+))}$$

This reward-weighted centroid sharpens $\cos(\mathbf{E}^m_+, \mathbf{E}^m_i)$ for known database isomers (Tier 2) while supplying chemically valid, mass-exact de novo candidates for novel structures (Tier 3).

<!-- end -->
## How the Q-Learning Wrapper Elevates Stage 2 Re-Ranking & Tier 3 Novel Scaffolds

| Decoding Mechanism | Syntactic Validity | Exact Mass Match (within 15 ppm) | Functional Group Consistency | Impact on CASMI 2026 `MRR@25` |
| --- | --- | --- | --- | --- |
| **Standard Beam Search ($L_{gen}$ only)** | ~95% | 28% to 36% | Unconstrained | Generated SMILES $\mathbf{E}^m_+$ often drifts in molecular weight, adding noise to unimodal re-ranking. |
| **Q-Guided Beam Search ($L_{gen}$ + Dueling Q-RL)** | ~94% | **68% to 79%** | Enforced via 71-D FG reward | Produces multiple distinct `inchikey14` isomers matching exact precursor mass and neutral losses. |

*The table reproduces the PDF's expected ranges; `metrics.json` (`val.q_beam_*`, `val.mrr25_*`) holds what this run measured.*
