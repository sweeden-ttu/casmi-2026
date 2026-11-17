# [Colab Q-RL Cell 4] Mass-Constrained Q-Guided Beam Search & End-to-End Smoke Test
@torch.no_grad()
def q_guided_beam_search_decode(
    q_decoder: ChemFormerQDecoderWrapper,
    reward_engine: CASMI26ChemicalRewardEngine,
    memory: torch.Tensor,                 # (B, P, D) H^ca
    target_neutral_masses: np.ndarray,     # (B,)
    fg_71_preds: np.ndarray,               # (B, 71)
    formula_priors_batch: Optional[List[List[str]]] = None,
    beam_size: int = 6,
    q_weight_beta: float = 0.35,
    max_len: int = 128,
) -> Tuple[List[List[str]], List[List[str]]]:
    """
    Executes Mass-Constrained Q-Guided Beam Search per molecule_id and returns:
      - top_valid_smiles_per_mol: List[List[str]] of canonicalized, mass-consistent de novo SMILES
      - top_valid_ik14_per_mol:   List[List[str]] of corresponding unique InChIKey14 blocks
    """
    q_decoder.eval()
    device = memory.device
    bos_id = q_decoder.base_decoder.bos_id
    eos_id = q_decoder.base_decoder.eos_id
    token_masses = reward_engine.token_masses

    B = memory.size(0)
    all_mol_smiles: List[List[str]] = []
    all_mol_ik14s: List[List[str]] = []

    for b in range(B):
        mem_b = memory[b : b + 1]  # (1, P, D)
        target_m = float(target_neutral_masses[b])
        target_m_t = torch.tensor([target_m], dtype=torch.float32, device=device)

        # Each beam stores: (token_list, cumulative_score, cumulative_heavy_mass)
        beams: List[Tuple[List[int], float, float]] = [([bos_id], 0.0, 0.0)]
        completed_seqs: List[Tuple[List[int], float]] = []

        for _ in range(max_len - 1):
            active_seqs = [seq for seq, _, _ in beams if seq[-1] != eos_id]
            if not active_seqs:
                break

            # Batched evaluation of all active beams for molecule b
            n_act = len(active_seqs)
            inp_t = torch.tensor(active_seqs, dtype=torch.long, device=device)
            mem_exp = mem_b.expand(n_act, -1, -1)
            tm_exp = target_m_t.expand(n_act)

            rem_ratio, _ = reward_engine.compute_prefix_mass_and_potential(inp_t, tm_exp)
            policy_logits, q_vals, _ = q_decoder.forward_policy_and_q(
                inp_t, mem_exp, rem_ratio, tm_exp, use_target_q=False
            )

            last_log_probs = F.log_softmax(policy_logits[:, -1, :], dim=-1)  # (n_act, V)
            last_q_vals = q_vals[:, -1, :]                                   # (n_act, V)
            joint_step_scores = last_log_probs + q_weight_beta * torch.tanh(last_q_vals)

            next_candidates: List[Tuple[List[int], float, float]] = []
            act_ptr = 0

            for seq, cum_score, cum_mass in beams:
                if seq[-1] == eos_id:
                    completed_seqs.append((seq, cum_score / len(seq)))
                    continue

                row_scores = joint_step_scores[act_ptr]
                act_ptr += 1

                topk_sc, topk_tok = torch.topk(row_scores, k=beam_size * 2)
                for sc, tok_id in zip(topk_sc.tolist(), topk_tok.tolist()):
                    new_mass = cum_mass + float(token_masses[tok_id])
                    # Hard physical pruning: heavy atoms alone cannot exceed neutral_mass + 2.0 Da
                    if new_mass > (target_m + 2.0):
                        continue
                    next_candidates.append((seq + [tok_id], cum_score + sc, new_mass))

            if not next_candidates:
                break
            next_candidates.sort(key=lambda x: x[1] / len(x[0]), reverse=True)
            beams = next_candidates[:beam_size]
            if len(completed_seqs) >= beam_size * 2:
                break

        for seq, cum_score, _ in beams:
            completed_seqs.append((seq, cum_score / len(seq)))

        # Re-rank all completed beams using the exact chemical reward engine (Mass + 71-D FG + Formula)
        seq_list = [s for s, _ in completed_seqs]
        beam_rewards, canon_smis, ik14s = reward_engine.evaluate_terminal_smiles_batch(
            generated_seqs=seq_list,
            target_neutral_masses=np.full(len(seq_list), target_m, dtype=np.float32),
            target_fg71_preds=np.repeat(fg_71_preds[b : b + 1], len(seq_list), axis=0),
            formula_priors_batch=[formula_priors_batch[b]] * len(seq_list) if formula_priors_batch else None,
            device=device,
        )

        # Combine normalized beam search trajectory score with terminal chemical reward
        final_beam_Order = np.argsort(
            -(beam_rewards.cpu().numpy() + 0.25 * np.array([sc for _, sc in completed_seqs]))
        )

        seen_ik14 = set()
        mol_valid_smis, mol_valid_iks = [], []
        for idx in final_beam_Order:
            c_smi, ik = canon_smis[idx], ik14s[idx]
            if c_smi is not None and ik is not None and ik not in seen_ik14:
                seen_ik14.add(ik)
                mol_valid_smis.append(c_smi)
                mol_valid_iks.append(ik)

        all_mol_smiles.append(mol_valid_smis)
        all_mol_ik14s.append(mol_valid_iks)

    return all_mol_smiles, all_mol_ik14s


if __name__ == "__main__":
    # Self-contained verification smoke test of the Q-Learning Decoder Wrapper & Reward Engine
    # (Assumes SMILESTokenizer, ChemFormerMolecularDecoder, canonicalize_and_inchikey14,
    #  and smiles_to_ecfp4_and_mass2smiles_71d from previous cells are in scope)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = SMILESTokenizer(max_len=64)

    base_dec = ChemFormerMolecularDecoder(
        vocab_size=len(tokenizer),
        pad_id=tokenizer.pad_id,
        bos_id=tokenizer.bos_id,
        eos_id=tokenizer.eos_id,
        d_model=256,
        n_heads=8,
        num_layers=2,
        max_len=64,
    ).to(device)

    q_wrapper = ChemFormerQDecoderWrapper(base_dec, d_model=256).to(device)
    reward_eng = CASMI26ChemicalRewardEngine(
        tokenizer=tokenizer,
        canonicalize_fn=canonicalize_and_inchikey14,
        fg_extractor_fn=smiles_to_ecfp4_and_mass2smiles_71d,
    )

    # Test terminal reward discrimination between Chrysin (254.0579 Da) and an off-mass structure
    chrysin_smi = "O=c1cc(-c2ccccc2)oc2cc(O)cc(O)c12"
    off_mass_smi = "CC(=O)Nc1ccc(O)cc1"  # Paracetamol (151.0633 Da)
    _, target_fg71 = smiles_to_ecfp4_and_mass2smiles_71d(chrysin_smi)

    seqs_test = [
        tokenizer.encode(chrysin_smi, add_cls=False).tolist(),
        tokenizer.encode(off_mass_smi, add_cls=False).tolist(),
    ]
    r_vals, c_smis, iks = reward_eng.evaluate_terminal_smiles_batch(
        generated_seqs=seqs_test,
        target_neutral_masses=np.array([254.0579, 254.0579], dtype=np.float32),
        target_fg71_preds=np.stack([target_fg71, target_fg71], axis=0),
        formula_priors_batch=[["C15H10O4"], ["C15H10O4"]],
        device=device,
    )
    print(f"Reward [Exact Mass + FG Match (Chrysin)]: {r_vals[0].item():.4f} | InChIKey14: {iks[0]}")
    print(f"Reward [Off-Mass Structure (Paracetamol)]: {r_vals[1].item():.4f} | InChIKey14: {iks[1]}")