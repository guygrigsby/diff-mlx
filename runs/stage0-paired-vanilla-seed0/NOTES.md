# Stage 0 paired vanilla seed 0 — run notes

## Run summary
- **Variant:** vanilla MHA (control arm of the paired A/B)
- **Init source:** paired-seed init protocol (design §9.7) — `runs/init-seed0/vanilla.safetensors`
- **Config:** ModelConfig.stage0() + TrainConfig.stage0()
- **Total steps:** 6,103 (target 6,103 = 100M tokens / 16k tokens-per-step)
- **Wall time:** 70.6 min
- **Final tps:** 23,613 tokens/sec
- **NaN/Inf:** none

## Final state
- step 6000 train_loss: 4.6707
- step 6000 val_monitor: 4.6707
- step 5000 val_full: 4.7200 (perplexity ≈ 112.2)

## Comparison to Phase A vanilla
Identical within noise. The paired-seed init path adds no functional difference on the vanilla side — it's the same initialization, just routed through `paired_init.build_paired_models` + saved to a state-dict + loaded. Confirms the protocol's vanilla half is a clean pass-through.

## RoPE convention
This run used `mx.fast.rope(traditional=True)` (paper-canonical interleaved). Different from Phase A's vanilla which used `traditional=False` (LLaMA rotate-halves; obsolete after design round 7). Comparable losses across the two indicate RoPE convention doesn't materially affect vanilla MHA training at this scale.
