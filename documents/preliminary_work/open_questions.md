# Open Questions

These questions are intentionally unresolved. They should guide the next research and design passes rather than be answered through premature implementation choices.

## Model architecture

1. What ratio and ordering of plain DeltaNet and Gated DeltaNet layers best separates persistent memory from adaptive working memory?
2. Can a DeltaNet-family-only stack handle ordinary short- and medium-term prediction well enough, or does evidence justify adding Mamba2?
3. How much recurrent state is required before capacity and address interference stop dominating the generator task?
4. How should persistent state be reset, continued, serialized, batched, and restored?
5. What training lengths are sufficient to teach behavior that extrapolates to much longer inference streams?
6. How should the experiment detect gradient contraction or effective memory loss across repeatedly addressed directions?

## Generator

1. What is the smallest exact definition of the hashed controller?
2. How many hash channels, bins, and controller states produce useful collision rates for a normal vocabulary?
3. How does controller state define its candidate set without smoothing the distribution into near-uniform noise?
4. How should the sparse local filter share structure with the controller so their intersection remains nonempty?
5. Can a sparse bigram or trigram graph provide enough grammar-like and collocational structure for the first experiment?
6. Does the local system become powerful enough that the controller is unnecessary?
7. Which generator parameters should be resampled per episode, and which encoding conventions should remain stable across training?
8. How can process complexity be measured before investing in model training?

## Training

1. What first task clearly requires online process identification rather than local memorization?
2. Should each training example contain one long process, multiple process changes, or both?
3. How should true rule changes be distinguished from stochastic exceptions?
4. What curriculum, if any, is needed across controller complexity, local-filter sparsity, and episode length?
5. How can training prevent the model from relying only on the gated path or only on short-context cues?
6. What is the cheapest experiment that can disprove the current generator or architecture direction?

## Evaluation

1. Does prediction improve as more evidence from one fixed process is observed?
2. Does performance depend on state carried from evidence outside the model's recent local context?
3. Can the model retain several independent inferred rules without catastrophic interference?
4. Can it revise one rule after a genuine change without globally erasing unrelated knowledge?
5. Does the learned procedure transfer to held-out generator parameters and held-out generator families?
6. What evidence would distinguish semantic or structural transfer from simple token-frequency adaptation in an unseen-language test?

## Repository and engineering

1. Which FLA model classes provide the best reference for mixed recurrent-state handling?
2. Can the custom model remain compatible with Hugging Face generation and checkpoint conventions without obscuring its state contracts?
3. What minimal tests are required before any scientific run is trustworthy?
4. What hardware and sequence lengths can support the initial experiment without designing around an unrealistic training regime?
