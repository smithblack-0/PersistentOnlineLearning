# Open Questions

These questions are intentionally unresolved. They should guide the next research and design passes rather than be answered through premature implementation choices.

## Model architecture

1. What ratio and ordering of plain DeltaNet and Gated DeltaNet layers best separates persistent memory from adaptive working memory?
2. Can a DeltaNet-family-only stack handle ordinary short- and medium-term prediction well enough, or does evidence justify adding Mamba2?
3. How much recurrent state is required before capacity and address interference stop dominating the generator task?
4. How should persistent state be reset, continued, serialized, batched, and restored?
5. What training lengths are sufficient to teach behavior that extrapolates to much longer inference streams?
6. How should the experiment detect gradient contraction or effective memory loss across repeatedly addressed directions?

## Synthetic language construction

1. Which exact and range controls are needed first around the literature CFG constructor?
2. What distributions over rule families, symbol maxima, and grammar sizes produce useful rather than merely valid languages?
3. Should a later choice policy compensate for the constructor's age bias, and what evidence would justify doing so?
4. How should lexical-category sizes and overlap be sampled while guaranteeing complete vocabulary coverage?
5. How should ordinary production probabilities be drawn so recursive grammars terminate often enough while retaining useful structural ambiguity?
6. How different should state-conditioned production distributions be across epsilon states?
7. What epsilon-machine transition families remain learnable as state count grows?
8. Which properties should be resampled per language, and which encoding conventions should remain stable across training?

## Training

1. What first curriculum tier clearly requires online grammar identification rather than only local token statistics?
2. How many sentences or tokens from one generated language are needed at each grammar and epsilon-state complexity?
3. Should multiple sentences begin from the same epsilon state to make early latent-state identification tractable?
4. How should language changes and recurrent-state resets be represented in a batch?
5. How can training prevent the model from relying only on the gated path or only on short-context cues?
6. What is the cheapest experiment that can disprove the current generator or architecture direction?

## Evaluation

1. Does prediction improve as more evidence from one fixed synthetic language is observed?
2. Does performance depend on state carried from evidence outside the model's recent local context?
3. How do unigram, bigram, trigram, and four-gram online controls compare under the same stream?
4. Can the model retain several independent inferred rules without catastrophic interference?
5. Can it revise one rule after a genuine change without globally erasing unrelated knowledge?
6. Does the learned procedure transfer to held-out grammar and epsilon-machine parameters?
7. What evidence would distinguish structural transfer from simple token-frequency adaptation in a stemmed-English test?

## Repository and engineering

1. Which configuration fields become necessary as each generator unit is implemented, without prematurely finalizing a project-wide schema?
2. Can the exact CFG constructor remain independently auditable as range selection and lexical realization are added around it?
3. What minimal tests are required before any scientific run is trustworthy?
4. What hardware and sequence lengths can support the initial experiment without designing around an unrealistic training regime?
