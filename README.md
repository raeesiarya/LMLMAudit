<h2 align="center">LMLM Audit: Auditing Forgetting in Limited Memory Language Models</h2>

<p align="center">
  <strong>Project repository for auditing whether factual knowledge in Limited Memory Language Models is truly externalized and removable.</strong>
</p>

<p align="center">
  Hanna Roed, Arya Raeesi, Rohan Bijukumar<br>
  <a href="mailto:hanna.roed@berkeley.edu">hanna.roed@berkeley.edu</a>,
  <a href="mailto:aryaraeesi@berkeley.edu">aryaraeesi@berkeley.edu</a>,
  <a href="mailto:rohanbijukumar@berkeley.edu">rohanbijukumar@berkeley.edu</a>
</p>

<p align="center">
</p>

<p align="center">
  <img src="./badges/coverage.svg" alt="Coverage badge" />
</p>

<p align="center">
  <a href="./docs/Auditing_Forgetting_in_Limited_Memory_Language_Models.pdf">Project Proposal</a>
  |
  <a href="https://arxiv.org/abs/2505.15962">LMLM Paper</a>
  |
  <a href="https://github.com/kilian-group/LMLM">Original LMLM Repository</a>
</p>

## Overview

Limited Memory Language Models (LMLMs) aim to separate language ability from factual knowledge by storing facts in an external database instead of only in model parameters. During pretraining, factual values are masked from the loss so the model learns when to issue structured lookups rather than memorizing every fact internally.

This repository focuses on a narrower and more diagnostic question:

**When a fact is deleted from the LMLM database, has the model actually forgotten it?**

That question is more subtle than measuring whether the model still answers correctly after deletion. A correct answer can come from at least three different sources:

- residual parametric memory
- alternative retrieval paths in the database
- approximate or semantically related retrieval matches

The goal of this project is to disentangle those mechanisms and provide a clean audit of forgetting in LMLMs.

This is an audit project built on top of the LMLM framework, not the upstream LMLM training repository itself.

## Core Idea

We evaluate each target fact under three intervention settings:

| Setting | Database state | Retrieval | What it measures |
| --- | --- | --- | --- |
| `FULL` | intact | enabled | baseline factual access |
| `DEL-ON` | target fact deleted | enabled | post-deletion behavior with retrieval still available |
| `DEL-OFF` | target fact deleted | disabled | residual parametric recall without retrieval |

This intervention set lets us ask whether post-deletion correctness comes from the model's parameters, from the external memory, or from retrieval artifacts.

We use the following quantities to decompose forgetting behavior:

- `L(f) = 1[Y(f, DEL-OFF) = o]`
  Parametric leakage: the model still produces the gold object even when retrieval is disabled.
- `R(f) = 1[Y(f, DEL-ON) = o and Y(f, DEL-OFF) != o]`
  Retrieval-mediated correctness: the model remains correct only when retrieval is enabled.
- Retrieval artifacts
  Cases where `DEL-ON` is correct even though no gold-equivalent retrieved entry appears in the inference trace.

Together, these metrics distinguish true forgetting from apparent forgetting.

## Project Goals

- Audit whether deletion-based unlearning in LMLMs removes factual access or only changes how the fact is recovered.
- Measure how often deleted facts remain accessible through internal memory alone.
- Attribute post-deletion correctness to explicit retrieved evidence versus implicit model behavior.
- Study how leakage varies across relation type, entity popularity, and prompt formulation.
- Build a reproducible evaluation framework for memory separation in modular and retrieval-augmented language models.

## Proposed Workflow

The project proposal centers on the following pipeline:

1. Draw evaluation facts directly from the released LMLM database so training, retrieval, and evaluation stay aligned.
2. Apply verified alias-closure deletion for each target fact, removing canonical and alias-equivalent realizations of `(subject, relation, object)`.
3. Run inference under `FULL`, `DEL-ON`, and `DEL-OFF`.
4. Normalize outputs against canonical entity forms so evaluation is deterministic.
5. Log retrieval traces during inference to determine whether correctness is supported by explicit database evidence.
6. Aggregate leakage, retrieval-mediated correctness, and artifact rates across fact categories.

## Why This Matters

Architectures like LMLM are appealing because they promise editable and removable knowledge without retraining. If that promise holds, they could support more reliable factual updates, deletion requests, and governance workflows.

But if a deleted fact still survives in model parameters, then database deletion alone may not constitute true forgetting. This project is designed to test that boundary directly.

## References

- Linxi Zhao et al. *Pre-training Limited Memory Language Models with Internal and External Knowledge*. 2025.
- Kevin Meng et al. *Locating and Editing Factual Associations in GPT*. NeurIPS 2022.
- Nicholas Carlini et al. *Extracting Training Data from Large Language Models*. USENIX Security 2021.
- Patrick Lewis et al. *Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks*. NeurIPS 2020.
- Kelvin Guu et al. *REALM: Retrieval-Augmented Language Model Pre-Training*. ICML 2020.

## Acknowledgments

This project builds on the LMLM framework and the public release from the original authors. Their work provides both the motivation and the foundation for this audit.

## License

This project is licensed under the MIT License. See [LICENSE](./LICENSE) for details.
