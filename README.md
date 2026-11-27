# PoseEvidence

![PoseEvidence agent architecture](docs/architecture.svg)

PoseEvidence is a training-free scientific agent for protein–ligand structure prediction with frozen Boltz-2. We built it around a question: **can scientific evidence improve what the folding model predicts in the first place?**

[Research presentation and interactive case study](https://losewords.github.io/pose-evidence/)

## Design rationale

**Start with the limits of local refinement.** Our first prototype predicted a structure and then asked the agent to inspect and, when warranted, revise it. A short contact made the limits of that approach concrete. Distance can flag a geometry problem, while chemical identity and orientation are needed to interpret it. A local repair can improve geometry while leaving the accuracy of the binding pose unresolved.

**Use evidence to guide the next prediction.** Boltz-2 accepts explicit binding-site conditions without changing its weights. The agent retrieves literature and related structures, checks how reference residues map to the target sequence, and decides whether the evidence warrants a `pocket` constraint for a region or a `contact` constraint for a specific pair. Each condition makes a structural hypothesis testable through a new prediction.

**Measure, update, and decide.** After prediction, tools compare candidate poses and measure contact geometry and stereochemistry. The research board records these observations separately from the agent’s interpretations. The agent uses them to revise a hypothesis, plan another round, or deliver a structure with its limitations.

We evaluate experimental agreement offline using ligand RMSD and lDDT-PLI. The target experimental structure is held out from the agent’s decisions. Model confidence and physical checks describe different properties of a candidate.

## Run locally

The prediction worker targets Linux with an NVIDIA GPU. The code was exercised with Python 3.10 and Boltz 2.2.1. Follow the [Boltz installation guide](https://github.com/jwohlwend/boltz#installation) for your CUDA setup.

```sh
python3.10 -m venv .venv
. .venv/bin/activate
python -m pip install -r requirements.txt
export TETRA_GPU_INDEX=0
```

Set `OPENROUTER_API_KEY` in your environment, or point `OPENROUTER_API_KEY_FILE` at a key file outside this checkout. The default model is `deepseek/deepseek-v4-flash` and `TETRA_LLM_MODEL` overrides it. For native DeepSeek, set `TETRA_LLM_PROVIDER=deepseek` and configure `DEEPSEEK_API_KEY` or `DEEPSEEK_API_KEY_FILE`.

Create a request JSON with an explicit `goal` and `input`. The placeholders below stand for your actual protein sequence, stereochemical ligand SMILES, and a precomputed MSA path.

```json
{
  "goal": "Predict the supplied protein–ligand complex. Use literature evidence to decide whether a binding-site condition is warranted and deliver one primary mmCIF.",
  "input": {
    "sequence": "<protein sequence>",
    "smiles": "<isomeric ligand SMILES>",
    "msa": "/path/to/input.a3m"
  }
}
```

```sh
python -m tetra.scientific_agent --state state-research run request.json --literature --max-steps 24
python -m tetra.scientific_agent --state state-research status SESSION_ID
python -m tetra.scientific_agent --state state-research resume SESSION_ID
```

When the agent reports `waiting_prediction`, let the background worker finish and then resume the same session. The output directory contains the job artifacts, candidate structures, evidence, and versioned memory. The literature route queries public services and may send the input protein sequence to RCSB. If an MSA is omitted, the prediction backend can send the sequence to the configured MSA service.

The standard path uses PoseBusters for physical checks. `TETRA_ASSESS_PYTHON` can select a separate worker interpreter. OpenMM refinement, GNINA, fpocket, and OpenStructure scoring are advanced paths that require local installation and, where enforced by the code, validated tool records. Benchmark runners are separate from the standard agent command. Set `TETRA_GPU_LOCK_DIR` explicitly when sibling checkouts use the same GPUs.

## Code map

| Package | Responsibility |
| --- | --- |
| `tetra/orchestration/` | Agent loops, tool routing, retrieval readiness, and prompts |
| `tetra/research/` | Literature and structure retrieval, mapped evidence, hypotheses, and working memory |
| `tetra/prediction/` | Experiment registration, constraints, Boltz jobs, and GPU locking |
| `tetra/evaluation/` | Reference-free structure checks, pose comparison, refinement, selection, and mmCIF delivery |
| `tetra/benchmarks/` | Offline runners and scoring against experimental structures |
| `tetra/runtime/` | Provider clients, I/O, persistence, and cost accounting |

Existing command-line module names remain entry points. The GPU worker uses lock files in the local `state/` directory, or in the directory selected by `TETRA_GPU_LOCK_DIR`.

## Offline tests

```sh
python -m unittest discover -s tests -q
```

The unit suite uses mocks for model and network calls. Two tests that expect a prior local Boltz run skip when its artifacts are absent.
