# Component-state endpoint pilot

This directory is a controlled frozen-host study, not a reproduction of all
historical ASCC training. Read `PROTOCOL.md` before interpreting results.

The environment needs Python 3.11, PyTorch 2.6 and NumPy. Set `GROUP_OPT_ROOT`
to a project checkout containing the official repositories and checkpoints:

```
repos/groupopt-modern-hosts/ICAM/ICAM_TSP/TSPModel_ICAM.py
repos/groupopt-modern-hosts/ICAM/pretrained/icam_tsp.pt
repos/groupopt-official-am-kool/nets/attention_model.py
repos/groupopt-official-am-kool/pretrained/tsp_100/{args.json,epoch-99.pt}
```

Run from this directory (substitute a new output directory for each host):

```bash
python run_pilot.py smoke --host icam --out smoke.json
python run_pilot.py prepare --host icam --out results_icam
CUDA_VISIBLE_DEVICES=0 python run_pilot.py train --host icam --out results_icam --mode route
CUDA_VISIBLE_DEVICES=1 python run_pilot.py train --host icam --out results_icam --mode random
python run_pilot.py finalize --host icam --out results_icam
python summarize_pilot.py --out results_icam
```

The two training commands may run concurrently. Finalize refuses to run until
all 12 arms finish; selected-checkpoint hashes are frozen before test creation.
It also refuses to overwrite an existing final test. Repeat with `--host am`
and a fresh directory for the prespecified second-host replication.

`component_adapter.py` contains the proposed residual. `strong_adapter.py`
is the existing audited official-host wrapper with a portable root setting.
`run_strong.py` supplies only deterministic 2-opt teacher improvement and
`verify_strong.py` supplies Hamiltonian-tour validation. Host encoders and
decoders are frozen; only the residual is trained. Actual parameter counts,
source/checkpoint/data hashes, complete curves and per-instance costs are
recorded. Reported evaluation times cover decoding from cached embeddings;
the final-test manifest separately records encoding time. They are pilot wall
times with concurrent jobs, not isolated hardware benchmarking results.
