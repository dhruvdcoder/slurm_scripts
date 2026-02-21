# Setup

Clone as a submodule in the root directory.
```bash
git submodule add https://github.com/dhruvdcoder/slurm_scripts.git slurm_scripts
```

Install the requirements.
```bash
pip install -r slurm_scripts/requirements.txt
```

# Usage Examples

```bash
python slurm_scripts/submit_train.py "do=print" "job_name=owt_idlm" "train.experiment=owt_idlm" "train.batch_size=32" "train.compile=false" "train.precision=bf16-mixed" "hardware=1_node_1_gpu" "slurm.constraint=\"vram80,bf16,ib\"" "++slurm.exclude=gpu016"
```