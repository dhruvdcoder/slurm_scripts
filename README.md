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

These scripts are lightweight python wrappers that are designed to work with models that are written to work with the [xlm-core](https://github.com/dhruvdcoder/xlm-core) framework.
The goal of these scripts is to write slurm sbatch scripts store them and submit a job to the slurm scheduler.

## Training

The base training config is located in `slurm/train_sbatch`. It has two main sections `slurm.` and `train.`. The `slurm.` section contains SLURM settings for your job, e.g. number of nodes, GPUs per node, etc. The `train.` section contains training settings for your job, e.g. batch size, precision, etc. which are passed into the `xlm job_type=train` command.
You can pass additional arguments (the ones that are not exposed by `train.` of train_sbatch.yaml) to the inner `xlm job_type=train` command by adding them after `---` in the command line.
Having `---` is optional.

> [!NOTE]
> Make sure to use login node to submit the job and not the terminal of your editor which might be running on a compute node.

> [!NOTE]
> Make sure that you have activated the virtual environment in the terminal from which you are submitting the job before you submit the job.

### Example 1: Basic usage with additional overrides
Running this 
```bash
python slurm_scripts/submit_train.py \
"do=submit" \
"job_name=star_easy_idlm" \
"train.experiment=star_easy_idlm" \
"train.batch_size=64" \
"train.compile=false" \
"train.compile=true" \
"train.precision=bf16-mixed" \
"hardware=1_node_1_gpu" \
"slurm.constraint=\"vram80,bf16,ib\"" \
"++slurm.exclude=gpu016" \
"++use_job_name_as_id=false" \
--- \
"trainer.max_steps=1000"
```

will generate the following sbatch script. If you have not made changes to the default path settings then you will find the generated `sbatch.sh` in `logs/star_easy_idlm/sbatch/<datetime>/sbatch.sh`. The slurm logs will be stored in `logs/star_easy_idlm/sbatch/<datetime>/<job_name>.out` as shown in the sbatch script below.
```bash
#!/bin/sh

#SBATCH --constraint          vram80,bf16,ib
#SBATCH --cpus-per-task       5
#SBATCH --exclude             gpu016
#SBATCH --gres                gpu:1
#SBATCH --job-name            star_easy_idlm
#SBATCH --mail-type           BEGIN,END,FAIL,REQUEUE,TIME_LIMIT_80
#SBATCH --mail-user           dhruveshpate_umass_edu
#SBATCH --mem                 20GB
#SBATCH --nodes               1
#SBATCH --ntasks-per-node     1
#SBATCH --open-mode           append
#SBATCH --output              logs/star_easy_idlm/sbatch/2026-02-22_12-45-01/%x.out
#SBATCH --partition           gpu,gpu-preempt,superpod-a100
#SBATCH --requeue             
#SBATCH --time                1-00:00:00

export HYDRA_FULL_ERROR=1
export NCCL_NSOCKS_PERTHREAD=4
export NCCL_SOCKET_NTHREADS=2
export TORCH_LOGS=recompiles
export TQDM_MINITERS=1000
srun xlm job_name=star_easy_idlm job_type=train experiment=star_easy_idlm loggers=wandb per_device_batch_size=64 trainer_strategy=single_device trainer.devices=1 trainer.num_nodes=1 ++trainer.precision=bf16-mixed compile=True trainer.max_steps=1000
```

### Example 2: Just print the sbatch script
You can just print the sbatch script by changing `do=submit` to `do=print` in the command line.

## Evaluation

The base config is `slurm/eval_sbatch.yaml` with `eval.` and `slurm.` sections. Arguments after `---` are passed to the inner `xlm job_type=eval` command.
Key outer args: `job_name`, `experiment`, `eval.checkpoint_path`, `eval.split`, `eval.dms_to_remove`.

### Example 1: Standard MLM eval (e.g. sudoku with predictor params)

```bash
python lib/slurm_scripts/submit_eval.py \
  "do=submit" \
  "job_name=sudoku_extreme_mlm_eval_top_prob_0.5" \
  "experiment=sudoku_extreme_mlm" \
  "slurm.constraint=vram80" \
  "slurm.partition=\"gpu,gpu-preempt,superpod-a100\"" \
  "eval.split=validation" \
  "eval.dms_to_remove=[val.lm]" \
  "eval.checkpoint_path=logs/sudoku_extreme_mlm/checkpoints/10-80000.ckpt" \
  "paths.log_dir=logs/eval" \
  --- \
  ++predictor.confidence=top_prob \
  predictor.max_steps=81 \
  ++predictor.top_k=1 \
  ++predictor.top_p=null \
  ++predictor.threshold=0.5
```

### Example 2: NLL eval (generative perplexity, unconditional prediction)

For NLL eval with a generative-perplexity evaluator (e.g. gpt2-large), use a list for `experiment` where the second element is a config name like `gpt2_generative_perplexity`.

```bash
python lib/slurm_scripts/submit_eval.py \
  "do=submit" \
  "job_name=lm1b_pgdd_mlm_gpt2_gp_1-100000_nll" \
  "experiment=[lm1b_pgdd_mlm,gpt2_generative_perplexity]" \
  "eval.checkpoint_path=logs/lm1b_pgdd_mlm/checkpoints/1-100000.ckpt" \
  "eval.dms_to_remove=[val.lm]" \
  "paths.log_dir=logs/eval" \
  --- \
  per_device_batch_size=64 \
  global_batch_size=64 \
  +tags.eval_type=nll \
  +tags.checkpoint=1-100000 \
  datamodule.dataset_managers.val.unconditional_prediction.num_examples=1000
```

### Common INNER_ARGS quick reference

- **Predictor (MLM)**: `++predictor.confidence=top_prob` `++predictor.threshold=0.5` `++predictor.top_k=1` `++predictor.top_p=null` `predictor.max_steps=81`
- **Batch size**: `per_device_batch_size=64` `global_batch_size=64`
- **Wandb tags**: `+tags.eval_type=nll` `+tags.checkpoint=1-100000`
- **Unconditional prediction**: `datamodule.dataset_managers.val.unconditional_prediction.num_examples=1000`
- **Misc**: `trainer.limit_val_batches=null` `paths.log_dir=logs/eval`

### dms_to_remove patterns

- Standard eval (skip LM metrics): `eval.dms_to_remove=[val.lm]` or `[val.lm, test.lm]` (default)
- NLL infill (skip unconditional pred): `eval.dms_to_remove=[val.unconditional_prediction]`
