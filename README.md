## Semantic Dependency Parsing is More Parameter-Efficient with Normalization

This repo hosts the code needed to reproduce the results for the NeuRIPS 2025 submission "Semantic Dependency Parsing is More Parameter-Efficient with Normalization".

The repo can be installed with pip:
```bash
pip install -e .
pip install -r requirements.txt
```
although we recommend simply iteratively running ``./src/train.py`` and waiting each time for it to throw a ``ModuleNotFoundError``.

The script ``./src/experiments.sh`` can be used to launch ``./src/train.py`` in parallel for all of the hyperparameter combinations considered in our paper. First, launch the command without ``sbatch``:

```bash
./src/experiments.sh
>> Total combinations: 80
>> This script should be run as a SLURM array job.
>> Use: sbatch --array=0-79 exp_lstm.sh
>> This will distribute 80 jobs across N GPUs.
```

This will print the command you need to execute in order to submit the array of jobs. Note that on your particular cluster you might have a limit on the size of the array or the number of jobs you can run in parallel.