# Galaxy_Zoo

Project of Computer Vision 2018 NYU course.

## To run on HPC cluster

1. Create conda environment, but you may not be supposed to run it with single core cpu.


```bash
conda env create -f requirements.yaml
```

2. Run the training phase

```bash
mkdir -p hpc_outputs
mkdir -p models
sbatch shell/hpcrun.sb
```

3. To generate test result to submit

```bash
sbatch shell/hpceval.sb
```


