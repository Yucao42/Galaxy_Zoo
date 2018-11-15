# Galaxy_Zoo

Project of Computer Vision 2018 NYU course.

## To run on HPC cluster

1. Create conda environment

```bash
conda env create -f requirements.yaml
```

2. Run the training phase

```bash
sbatch shell/hpcrun.sb
```

3. To generate test result to submit

```bash
sbatch shell/hpceval.sb
```


