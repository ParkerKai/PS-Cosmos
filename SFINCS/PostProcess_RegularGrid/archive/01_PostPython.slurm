#!/bin/bash
#SBATCH --job-name=Sfincs_Post01
#SBATCH --mail-type=begin,end,fail    # Mail events (BEGIN, END, FAIL)
#SBATCH --mail-user=kaparker@usgs.gov
#SBATCH --nodes=1                     # Number of nodes requested (must be >3)
#SBATCH --ntasks=1                    # Number of tasks per node
#SBATCH --cpus-per-task=256
#SBATCH --account=pcmsc               # Account to charge the job to
#SBATCH --time=1:00:00             # requested job time D-HH:MM:SS
#SBATCH --output=Post01_%j.log       # Standard output and error log

## source the conda base env you need (and installed prior)
source ~/miniconda3/etc/profile.d/conda.sh
## activate the virtual env you need within the conda base env
conda activate SfincsPostProcess

# python environment should have been created prior to submitting job
srun python part1_read_process_SFINCS_runs_v4_synthetic.py
                                                                
