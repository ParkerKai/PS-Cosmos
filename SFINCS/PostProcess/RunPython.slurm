#!/bin/bash
#SBATCH --mail-type=begin,end,fail    # Mail events (BEGIN, END, FAIL)
#SBATCH --mail-user=kaparker@usgs.gov
#SBATCH --nodes=1                     # Number of nodes requested (must be >3)
#SBATCH --ntasks=1                    # Number of tasks per node
#SBATCH --cpus-per-task=256
#SBATCH --account=pcmsc               # Account to charge the job to
#SBATCH --time=30:00:00             # requested job time D-HH:MM:SS
#SBATCH --output=Post_%j.log       # Standard output and error log

## source the conda base env you need (and installed prior)
source ~/miniconda3/etc/profile.d/conda.sh
## activate the virtual env you need within the conda base env
conda activate SfincsPostProcess


echo "$SLURM_JOBID    `date`"

echo "`which python3`"
echo "args=$@"
echo "Running .."
echo

srun python $@
code=$?

echo
echo
echo "SLURM STATS"
echo "$SLURM_JOBID FIN `date`"
echo
scontrol show job --details $SLURM_JOBID #| grep "JobId\|\ Nodes"
echo

# to use:
# sbatch run_tf.slurm my_code.py arg1 arg2
# sbatch --job-name=test01 --gres=gpu:2 run_tf.slurm my_code.py arg1 arg2 # for gpu > 1, must use different strategy

exit $code
