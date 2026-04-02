#! /bin/bash

Dir="/caldera/hovenweep/projects/usgs/hazards/pcmsc/cosmos/PugetSound/sfincs/20250122_synthetic_future_meanchange_100yr_Intel"
RP="1,2,5,10,20,50,100"
SLR="000,025,050,100,150,200,300"
Domains="01_King,02_Pierce"   #"01_King,02_Pierce"
SubCategories="_median,_low,_high"



# Submit the First PostProcessing Step
#jid1=$(sbatch --parsable --job-name=01_PostPython RunPython.slurm part1_read_process_SFINCS_runs_v4_synthetic.py "$Dir" "$RP" "$SLR" "$Domains" "$SubCategories")

# Submit the Second PostProcessing Step (subgridding)
jid2=$(sbatch --parsable --job-name=02_PostPython RunPython.slurm part2_downscale_pugetsound_v9.py "$Dir" "$RP" "$SLR" "$Domains" "$SubCategories")   # --dependency=afterany:$jid1

# Submit the Third PostProcessing Step (Disconnected Flooding)
jid3=$(sbatch --parsable --job-name=03_PostPython --dependency=afterany:$jid2 RunPython.slurm part3_remove_disconnected_flooding_v3.py "$Dir" "$RP" "$SLR" "$Domains" "$SubCategories")
	

