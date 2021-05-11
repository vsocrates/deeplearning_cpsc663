#!/bin/bash
#SBATCH --job-name=vsoc_nodeclass
#SBATCH --gpus=1
#SBATCH --partition=gpu,general,interactive
#SBATCH --out="nodeclass-%j.out"
#SBATCH --time=1:00:00
#SBATCH --gpus-per-node=1
#SBATCH --nodes=1 --ntasks=1
#SBATCH --mem-per-cpu=50G
#SBATCH --mail-type=ALL

mem_bytes=$(</sys/fs/cgroup/memory/slurm/uid_${SLURM_JOB_UID}/job_${SLURM_JOB_ID}/memory.limit_in_bytes)
mem_gbytes=$(( $mem_bytes / 1024 **3 ))

echo "Starting at $(date)"
echo "Job submitted to the ${SLURM_JOB_PARTITION} partition, the default partition on ${SLURM_CLUSTER_NAME}"
echo "Job name: ${SLURM_JOB_NAME}, Job ID: ${SLURM_JOB_ID}"
echo "  I have ${SLURM_CPUS_ON_NODE} CPUs and ${mem_gbytes}GiB of RAM on compute node $(hostname)"

module restore deeplearningmods
conda activate deeplearning_env2

python NodeClassification.py  xu
python NodeClassification.py  kipfwelling

