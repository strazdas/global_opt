#!/bin/bash
#
#$ -j y
#$ -S /bin/bash
#$ -cwd

mpirun $HOME/global_opt/exps_mpi.py 0.01 False disimpl-2v
