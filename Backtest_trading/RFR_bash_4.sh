#!/bin/bash
#PBS -N Dynamics_X
#PBS -S /bin/sh
#PBS -r n 
#PBS -l walltime=167:00:00
#PBS -l nodes=1:ppn=1
#PBS -l mem=80000MB
#PBS -A PAA0009
#PBS -V
#PBS -m ae
#--------------------------------------------------------------#
NPROCS=`wc -l < $PBS_NODEFILE`
cd $PBS_O_WORKDIR
#--------------------------------------------------------------#
# Molcas settings 
#--------------------------------------------------------------#
export MOLCAS="/users/PCS0202/bgs0361/bin/7.8.dev"
export MOLCASMEM=8000MB
export MOLCAS_MOLDEN=ON
export MOLCAS_PRINT=normal
export TINKER="/users/PCS0202/bgs0361/bin/7.8.dev/tinker/bin_qmmm"
#--------------------------------------------------------------#
#  Change the Project!!!
#--------------------------------------------------------------#
#export Project=$PBS_JOBNAME
export Project=RFR_bash_3
export WorkDir=/tmp/$Project.$PBS_JOBID
mkdir -p $WorkDir
export InpDir=$PBS_O_WORKDIR
echo $HOSTNAME > $InpDir/nodename_$PBS_JOBNAME
echo $JOBID > $InpDir/jobid
#--------------------------------------------------------------#
# Copy of the files - obsolete
#--------------------------------------------------------------#
#cp $InpDir/$Project.xyz $WorkDir/$Project.xyz
#cp $InpDir/$Project.key $WorkDir/$Project.key
#cp $InpDir/*.prm $WorkDir/
#--------------------------------------------------------------#
# Start job
#--------------------------------------------------------------#
sed -i "s|PBS_O_WORKDIR|$PBS_O_WORKDIR|g" $InpDir/$Project.py
cd $WorkDir
 /users/PCS0202/bgs0361/bin/Python-3.6.2/bin/python $InpDir/$Project.py 1> $InpDir/$Project.out 2> $InpDir/$Project.err
cp -r $WorkDir/* $InpDir

