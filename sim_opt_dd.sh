#!/bin/bash

# Template batch script for running sim_opt_dd across all scales
shopt -s expand_aliases
source ~/.bash_profile
# base name of this script (without extension)
scriptname=$(basename $0 .sh)

echo -e "\n*** Running batch script '"$scriptname"' ***\n"

# path to Matlab code directory
codedir=$GITDIR/TVBEmergence

# current directory (the directory this script is run in - log files will go here)
currdir=$(pwd -P)

# Restrict cores?
#if [ -n "$1" ]; then
#	tset="taskset -c $1"
#fi

# Matlab invocation
runmatlab="matlab -nojvm -nodisplay"

# Parameters

runid="my_run"

# model/data
n=40
r=60
rho=0.9
gvdisp=[]

# preoptimise
nrunsp=1000
nitersp=10000
gpplot=0

# optimise
nrunso=1000
niterso=10000
gpplot=0

# run multiple concurrent Matlab sessions with different parameters
for mdim in {1..8}
do
    # the log file
    logfile=$currdir/$scriptname\_m\_$m.log

    # Matlab commands
    matcmds="\
		clear;\
        moddir   = '$currdir';\
        modname  = '$runid_model';\
        n        =  $n;\
        r        =  $r;\
        rho      =  $rho;\
        gvdisp   =  $gvdisp;\
        sim_model;\
		clear;\
        poptdir  = '$currdir';\
        poptname = '$runid_preopt';\
        nrunsp   =  $nrunsp;\
        nitersp  =  $nitersp;\
        preoptimise_dd;\
        clear;\
        poptdir  = '$currdir';\
        poptname = '$runid_preopt';\
        optdir   = '$currdir';\
        optname  = '$runid_opt';\
        nrunso   =  $nrunso;\
        niterso  =  $niterso;\
        optimise_dd;\
        quit"

    # run Matlab as background job
    cd $codedir && matlab -nojvm -nodisplay -r "$matcmds" > $logfile < /dev/null 2>&1 &
done
