% This script compares Dynamical Indpendence and Strong Causal Emergence
% on different simple MVAR models.


sim_model;
preoptimise_dd;
optimise_dd;

X = var_to_tsdata(ARA0,V0,1000,1,[],[]);  
Synergy = PhiIDFull(X);


