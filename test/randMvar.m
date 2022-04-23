function [tsdata, A, V, connMatrix] = randMvar(dimension, varargin)
%RANDMVAR Siimulates a random connectivity matrix and the correspondinng
%random var model with its corresponding time series

defaultMorder = 10;
defaultRho = 0.98;      %spectral radius
defaultW = [];          %decay weighting parameter
defaultRmii = [];          % residuals multi-information (g = -log|R|), g = 0 means zero correlation
defaultNtrials = 1;
defaultNobs = 1000;

p = inputParser;

addRequired(p, 'dimension');
addRequired(
addParameter(p, 'morder', defaultMorder);
addParameter(p, 'rho', defaultRho);
addParameter(p, 'w', defaultW);
addParameter(p, 'rmii', defaultRmii);
addParameter(p, 'ntrials', defaultNtrials);
addParameter(p, 'nobs', defaultNobs);

parse(p, dimension, varargin{:});

dimension = p.Results.dimension;
morder = p.Results.morder;
specrad = p.Results.rho;
w = p.Results.w;
g = p.Results.rmii;
ntrials = p.Results.ntrials;
nobs = p.Results.nobs;

% actual analysis and output
connMatrix = randi([0,1],dimension);
connMatrix = connMatrix - diag(diag(connMatrix));
A = var_rand(connMatrix, morder, rho, w);
V = corr_rand(dimension, rmii);
tsdata = var_to_tsdata(A, V, nobs, ntrials);




end


