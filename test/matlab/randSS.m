function [A,C,K,V,H,CAK] = randSS(dimension, varargin)
%RANDSS Creates random SS model and corresponding underlying connectivity
%matrix given a specific dimension size 

%%%   INCOMPLETE!!!!!



defaultMorder = 10;
defaultSpecrad = 0.98;  % rho, spetral norm
defaultW = [];          % var coefficients decay weighting parameter
defaultG = [];          %multi-information (g = -log|R|), g = 0 means zero correlation
defaultNtrials = 1;
defaultNobs = 1000;

p = inputParser;

addRequired(p, 'dimension');
addParameter(p, 'morder', defaultMorder);
addParameter(p, 'specrad', defaultSpecrad);
addParameter(p, 'w', defaultW);
addParameter(p, 'g', defaultG);
addParameter(p, 'ntrials', defaultNtrials);
addParameter(p, 'nobs', defaultNobs);

parse(p, dimension, varargin{:});

dimension = p.Results.dimension;
morder = p.Results.morder;
specrad = p.Results.specrad;
w = p.Results.w;
g = p.Results.g;
ntrials = p.Results.ntrials;
nobs = p.Results.nobs;

randomState = rng_seed(mseed);

% actual analysis and output
connMatrix = randi([0,1],dimension);
connMatrix = connMatrix - diag(diag(connMatrix));
A = var_rand(connMatrix, morder, specrad, w);
V = corr_rand(dimension, g);
tsdata = var_to_tsdata(A, V, nobs, ntrials);

% additional stuff for ss
V0 = corr_rand(dimension, g);
[A0,C0,K0] = iss_rand(dimension, morder, specrad, w);
[A,C,K,V] = transform_ss(A0,C0,K0,V0); % transform model to decorrelated-residuals form
H = ss2trfun(A,C,K, fres); % transfer function
CAK = iss2cak(A,C,K); % CAK sequence for pre-optimisation

rng_restore(randomstate);

end

