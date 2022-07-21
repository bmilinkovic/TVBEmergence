% Here add code to compute a Multivariate Granger-Causality analysis on The
% Virtual Brain (TVB) simulations.

%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%   MVGC Parameters     %%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% VAR model order estimation
varmosel = 'AIC';
varmomax = 32;

% SS model order estimation
ssmosel = 'SVC';

% MVGC frequency domain [for spectral analysis]
fres = [];

if ~exist('seed', 'var'), seed = 0; end % setting ranndom seed for replication
if ~exist('plotm', 'var'), plotm = 0; end %plotting offset for plotting

%% Plugging in TVB data

rng_seed(seed);
