%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Dynamical dependence pre-optimisation (via proxy DD)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% Supply pair-wise Granger-causal model for simulated TVB data with 
% decorrelated and normalised residuals 
% Model properties required for this script are:
%
% mdescript - a model description string
% V        - original (i.e., pre-decorrelation) residuals covariance matrix
% CAK       - CAK sequence for pre-optimisation by proxy DD
% H         - transfer function for DD calculation by spectral method
%
% Specify a macroscopic dimension mdim and simulation parameters, or accept
% defaults (see below). After running this script, run Lcluster until you are
% happy with the hyperplane clustering; e.g.:
%
% ctol = 1e-6; Lcluster(goptp,ctol,doptp);
%
% Then run preopt_dd_to_opt_dd.
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
mdim = 3;
defvar('moddir',   tempdir     );  % model directory
defvar('modname',  'gc_model' );  % model filename root
defvar('poptdir',  tempdir     );  % pre-optimisation directory
defvar('poptname', 'preopt_dd' );  % pre-optimisation filename root

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

defvar('iseed',    0           ); % initialisation random seed (0 to use current rng state)

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

defvar('nrunsp',   100         ); % pre-optimisation runs (restarts)
defvar('nitersp',  10000       ); % pre-optimisation iterations
defvar('sig0p',    1           ); % pre-optimisation (gradient descent) initial step size
defvar('gdlsp',    2           ); % gradient-descent "line search" parameters
defvar('gdtolp',   1e-10       ); % gradient descent convergence tolerance
defvar('histp',    true        ); % calculate optimisation history?
defvar('ppp',      false       ); % parallelise multiple runs?

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

defvar('gpterm',   'x11'       ); % Gnuplot terminal
defvar('gpscale',  [Inf,0.6]   ); % Gnuplot scale
defvar('gpfsize',  14          ); % Gnuplot font size
defvar('gpplot',   2           ); % Gnuplot display? (0 - generate command files, 1 - generate image files, 2 - plot)

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


%assert(exist('mdim','var'),'Must supply macro dimension ''mdim''');

% Load model

modfile = [fullfile(moddir,modname) '.mat'];
fprintf('\n*** loading model from ''%s''... ',modfile);
load(modfile);
fprintf('done\n\n');

n = size(V,1);
m = mdim;
fres = size(H,3)-1;

fprintf('%s: pre-optimisation for m = %d\n\n',mdescript,m);

% Initialise optimisations

rstate = rng_seed(iseed);
L0p = rand_orthonormal(n,m,nrunsp); % initial (orthonormalised) random linear projections
rng_restore(rstate);

% Multiple optimisation runs

st = tic;
[doptp,Lp,convp,ioptp,soptp,cputp,ohistp] = opt_gd_ddx_mruns(CAK,L0p,nitersp,sig0p,gdlsp,gdtolp,histp,ppp);
et = toc(st);

% Inverse-transform Lopto back for un-decorrelated residuals

Loptp = transform_proj(Lp,V);

fprintf('\noptimal dynamical dependence =\n'); disp(doptp');
fprintf('Simulation time = %s\n\n',datestr(seconds(et),'HH:MM:SS.FFF'));
fprintf('CPU secs per run = %7.4f +- %6.4f\n\n',mean(cputp),std(cputp));

% Plot optimisation histories

if histp && ~isempty(gpterm)
	gptitle  = sprintf('Pre-optimisation history: %s, m = %d',mdescript,m);
	gpstem   = fullfile(tempdir,'preopt_hist');
	gp_opthist({ohistp},nitersp,true,true,{'Pre-optimisation (GD)'},gptitle,gpstem,gpterm,gpscale,gpfsize,gpplot);
end

% Plot inter-optima subspace distances

goptp = gmetrics(Loptp);
if ~isempty(gpterm)
	gptitle = sprintf('Inter-preoptimum distance: %s, m = %d',mdescript,m);
	gpstem = fullfile(tempdir,'preopt_iodist');
	gp_iodist(goptp,gptitle,gpstem,gpterm,[1.2,1.1],gpfsize,gpplot);
end

% Save pre-optimisation results


preoptimisation_history = cell(1, length(ohistp));
for i = 1:length(ohistp)  % -> numel(cellmatrix)
preoptimisation_history{i}= ohistp{i,1}(:,1);   % change 1 to 7 if you want to extract 7th column
end

preoptimisationDataDir = '/Users/borjanmilinkovic/Documents/gitdir/TVBEmergence/results/ssdiDataMATLAB/preoptData';
preoptimisationFile = fullfile(preoptimisationDataDir, ['preoptDD_mdim' num2str(mdim) '_plotting data.mat']);
fprintf('\n **** saving pre-optimisation files for plotting in matplotlib: %s', preoptimisationFile);
save(preoptimisationFile, 'preoptimisation_history', 'nitersp', 'goptp');
fprintf('..Saved! \n\n');

clear n m st et tcpu rstate gpplot gpterm gpscale gpfsize gptitle gpstem
if histp
	poptfile = fullfile(poptdir,[poptname '_mdim_' num2str(mdim) '_H.mat']);
else
	poptfile = fullfile(poptdir,[poptname '_mdim_' num2str(mdim) '_N.mat']);
end
fprintf('\n*** saving pre-optimisation results in ''%s''... ',poptfile);
save(poptfile);
fprintf('done\n\n');
