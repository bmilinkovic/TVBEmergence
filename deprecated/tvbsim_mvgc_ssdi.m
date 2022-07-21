clear all;
close all;
clc;

% file directory

% file_dir = "/Users/borjanmilinkovic/PycharmProjects/tvb/results/simulationData.mat";
% load(file_dir);
% data = data'; % transposing the data.
%% Pair-wise Granger Causality Analysis - Parameters
% VAR model order estimation
if ~exist('moregmode', 'var'), moregmode = 'LWR';   end % VAR model estimation regression mode ('OLS' or 'LWR')
if ~exist('mosel',     'var'), mosel     = 'LRT';   end % model order selection ('ACT', 'AIC', 'BIC', 'HQC', 'LRT', or supplied numerical value)
if ~exist('momax',     'var'), momax     = 20; end % maximum model order for model order selection

% VAR model parameter estimation
if ~exist('regmode',   'var'), regmode   = 'LWR';   end % VAR model estimation regression mode ('OLS' or 'LWR')

% MVGC (time domain) statistical inference
if ~exist('alpha',     'var'), alpha     = 0.05;    end % significance level for Granger casuality significance test
if ~exist('stest',     'var'), stest     = 'F';     end % statistical inference test: 'F' or 'LR' (likelihood-ratio chi^2)
if ~exist('mhtc',      'var'), mhtc      = 'FDRD';  end % multiple hypothesis test correction (see routine 'significance')


if ~exist('fres',      'var'), fres      = [];       end

% control of coding
if ~exist('seed',      'var'), seed      = 0;       end % random seed (0 for unseeded)
if ~exist('plotm',     'var'), plotm     = 0;       end % plot mode (figure number offset, or Gnuplot terminal string)

%% State-space Dynamical Independence Analysis - Parameters

% Must supply m = macroscopic state dimension
if ~exist('m',        'var'), m        = 3;         end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

if ~exist('psig0',    'var'), psig0    = 1;         end % pre-optimisation (gradient descent) initial step size
if ~exist('ssig0',    'var'), ssig0    = 0.1;       end % dd optimisation (gradient descent) initial step size
if ~exist('dsig0',    'var'), dsig0    = 0.0001;    end % dd optimisation (evolution strategy) initial step size
if ~exist('gdrule',   'var'), gdrule   = [2,1/2];   end % gradient-descent "line search" parameters
if ~exist('gdtol',    'var'), gdtol    = 1e-10;     end % gradient descent convergence tolerance
if ~exist('esrule',   'var'), esrule   = 1/5;       end % evolution strategy step-size adaptation rule
if ~exist('estol',    'var'), estol    = 1e-8;      end % evolution strategy convergence tolerance
if ~exist('npiters',  'var'), npiters  = 10000;     end % pre-optimisation (gradient descent) iterations
if ~exist('nsiters',  'var'), nsiters  = 1000;      end % spectral method optimisation (gradient descent) iterations
if ~exist('nditers',  'var'), nditers  = 100;       end % state-space method optimisation (evolution strategy) iterations
if ~exist('nruns',    'var'), nruns    = 10;        end % runs (restarts)
if ~exist('hist',     'var'), hist     = true;      end % calculate optimisation history?
if ~exist('iseed',    'var'), iseed    = 0;         end % initialisation random seed (0 to use current rng state)
if ~exist('oseed',    'var'), oseed    = 0;         end % optimisation random seed (0 to use current rng state)

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

if ~exist('resdir',   'var'), resdir   = tempdir;   end % results directory
if ~exist('rid',      'var'), rid      = '';        end % run ID tag

%%%%%%%%%%%% For plotting this might not be neceesary if I migrate graphing
%%%%%%%%%%%% into Python. 

if ~exist('gpterm',   'var'), gpterm   = 'x-pdf';   end % Gnuplot terminal
if ~exist('gpfsize',  'var'), gpfsize  = 14;        end % Gnuplot font size
if ~exist('gpplot',   'var'), gpplot   = 2;         end % Gnuplot display? (0 - generate command files, 1 - generate image files, 2 - plot)

%% Pair-wise Conditional Granger Causality Analysis

% demean activity
dataDetrend = demean(data, true);

% 1. selecting model order & graphing
if isnumeric(plotm), plotm = plotm+1; end
[moaic,mobic,mohqc,molrt] = tsdata_to_varmo(dataDetrend,momax,moregmode,[],[],plotm);
morder = moselect(sprintf('VAR model order selection (max = %d)',momax), mosel,'AIC',moaic,'BIC',mobic,'HQC',mohqc,'LRT',molrt);
assert(morder > 0,'selected zero model order! GCs will all be zero!');
if morder >= momax, fprintf(2,'*** WARNING: selected maximum model order (may have been set too low)\n'); end

% 2. estimating VAR model
[Avar,Vvar] = tsdata_to_var(dataDetrend,morder,regmode);
% 3. check VAR model information
info = var_info(Avar,Vvar);
% 4. estimate time-domain pwcGC.
G = var_to_pwcgc(Avar,Vvar);
% 5. calculate test-statistics
Gstat = var_to_pwcgc_tstat(dataDetrend,Vvar,morder,regmode,stest);
% 6. calculate p-values
Gpval = mvgc_pval(Gstat,stest,1,1,size(Avar, 1)-2,morder,size(dataDetrend, 2),1);
% 7. multiple comparisons
GmultiComparison = significance(Gpval,alpha,mhtc);

% 8. plot pwcGC and F-test matrix (this can be done in Python with
% seaborn/matplotlib) = export _G_ and _GmultiComparison_ as numpy arrays
plotData = {G, GmultiComparison};       %setting which data to plot
plotTitle = {'pwcGC', [stest '-test']}; %setting titles of graphs
maxG = 1.1*max(nanmax(G(:)));
if isnumeric(plotm), plotm = plotm+1; end
plot_gc(plotData,plotTitle,[],maxG,plotm,[0.6,2.5]);

%% necessary parameters for SSDI

n = size(G, 1); % size of network, i.e. number of nodes
[ARA,V] = transform_var(Avar,Vvar);       % transform model to decorrelated-residuals form
[A,C,K] = var_to_ss(ARA);
if isempty(fres)
		[fres,ierr] = var2fres(ARA,V,[sitol,siminp2,simaxp2]);
		if isnan(fres) % failed!
			fprintf(2,'WARNING: Spectral integral frequency resolution estimation failed - defaulting to autocorrelation estimate');
			[fres,ierr] = var2fres(ARA,V);  % use autocorrelation-based estimate
		end
	end
	H = var2trfun(ARA,fres);                % transfer function
	CAK = ARA;       
 % state-space model parameters


%% State-space Dynamical Independence - Initialisation

% set gradient descent strategy parameters
ifgd = gdrule(1);
nfgd = gdrule(2);
% set 1+1 evolution strategy parameters
[ifes, nfes] = es_parms(esrule, m*(size(A, 1)-m));
% initialising the entire optimisation
dopt  = zeros(1,nruns);
ioptp = zeros(1,nruns);
iopts = zeros(1,nruns);
ioptd = zeros(1,nruns);
cputp = zeros(1,nruns);
cputs = zeros(1,nruns);
cputd = zeros(1,nruns);
% initialising histograms
if hist
	dhistp = cell(nruns,1);
	dhists = cell(nruns,1);
	dhistd = cell(nruns,1);
end

rstate = rng_seed(iseed);
Lopt = rand_orthonormal(size(A,1),m,nruns); % initial (orthonormalised) random linear projections
rng_restore(rstate);

%% State-space Dynamical Independence - Analysis


rstate = rng_seed(oseed);
st = tic;
for k = 1:nruns

	fprintf('run %2d of %2d\n',k,nruns);

	Loptk = Lopt(:,:,k);

	% "Proxy" DD pre-optimisation (gradient descent)

	tcpu = cputime;
	[doptk,Loptk,converged,sigk,ioptp(k),dhistk] = opt_gd_ddx(CAK,Loptk,npiters,psig0,ifgd,nfgd,gdtol,hist);
	cputp(k) = cputime-tcpu;
	if hist, dhistp{k} = dhistk; end
	fprintf('\tpopt : dd = %.4e : sig = %.4e : ',doptk,sigk);
	if converged > 0, fprintf('converged(%d)',converged); else, fprintf('unconverged '); end
	fprintf(' in %4d iterations : CPU secs = %6.2f\n',ioptp(k),cputp(k));

	% DD optimisation (gradient descent) using spectral integration method

% 	tcpu = cputime;
% 	[doptk,Loptk,converged,sigk,iopts(k),dhistk] = opt_gd_dds(H,Loptk,nsiters,ssig0,ifgd,nfgd,gdtol,hist);
% 	cputs(k) = cputime-tcpu;
% 	if hist, dhists{k} = dhistk; end
% 	fprintf('\tsopt : dd = %.4e : sig = %.4e : ',doptk,sigk);
% 	if converged > 0, fprintf('converged(%d)',converged); else, fprintf('unconverged '); end
% 	fprintf(' in %4d iterations : CPU secs = %6.2f\n',iopts(k),cputs(k));

	% DD optimisation (evolutionary strategy) using state-space (DARE) method (most accurate, but may be slower)

	tcpu = cputime;
	[doptk,Loptk,converged,sigk,ioptd(k),dhistk] = opt_es_dd(A,C,K,Loptk,nditers,dsig0,ifes,nfes,estol,hist);
	cputd(k) = cputime-tcpu;
	if hist, dhistd{k} = dhistk; end
	fprintf('\tdopt : dd = %.4e : sig = %.4e : ',doptk,sigk);
	if converged > 0, fprintf('converged(%d)',converged); else, fprintf('unconverged '); end
	fprintf(' in %4d iterations : CPU secs = %6.2f\n',ioptd(k),cputd(k));

	Lopt(:,:,k) = Loptk;
	dopt(k) = doptk;

end
et = toc(st);
rng_restore(rstate);

% Transform Lopt back to correlated residuals form

Lopt = transform_proj(Lopt,V0); % V0 is the original residuals covariance matrix

%% ALL NEW STUFF TO BE SORTED
% Sort (local) optima by dynamical dependence

[dopt,sidx] = sort(dopt);
ioptp = ioptp(sidx);
iopts = iopts(sidx);
ioptd = ioptd(sidx);
cputp = cputp(sidx);
cputs = cputs(sidx);
cputd = cputd(sidx);
if hist
	dhistp = dhistp(sidx);
	dhists = dhists(sidx);
	dhistd = dhistd(sidx);
end
Lopt = Lopt(:,:,sidx);
fprintf('\noptimal dynamical dependence =\n'); disp(dopt');

fprintf('\ntotal time = %s\n\n',datestr(seconds(et),'HH:MM:SS.FFF'));

fprintf('popt CPU secs = %7.4f +- %6.4f\n',mean(cputp),std(cputp));
fprintf('sopt CPU secs = %7.4f +- %6.4f\n',mean(cputs),std(cputs));
fprintf('dopt CPU secs = %7.4f +- %6.4f\n',mean(cputd),std(cputd));
fprintf('\n');

nweight = zeros(n,nruns);
for k = 1:nruns
	nweight(:,k) = 1-gmetricsx(Lopt(:,:,k));
end

Loptd = gmetrics(Lopt);

% Save workspace

clear k doptk Loptk sigk dhistk converged
if hist
	wsfile = fullfile(resdir,[scriptname '_hist' rid '.mat']);
else
	wsfile = fullfile(resdir,[scriptname '_nohist' rid '.mat']);
end
fprintf('*** saving workspace in ''%s''... ',wsfile);
save(wsfile);
fprintf('done\n\n');

% Plot optimisation histories

if hist
	gptitle = sprintf('Optimisation history : n = %d, r = %d, m = %d',n,r,m);
	gpstem   = fullfile(resdir,[scriptname '_opthist' rid]);
	gpscale  = [Inf,1.5];
	dhist    = {dhistp;dhists;dhistd};
	niters   = [npiters;nsiters;nditers];
	havegrad = [true,true,false];
	titles   = {'Pre-optimisation (GD)';'Spectral optimisation (GD)';'SS optimisation (ES)'};
	gp_opthist(dhist,niters,havegrad,titles,gptitle,gpstem,gpterm,gpscale,gpfsize,gpplot);
end

% Plot inter-optima subspace distances

gptitle = sprintf('Inter-optimum distance : n = %d, r = %d, m = %d',n,r,m);
gpstem = fullfile(resdir,[scriptname '_iodist' rid]);
gpscale = [1.2,1.1];
gp_iodist(Loptd,gptitle,gpstem,gpterm,gpscale,gpfsize,gpplot);

% To display axis projection-weighted graph, e.g.:
%
k = 3; wgraph2dot(nweight(:,k),eweight,fullfile(resdir,sprintf('graph%s_run%03d',rid,k)),[],gvprog);