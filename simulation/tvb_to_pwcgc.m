% This code loads in The Virtual Brain simulations conducted in Python on 
% that have simuated coupled neural mass models simulating neural-like 
% behaviour.

%% LOADING NUMPY DATA
%  Loading in a dataset that has been saved as a .mat file from the TVB
%  simulation

data_dir = '/Users/borjanmilinkovic/Documents/gitdir/TVBEmergence/results/osc2d_3node_nodelay_a-0.5_ps_gc-noise/data';
data_file = 'osc2d_3node_nodelay_gc-0.545559_noise-0.002336.mat'

%  Here set the datafile name to whichever file you wish to analyse. 
load([data_dir filesep data_file]); 

data = data'; % transpose data
data = data(:, 251:end) % get rid of the initial transience by splicing the first second of data. 

%% SETTING VAR MODELLING PARAMETERS

if ~exist('momax',      'var'), momax       = 20; end       % maximum model order
if ~exist('moregmode',  'var'), moregmode   = 'LWR'; end    % model order regression mode 
if ~exist('mosel',      'var'), mosel       = 'BIC'; end    % which model order to select
if ~exist('plotm',      'var'), plotm       = 0; end        % for plotting on seperate figures

%% VAR MODELLING
%  Model order selection and VAR coefficients are estimated here which will
%  then be used for all subsequent analyses.

data = demean(data, true); % do this only if data was not demeaned of Z-scored in python

% 1. Model order estimation
%    Here we estimate the optimal model order given out _data_
%    Using either Bayesian or Akaike Information Criterion (BIC or AIC).

if isnumeric(plotm), plotm = plotm+1; end
[moaic, mobic] = tsdata_to_varmo(data, momax, moregmode, [], [], plotm);

% 2. Select the model order we wish to go with from the tsdata_to_varmo
%    computation above

morder = moselect(...
         sprintf('VAR model order selection, max = %d',momax),...
         mosel, 'AIC', moaic, 'BIC', mobic);

% 3. VAR model estimation with above _morder_ selection
%    This provides us with two matrices, A and V. A, represents the VAR
%    coefficients and V is the residuals covariance matrix. We will also
%    need to add a new variable here: the regression mode used for the VAR
%    model estimation. Similarly as we had 'LWR' and 'OLS' regression modes
%    for model order estimation, both of these options are available to us
%    here too.

if ~exist('regmode', 'var'), regmode = 'LWR'; end 

[A, V] = tsdata_to_var(data, morder, regmode);

% 4. Check the VAR model estimation information

info = var_info(A, V);

%% GRANGER CAUSALITY ANALYSIS - TIME DOMAIN
%  We finally estimate the pair-wise, conditional, Granger-causality
%  between each pair of nodes in our simulated subnetwork. This is the
%  simplest part of the analysis actually. :)

F = var_to_pwcgc(A, V);

%% TEST STATISTICS
%  1. Calculate the test statistics, we can either perform a F-test or a
%     likelihood-ratio chi^2 test, so we need to se this as another new
%     variable in our analysis before we perform our test statistic
%     calculation

if ~exist('stat', 'var'), stat = 'F'; end
    
tstat = var_to_pwcgc_tstat(data, V, morder, regmode, stat);

% 2. Calculate the p-values: here we need 3 new variables that includes the
%    number of variabes, observations and trials of our data.

nvars = size(A,1);
nobs = size(data,2);
ntrials = 1;

pval = mvgc_pval(tstat, stat, 1, 1, nvars-2, morder, nobs, ntrials);

% 3. Significance testing which includes for multiple comparisons: here
%    we will need to set 2 variables, our alpha, and our multiple
%    comparisons test.

if ~exist('alpha',  'var'), alpha   = 0.05; end
if ~exist('mhtc',   'var'), mhtc    = 'FDRD'; end

sig = significance(pval, alpha, mhtc);

%% PLOTTING TIME-DOMAIN PWCGC GRAPH, AS WELL AS SIGNIFICANCE MATRIX

maxF = 1.1*max(nanmax(F(:)));
pdata = {F,sig};
ptitle = {'PWCGC (estimated)',[stat '-test']};
maxp = [maxF,1];
if isnumeric(plotm), plotm = plotm+1; end
plot_gc(pdata,ptitle,[],maxp,plotm,[0.6,2.5]);

% Save this plot to a pwcgc figure directory

%% CONVERTING VAR MODEL INTO SS MODEL FOR SSDI ANALYSIS

[Ass, Vss] = transform_var(A, V);
[A, C, K] = var_to_ss(Ass);
[fres,ierr] = var2fres(Ass,Vss);

CAK = Ass; % we need this
H = var2trfun(Ass, fres); % and this

%% PLOTTING THE GC GRAPH IN GRAPHVIZ
%  This can potentially be called to python instead and graphed through
%  seaborn and matplotlib

if ~exist('gvprog',         'var'), gvprog          = 'neato'; end      % GraphViz program/format (also try 'neato', 'fdp')
if ~exist('gvdisp',         'var'), gvdisp          = true; end         % GraphViz display? Empty for no action, true to display, false to just generate files)
if ~exist('modelviz_dir',   'var'), modelviz_dir    = tempdir; end
if ~exist('model_name',     'var'), model_name      = 'gc_model'; end


eweight = F/nanmax(F(:)); % normalising eweights (can be done in networkx as well)
gfile = fullfile(modelviz_dir, model_name);
wgraph2dot(nvars,eweight,gfile,[],gvprog,gvdisp);

%% Model Details
r = morder
n = size(data,1)

mdescript = sprintf('%d-variable ISS(%d)',n,r);

fprintf('--------------------------------------------\n');
fprintf('Model                : %s\n',mdescript);
fprintf('--------------------------------------------\n');
fprintf('Dimension            : %d\n',n);
fprintf('Complexity (CAK)     : %d x %d x %d\n',size(CAK,1),size(CAK,2),size(CAK,3));
fprintf('Frequency resolution : %d\n',size(H,3)-1);
fprintf('--------------------------------------------\n\n');

%% SAVE PWCGC MODEL

modfile = [fullfile(tempdir, model_name) '.mat'];
fprintf('Saving PWCGC model ''%s''. ', modfile);
save(modfile, 'V', 'CAK', 'H');
fprintf('SAVED');


