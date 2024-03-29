%{ 
This script serves to upload all data files of simulations performed in The
Virtual Brain (TVB) in a Python environment and run them through a
Multivariate Granger Causality analysis and then a State-Space Dynamical
Independence analysis. 

For future work: Create a bash script that will run this code over multiple
scales.

Script written by Borjan Milinković, 2022

%}

%% SET ALL PARAMETERS FOR MVGC AND SSDI ANALYSIS HERE
if ~exist('momax',      'var'), momax       = 20;       end         % maximum model order
if ~exist('moregmode',  'var'), moregmode   = 'LWR';    end         % model order regression mode 
if ~exist('mosel',      'var'), mosel       = 'AIC';    end         % which model order to select
if ~exist('plotm',      'var'), plotm       = 0;        end         % for plotting on seperate figures
if ~exist('regmode',    'var'), regmode     = 'LWR';    end 
if ~exist('stat',       'var'), stat        = 'F';      end
if ~exist('alpha',      'var'), alpha       = 0.05;     end
if ~exist('mhtc',       'var'), mhtc        = 'FDRD';   end


%%%%%%%%%%%%%%% PREOPT

defvar('iseed',    0           ); % initialisation random seed (0 to use current rng state)

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

defvar('nrunsp',   100         ); % pre-optimisation runs (restarts)
defvar('nitersp',  10000       ); % pre-optimisation iterations
defvar('sig0p',    1           ); % pre-optimisation (gradient descent) initial step size
defvar('gdlsp',    2           ); % gradient-descent "line search" parameters
defvar('gdtolp',   1e-10       ); % gradient descent convergence tolerance
defvar('histp',    true        ); % calculate optimisation history?
defvar('ppp',      false       ); % parallelise multiple runs?

%%%%%%%%%%%%%% OPTIMISATION

defvar('ctol',     1e-6        ); % hyperplane clustering tolerance

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

defvar('niterso',  10000       ); % pre-optimisation iterations
defvar('sig0o',    0.1         ); % optimisation (gradient descent) initial step size
defvar('gdlso',    2           ); % gradient-descent "line search" parameters
defvar('gdtolo',   1e-10       ); % gradient descent convergence tolerance
defvar('histo',    true        ); % calculate optimisation history?
defvar('ppo',      false       ); % parallelise multiple runs?



%% LOAD IN DATA:

% Setting up the directory structure for saving files first

resultsDir = '/Users/borjanmilinkovic/Documents/gitdir/TVBEmergence/results';
pwcgcDir = [resultsDir '/ssdiDataMATLAB/SJ3D_3node_withlink_ps_gc-noise/pwcgc/'];
ssdiDataDir = [resultsDir '/ssdiDataMATLAB/SJ3D_3node_withlink_ps_gc-noise/ssdiData/'];
figuresDir = '/ssdiFiguresMATLAB/SJ3D_3node_withlink_ps_gc-noise/';


if exist([ssdiDataDir]) == 0
    mkdir([ssdiDataDir])
end


if exist([pwcgcDir]) == 0
    mkdir([pwcgcDir])
end

% Loading in TVB data.
tvbDir = '/Users/borjanmilinkovic/Documents/gitdir/TVBEmergence/results';
oscDir = 'SJ3D_3node_withlink_ps_gc-noise';
tvbDataDir = '/data';

files = dir([tvbDir filesep oscDir filesep tvbDataDir filesep '*.mat']);


edgeWeightsMatrix = zeros(3, 3, length(files));         % initialise pwcgc matrix                     

for fileNumber = 1:length(files)
    filename = files(fileNumber).name;
    folder = files(fileNumber).folder;
    
    
    tic;
    fprintf('..Starting on %s (%g / %g) of Oscillator Dynamics\n', filename, fileNumber, length(files));
    
    load([folder filesep filename]);        % load the data in
    %data = data';                           % if data needs to be transposed
    data = data(:, 251:end);                % remove first second of initial transients 
    data = demean(data, true);              % demeaning data only if it has not been Z-scored in TVB sim.
    
    % MVGC Analysis
    
    [moaic, mobic] = tsdata_to_varmo(data, momax, moregmode, [], [], []); % Calculate model order using AIC abd BIC criterion
    morder = moselect(sprintf('VAR model order selection, max = %d',momax),mosel, 'AIC', moaic, 'BIC', mobic);
    [A, V] = tsdata_to_var(data, morder, regmode);                        % Create VAR Model
    info = var_info(A, V);                                                % Check information of VAR model
    F = var_to_pwcgc(A, V);                                               % Run Pairwise Granger Causality
    tstat = var_to_pwcgc_tstat(data, V, morder, regmode, stat);           % Estimate Statistic
    
    
    nvars       = size(A,1);
    nobs        = size(data,2);
    ntrials     = 1;
    
    pval = mvgc_pval(tstat, stat, 1, 1, nvars-2, morder, nobs, ntrials);  % Calculate p-values
    sig = significance(pval, alpha, mhtc);                                % Multiple comparisons test.
    
    
    % Save all the edge weights into a single matrix for plotting PWCGC in
    % MATPLOTLIB & SEABORN in Python
    
    edgeWeightsMatrix(:, :, fileNumber) = F/nanmax(F(:));                                             % Normalising edge weights
    
    % Convert VAR Model to equivalent SS model
    
    [Ass, Vss] = transform_var(A, V);
    [A, C, K] = var_to_ss(Ass);
    [fres,ierr] = var2fres(Ass,Vss);
    CAK = Ass; % we need this
    H = var2trfun(Ass, fres); % and this
    
    % setting up model descripton
    r = morder;
    n = size(data,1);
    
    % Saving the MVGC variables that are necessary for SSDI estimates
    
    modfile = sprintf([pwcgcDir 'pwcgc_%s_%g-of-%g.mat'], filename(1:end-4), fileNumber, length(files));
    fprintf('Saving PWCGC model ''%s''. ', modfile);
    save(modfile, 'V', 'CAK', 'H');
    fprintf('Saved..\n');
    toc;
end

mdim = 2;           % select macroscopic dimension
pwcgcFiles = dir([pwcgcDir filesep '*.mat']);
doptp = cell(1, length(pwcgcFiles));

for fileNumber = 1:length(pwcgcFiles)
    filename = pwcgcFiles(fileNumber).name;
    folder = pwcgcFiles(fileNumber).folder;
    
    tic;
    % Load the data
    load([folder filesep filename]);
    
    n = size(V,1);
    m = mdim;
    fres = size(H,3)-1;
    
    
    fprintf('Beginning pre-optimisation for %d-macro on %g / %g Oscillator models\n',m, fileNumber, length(pwcgcFiles));
    
    
    % Initialise the optimisation runs
    rstate = rng_seed(iseed);
    L0p = rand_orthonormal(n,m,nrunsp); % initial (orthonormalised) random linear projections
    rng_restore(rstate);
    
    % Run optimisation
    [doptp{fileNumber},Lp,convp,ioptp,soptp,cputp,ohistp] = opt_gd_ddx_mruns(CAK,L0p,nitersp,sig0p,gdlsp,gdtolp,histp,ppp);

    % Inverse-transform Lopto back for un-decorrelated residuals
    Loptp = transform_proj(Lp,V);
    
    % Preoptima distances
    goptp{fileNumber} = gmetrics(Loptp);
    
    % Proper optimisation
    [uidx,usiz,nrunso] = Lcluster(cell2mat(goptp(1,fileNumber)),ctol,cell2mat(doptp(1,fileNumber))); 
    
    % initialise optimisation
    L0o = Lp(:,:,uidx);
    
    % Run this sucker!
    [dopto{fileNumber},Lo,convp,iopto,sopto,cputo,ohisto] = opt_gd_dds_mruns(H,L0o,niterso,sig0o,gdlso,gdtolo,histo,ppo);
    
    % transform to un-decorrelated again
    Lopto = transform_proj(Lo,Vss);
    
    %inter optima distance
    gopto{fileNumber} = gmetrics(Lopto);
    
    % weighting of nodes in contributing to macro
    
    nodeWeights{fileNumber} = zeros(n, size(Lopto,3));
    for k = 1:size(Lopto,3)
        nodeWeights{fileNumber}(:, k) = 1-gmetricsx(Lopto(:,:,k));
    end

    dynamical_dependence(fileNumber) = min(dopto{fileNumber})
end


%% Saving data

% Saving directories
saveDirDD = [ssdiDataDir 'dynamical_dependence_parametersweep_noise_gc'];
saveDirNodeWeights = [ssdiDataDir 'nodeWeights_parametersweep_noise_gc'];
saveDirEdgeWeights = [ssdiDataDir 'edgeWeights_parametersweep_noise_gc'];

% Dynamical Dependence Vector
dynamical_independence_matrix = reshape(dynamical_dependence, [20,20]);
dynamical_independence_matrix = dynamical_independence_matrix.';
save(saveDirDD, 'dynamical_independence_matrix');

% Maximal Node Weights for each Simulation
maximalNodeWeights=cell(1,length(nodeWeights));
for i = 1:length(nodeWeights)  % -> numel(cellmatrix)
    maximalNodeWeights{i}= nodeWeights{1,i}(:,1);   % change 1 to 7 if you want to extract 7th column
end
maximalNodeWeights = cell2mat(maximalNodeWeights);
save(saveDirNodeWeights, 'maximalNodeWeights');

% Edge weights (GC-graph) for each simulation
edgeWeights = reshape(edgeWeightsMatrix, [3,1200]);
save(saveDirEdgeWeights, 'edgeWeights');




    

    
    
    
    
    
    
    
    




