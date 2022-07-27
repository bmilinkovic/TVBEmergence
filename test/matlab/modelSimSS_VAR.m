function [A0, V0, A, C, K, V, CAK, H, gcVAR, gcSS] = modelSimSS_VAR(conn,model,varargin)
%MODELSIMSS_VAR: 
% Set up SS or VAR model, calculate transfer function, CAK sequence, etc.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


defaultMorder =         7;                      %model order
defaultW =              1;                      %var coefficient decay



defaultRho =            0.9;                    %spectral radius
defaultRmii =             1;                    %residuals multi-information
defaultFres =           [];                     %frequency resolution
defaultNsics =          100;                    %number of samples for spectral integration check
defaultMseed =          0;                      %model random seed

p = inputParser;

addRequired(p, 'conn');
addRequired(p, 'model');   % maybe useful for selecting whether the model is ss or var

addParameter(p, 'morder', defaultMorder);
addParameter(p, 'w', defaultW);

addParameter(p, 'rho', defaultRho);
addParameter(p, 'rmii', defaultRmii);
addParameter(p, 'fres', defaultFres);
addParameter(p, 'nsics', defaultNsics);
addParameter(p, 'mseed', defaultMseed);

parse(p, conn, model, varargin{:});

conn = p.Results.conn;
model = p.Results.model;

morder = p.Results.morder;
w = p.Results.w;

rho = p.Results.rho;
rmii = p.Results.rmii;
fres = p.Results.fres;
nsics = p.Results.nsics;
mseed = p.Results.mseed;



rstate = rng_seed(mseed);
n = size(conn,1);
if model == 'var'
	A0          = var_rand(n, morder,rho,w);               % random VAR model
    V0          = corr_rand(n,rmii);                       % residuals covariance matrix
    gcVar       = var_to_pwcgc(A0,V0);                     % causal graph
	[A1,V1]     = transform_var(A0,V0);                    % transform model to decorrelated-residuals form
	[A,C,K]     = var_to_ss(A1);                           % equivalent ISS model
% 	if isempty(fres)
% 		[fres,ierr] = var2fres(A1,V1);
% 	end
	CAK         = A1;                                      % CAK sequence for pre-optimisation
	H           = var2trfun(A1,fres);                      % transfer function
	mdescript   = sprintf('%d-variable VAR(%d)',n,morder);
else
    stateDim    = 3*n                                      % hidden state space dimenion
	[A0,C0,K0]  = iss_rand(n, stateDim, rho);                % random ISS model
    V0          = corr_rand(n,rmii);                       % residuals covariance matrix
	gcSS        = ss_to_pwcgc(A0,C0,K0,V0);                % causal graph
	[A,C,K,V]   = transform_ss(A0,C0,K0,V0);               % transform model to decorrelated-residuals form
% 	if isempty(fres)
% 		[fres,ierr] = ss2fres(A,C,K,V);
% 	end
	CAK         = iss2cak(A,C,K);                          % CAK sequence for pre-optimisation
	H           = ss2trfun(A,C,K,fres);                    % transfer function
	mdescript = sprintf('%d-variable ISS(%d)',n,r);
end
% fprintf('\nFrequency resolution = %d (integration error = %e)\n\n',fres,ierr);
rng_restore(rstate);    


% outputs here



end

