function VAR = varParams(X,varargin)
%PWCGCANALYSIS Returnes VAR parameters from input time-series X

defaultMorder = 10;
defaultRegmode = 'LWR';

p = inputParser;

addRequired(p, 'X');
addParameter(p, 'regmode', defaultRegmode);
addParameter(p, 'morder', defaultMorder);

parse(p, X, varargin{:});

X = p.Results.X;
morder = p.Results.morder;
regmode = p.Results.regmode;

% The actual analysis
[VAR.A, VAR.V] = tsdata_to_var(X, morder, regmode);
VAR.info = var_info(VAR.A, VAR.V);

end

