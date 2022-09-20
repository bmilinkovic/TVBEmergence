function F = pwcgcModel(X,varargin)
%pwcgcModel uses the varParams function and returns the pair-wise connditional Granger
%Causality analysis for a specific VAR model selection

defaultMorder = 10;
defaultRegmode = 'LWR';
defaultAlpha = 0.05;
defaultMhtc = 'FDRD';
defaultLR = true;

p = inputParser

addRequired(p, 'X');
addParameter(p, 'regmode', defaultRegmode);
addParameter(p, 'morder', defaultMorder);
addParameter(p, 'alpha', defaultAlpha);
addParameter(p, 'mhtc', defaultMhtc);
addParameter(p, 'LR', defaultLR);

parse(p, X, varargin{:});

X = p.Results.X;
morder = p.Results.morder;
regmode = p.Results.regmode;
alpha = p.Results.alpha;
mhtc = p.Results.mhtc;
LR = p.Results.LR;

% finding the correct VAR model
VAR = varParams(X, 'morder', morder, 'regmode', regmode);
disp(VAR.info)

% pwcgc analysis


A = VAR.A;
V = VAR.V;

F = var_to_pwcgc(A,V);
F(isnan(F))=0;



end

