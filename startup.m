global emergenceDir;
emergenceDir = fileparts(mfilename('fullpath'));
addpath(emergenceDir);

addpath(fullfile(emergenceDir, 'deprecated'));
addpath(fullfile(emergenceDir, 'results'));
addpath(fullfile(emergenceDir, 'simsdi'));
addpath(fullfile(emergenceDir, 'src'));
addpath(fullfile(emergenceDir, 'utils'));
addpath(fullfile(emergenceDir, 'test'));

global ssdiDir;
ssdiDir = '/Users/borjanmilinkovic/Documents/matlabProjects/TVBEmergence/src/ssdi';
addpath((ssdiDir));
cd(ssdiDir);
startup;
cd(emergenceDir);
fprintf('[Emergence Pipeline startup] Added path to State-Space Dynamical Indpendence toolbox: %s\n',ssdiDir);

global ceDir;
ceDir = '/Users/borjanmilinkovic/Documents/matlabProjects/TVBEmergence/src/sce';
addpath(genpath(ceDir));
fprintf('[Emergence Pipeline startup] Added path to Strong Causal Emergence toolbox: %s\n',ceDir);





