global emergenceDir;
emergenceDir = fileparts(mfilename('fullpath'));
addpath(emergenceDir);

addpath(fullfile(emergenceDir, 'deprecated'));
addpath(genpath(fullfile(emergenceDir, 'networks')));
addpath(genpath(fullfile(emergenceDir, 'plots')));
addpath(fullfile(emergenceDir, 'simulation'));
addpath(fullfile(emergenceDir, 'utils'));
addpath(fullfile(emergenceDir, 'test'));
addpath(fullfile(emergenceDir, 'test/matlab'));
addpath(fullfile(emergenceDir, 'utils'));

ssdi_path = getenv('SSDI');
assert(exist(ssdi_path,'dir') == 7,'bad SSDI path: ''%s'' does not exist or is not a directory',ssdi_path);
run(fullfile(ssdi_path,'startup'));
fprintf('[Emergence Pipeline startup] Added path to State-Space Dynamical Indpendence toolbox: %s\n',ssdi_path);


% global ssdiDir;
% ssdiDir = '/Users/borjanmilinkovic/Documents/gitdir/ssdi';
% addpath((ssdiDir));
% cd(ssdiDir);
% startup;
% cd(emergenceDir);


global ceDir;
ceDir = getenv('PHIID');
addpath(genpath(ceDir));
fprintf('[Emergence Pipeline startup] Phi-ID toolbox intialised at: %s\n',ceDir);





