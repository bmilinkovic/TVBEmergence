% Add TVB Emergence project directory
global tvbemergence_dir;
tvbemergence_dir = fileparts(mfile('fullpath'));
addpath(genpath(tvbemergence_dir));
fprintf('[startup setup] added TVB Emergence project directory %s\n', tvbemergence_dir);

% Add State-Space Dynamical Independence (ssdi) Toolbox
global ssdi_dir;
ssdi_dir = '/Users/borjanmilinkovic/Documents/toolboxes/ssdi';
addpath(ssdi_dir);
cd(ssdi_dir);
startup;
cd(nap_dir);
fprintf('[Emergence Pipeline startup] Added path to State-Space Dynamical Indpendence toolbox: %s\n',ssdi_dir);

