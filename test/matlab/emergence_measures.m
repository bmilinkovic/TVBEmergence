% This script compares Dynamical Indpendence and Strong Causal Emergence
% on different simple MVAR models.

% 
% sim_model;
% preoptimise_dd;
% optimise_dd;


% to compute the synergy and redundancy for each region

time_series = load('/Users/borjanmilinkovic/Documents/gitdir/TVBEmergence/results/SJ3D_3node_withlink_ps_gc-noise/data/SJ3D_3node_withlink_gc-0.263665_noise-0.232724.mat');
time_series = time_series.data;

source_data_dir = '/Volumes/dataSets/restEEGHealthySubjects/preprocessedData/sourceReconstructions/';
files = dir([source_data_dir '*.mat']);

% this is code for loading in the data and constructing a time-series
% properly for the synergistic and redundancy matrices. 

time_series = load([source_data_dir files(2).name]); 
time_series = permute(time_series.source_ts, [2,3,1]);
time_series = time_series(:,:); %this is a good trip to concatenate trials into one continuous time series. 

synergy_mat = zeros(size(time_series,1), size(time_series,1));
redundancy_mat = zeros(size(time_series,1), size(time_series,1));

for row = 1:size(time_series, 1)
    for col = 1:size(time_series, 1)
        if row == col
            continue
        else
        atoms = PhiIDFull([time_series(row,:); time_series(col, :)]);
        synergy_mat(row, col) = atoms.sts;
        redundancy_mat(row, col) = atoms.rtr;
        fprintf("Computing the PhiID Decomposition of row: %d and col: %d ... \n", row, col);
        end
    end
end

% Save synergy and redundancy matrices

phiid_dir = '/Users/borjanmilinkovic/Documents/gitdir/AnesthesiaProjectEmergence/results/phiid/data';
phiid_syn = '/wake_synergy_matrix';
phiid_red = '/wake_redundancy_matrix';
save([phiid_dir, phiid_syn], 'synergy_mat');
save([phiid_dir, phiid_red], 'redundancy_mat');

%% obtaining the synergy-redundancy gradient. 

gradient = floor(tiedrank(mean(synergy_mat))) - floor(tiedrank(mean(redundancy_mat)));
save([phiid_dir, '/wake_syn_red_gradient'], 'gradient');



