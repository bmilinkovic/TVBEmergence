% This script compares Dynamical Indpendence and Strong Causal Emergence
% on different simple MVAR models.

% 
% sim_model;
% preoptimise_dd;
% optimise_dd;


% to compute the synergy and redundancy for each region
% we need to load in the source reconstructed time series.

source_data_dir = '/Volumes/dataSets/restEEGHealthySubjects/restEEGHealthySubjects/preprocessedData/matlab_src_recon_files/';
files = dir([source_data_dir '*.mat']);

% this is code for loading in the data and constructing a time-series
% properly for the synergistic and redundancy matrices. 

phiid_dir = '/Users/borjan/code/python/TVBEmergence/results/phiid/';

if ~exist(phiid_dir, 'dir')
    mkdir(phiid_dir);
end

for file_idx = 1:length(files)
    time_series = load([source_data_dir files(file_idx).name]); 
    time_series = permute(time_series.source_ts, [2,3,1]);
    time_series = time_series(:,:); %this is a good trick to concatenate trials into one continuous time series. 

    % time_series = load('/Users/borjan/code/python/TVBEmergence/results/SJ3D_3node_withlink_ps_gc-noise/data/SJ3D_3node_withlink_gc-0.106247_noise-0.002974.mat')
    % time_series = time_series.data

    synergy_mat = zeros(size(time_series,1), size(time_series,1));
    redundancy_mat = zeros(size(time_series,1), size(time_series,1));

    for row1 = 1:size(time_series, 1)
        for row2 = 1:size(time_series, 1)
            if row1 == row2
                continue
            else
                atoms = PhiIDFull([time_series(row1,:); time_series(row2, :)]);
                sts_mat(row1, row2) = atoms.sts;
                rtr_mat(row1, row2) = atoms.rtr;
                fprintf("Computing the PhiID Decomposition of first row: %d and second row: %d ... \n", row1, row2);
            end
        end
    end

    % Save synergy and redundancy matrices

    save(fullfile(phiid_dir, [extractBefore(files(file_idx).name, 7) '_sts_mat' ]), 'sts_mat');
    save(fullfile(phiid_dir, [extractBefore(files(file_idx).name, 7) '_rtr_mat' ]), 'rtr_mat');

    %% obtaining the synergy-redundancy gradient. 

    gradient = floor(tiedrank(mean(synergy_mat))) - floor(tiedrank(mean(redundancy_mat)));
    save(fullfile(phiid_dir, [extractBefore(files(file_idx).name, 7) '_sr_gradient']), 'gradient');
end



