% This script compares Dynamical Indpendence and Strong Causal Emergence
% on different simple MVAR models.

% 
% sim_model;
% preoptimise_dd;
% optimise_dd;


% to compute the synergy and redundancy for each region
%% we need to load in the source reconstructed time series.

source_data_dir = '/Volumes/dataSets/restEEGHealthySubjects/restEEGHealthySubjects/preprocessedData/matlab_src_recon_files/';
files = dir([source_data_dir '*.mat']);


%% this is code for loading in the data and constructing a time-series
% properly for the synergistic and redundancy matrices. 

phiid_dir = '/Users/borjan/code/python/TVBEmergence/results/phiid/Idep_xtb/';

if ~exist(phiid_dir, 'dir')
    mkdir(phiid_dir);
end

for file_idx = 1:length(files)
    time_series = load([source_data_dir files(file_idx).name]); 
    time_series = permute(time_series.source_ts, [2,3,1]);
    time_series = time_series(:,1:20000); % this is a good trick to concatenate trials into one continuous time series, and also concatenates them to be the same length as files are off different length

    % time_series = load('/Users/borjan/code/python/TVBEmergence/results/SJ3D_3node_withlink_ps_gc-noise/data/SJ3D_3node_withlink_gc-0.106247_noise-0.002974.mat')
    % time_series = time_series.data

    synergy_mat = zeros(size(time_series,1), size(time_series,1));
    redundancy_mat = zeros(size(time_series,1), size(time_series,1));

    for row1 = 1:size(time_series, 1)
        for row2 = 1:size(time_series, 1)
            if row1 == row2
                continue
            else
                atoms = PhiIDFull([time_series(row1,:); time_series(row2, :)], 1, 'idep_xtb'); % can add argument to change the redundancy function, default = CCS
                sts_mat(row1, row2) = atoms.sts;
                rtr_mat(row1, row2) = atoms.rtr;
                rtx_mat(row1, row2) = atoms.rtx;
                rty_mat(row1, row2) = atoms.rty;
                rts_mat(row1, row2) = atoms.rts;
                xtr_mat(row1, row2) = atoms.xtr;
                xtx_mat(row1, row2) = atoms.xtx;
                xty_mat(row1, row2) = atoms.xty;
                xts_mat(row1, row2) = atoms.xts;
                ytr_mat(row1, row2) = atoms.ytr;
                ytx_mat(row1, row2) = atoms.ytx;
                yty_mat(row1, row2) = atoms.yty;
                yts_mat(row1, row2) = atoms.yts;
                str_mat(row1, row2) = atoms.str;
                stx_mat(row1, row2) = atoms.stx;
                sty_mat(row1, row2) = atoms.sty;

                fprintf("Computing the PhiID Decomposition of first row: %d and second row: %d ... \n", row1, row2);
            end
        end
    end

    % Save synergy and redundancy matrices

    save(fullfile(phiid_dir, [extractBefore(files(file_idx).name, 7) '_sts_mat_Idep_xtb' ]), 'sts_mat');
    save(fullfile(phiid_dir, [extractBefore(files(file_idx).name, 7) '_rtr_mat_Idep_xtb' ]), 'rtr_mat');
    save(fullfile(phiid_dir, [extractBefore(files(file_idx).name, 7) '_rtx_mat_Idep_xtb' ]), 'rtx_mat');
    save(fullfile(phiid_dir, [extractBefore(files(file_idx).name, 7) '_rty_mat_Idep_xtb' ]), 'rty_mat');
    save(fullfile(phiid_dir, [extractBefore(files(file_idx).name, 7) '_rts_mat_Idep_xtb' ]), 'rts_mat');
    save(fullfile(phiid_dir, [extractBefore(files(file_idx).name, 7) '_xtr_mat_Idep_xtb' ]), 'xtr_mat');
    save(fullfile(phiid_dir, [extractBefore(files(file_idx).name, 7) '_xtx_mat_Idep_xtb' ]), 'xtx_mat');
    save(fullfile(phiid_dir, [extractBefore(files(file_idx).name, 7) '_xty_mat_Idep_xtb' ]), 'xty_mat');
    save(fullfile(phiid_dir, [extractBefore(files(file_idx).name, 7) '_xts_mat_Idep_xtb' ]), 'xts_mat');
    save(fullfile(phiid_dir, [extractBefore(files(file_idx).name, 7) '_ytr_mat_Idep_xtb' ]), 'ytr_mat');
    save(fullfile(phiid_dir, [extractBefore(files(file_idx).name, 7) '_ytx_mat_Idep_xtb' ]), 'ytx_mat');
    save(fullfile(phiid_dir, [extractBefore(files(file_idx).name, 7) '_yty_mat_Idep_xtb' ]), 'yty_mat');
    save(fullfile(phiid_dir, [extractBefore(files(file_idx).name, 7) '_yts_mat_Idep_xtb' ]), 'yts_mat');
    save(fullfile(phiid_dir, [extractBefore(files(file_idx).name, 7) '_str_mat_Idep_xtb' ]), 'str_mat');
    save(fullfile(phiid_dir, [extractBefore(files(file_idx).name, 7) '_stx_mat_Idep_xtb' ]), 'stx_mat');
    save(fullfile(phiid_dir, [extractBefore(files(file_idx).name, 7) '_sty_mat_Idep_xtb' ]), 'sty_mat');

    %% obtaining the synergy-redundancy gradient. 

    gradient = floor(tiedrank(mean(sts_mat))) - floor(tiedrank(mean(rtr_mat)));
    save(fullfile(phiid_dir, [extractBefore(files(file_idx).name, 7) '_sr_gradient_Idep_xtb']), 'gradient');

    clear sts_mat rtr_mat rtx_mat rty_mat rts_mat xtr_mat xtx_mat xty_mat xts_mat ytr_mat ytx_mat yty_mat yts_mat str_mat stx_mat sty_mat
end



