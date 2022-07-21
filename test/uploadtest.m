% test for my SJ3D nodes

% loading data
load '/Users/borjanmilinkovic/PycharmProjects/tvb/results/3nodesj3d_1.mat'

VAR = varParams(mvar, 'morder', 30);

F = pwcgcModel(mvar, 'morder', 50);