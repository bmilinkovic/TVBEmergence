# preprocessing of data, this intialises an array, which will have the shape
# [time-points, nodes, coupling range], it will sum over the modes of each node,
# then, it will zscore the data to demean it. This has been done in previous research
# to prepare the data for further analysis. Importantly it takes the _xi_ variable,
# which is taken as the excitatory state variable representing the local field potential
# of the node. THE CODE BELOW IS WRONG AND IS NOT WORKING, might not be wrong, my code for simulating the
# SJ3D might be not numerically stable.
# array = np.empty(shape=(2060, 5, 2))
# dataz = []
# for i in range(0, 2, 1):
#     dataz.append(data[i][1][500:, 0, :, :])
#     for j in range(0, 5, 1):
#         np.append(array[:, j, i], zscore(np.sum(dataz[:, j, :], axis=1)))
#
# plt.plot(array[:,:,1])
# plt.show()

# data_short = []
# for i in range(len(data)):
#     data_short.append(data[i][1][500:, 0, :, :])


# Here below I would like to put this in a simpler code to loop over, but I cannot seem to manage to do that as I'm
# getting a little confused with the nested indexing.

# # pull the data out for each coupling range
# gc1 = data[0][1]
# # zscore the data
# gc11_zscored = zscore(np.sum(gc1[500:,0,0,:], axis=1))
# gc12_zscored = zscore(np.sum(gc1[500:,0,1,:], axis=1))
# gc13_zscored = zscore(np.sum(gc1[500:,0,2,:], axis=1))
# gc14_zscored = zscore(np.sum(gc1[500:,0,3,:], axis=1))
# gc15_zscored = zscore(np.sum(gc1[500:,0,4,:], axis=1))
# # stack the data
# gc1full =  np.c_[gc11_zscored,gc12_zscored, gc13_zscored, gc14_zscored, gc15_zscored]
#
# gc2 = data[1][1]
# gc21_zscored = zscore(np.sum(gc2[500:,0,0,:], axis=1))
# gc22_zscored = zscore(np.sum(gc2[500:,0,1,:], axis=1))
# gc23_zscored = zscore(np.sum(gc2[500:,0,2,:], axis=1))
# gc24_zscored = zscore(np.sum(gc2[500:,0,3,:], axis=1))
# gc25_zscored = zscore(np.sum(gc2[500:,0,4,:], axis=1))
# gc2full =  np.c_[gc21_zscored,gc22_zscored, gc23_zscored, gc24_zscored, gc25_zscored]
#
# gc3 = data[2][1]
# # zscore the data
# gc31_zscored = zscore(np.sum(gc3[500:,0,0,:], axis=1))
# gc32_zscored = zscore(np.sum(gc3[500:,0,1,:], axis=1))
# gc33_zscored = zscore(np.sum(gc3[500:,0,2,:], axis=1))
# gc34_zscored = zscore(np.sum(gc3[500:,0,3,:], axis=1))
# gc35_zscored = zscore(np.sum(gc3[500:,0,4,:], axis=1))
# # stack the data
# gc3full =  np.c_[gc31_zscored,gc32_zscored, gc33_zscored, gc34_zscored, gc35_zscored]
#
# gc4 = data[3][1]
# # zscore the data
# gc41_zscored = zscore(np.sum(gc4[500:,0,0,:], axis=1))
# gc42_zscored = zscore(np.sum(gc4[500:,0,1,:], axis=1))
# gc43_zscored = zscore(np.sum(gc4[500:,0,2,:], axis=1))
# gc44_zscored = zscore(np.sum(gc4[500:,0,3,:], axis=1))
# gc45_zscored = zscore(np.sum(gc4[500:,0,4,:], axis=1))
# # stack the data
# gc4full =  np.c_[gc41_zscored,gc42_zscored, gc43_zscored, gc44_zscored, gc45_zscored]
#
# # plotting here
#
# plt.plot(gc1full)
# plt.show()
# plt.plot(gc2full)
# plt.show()
# plt.plot(gc3full)
# plt.show()
# plt.plot(gc4full)
# plt.show()