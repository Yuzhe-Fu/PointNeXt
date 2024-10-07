import torch

# 示例张量
xyz = torch.tensor([[[1, 2, 3], [4, 5, 6], [7, 8, 9], [2, 3, 4], [5, 6, 7]]])
xyz_L1 = torch.tensor([[[7, 8, 9], [5, 6, 7], [1, 2, 3]]])
indx_xyz_L1 = torch.tensor([[[1, 2, 1], [0, 2, 1]]])

# 创建一个映射，将 xyz_L1 中的点映射到 xyz 中的索引
mapping = {}
for i in range(xyz.size(1)):
    point = tuple(xyz[0, i].tolist())
    mapping[point] = i

# 使用这个映射来转换 indx_xyz_L1 中的索引
indx_xyz = torch.zeros_like(indx_xyz_L1)
for i in range(indx_xyz_L1.size(1)):
    for j in range(indx_xyz_L1.size(2)):
        point = tuple(xyz_L1[0, indx_xyz_L1[0, i, j]].tolist())
        indx_xyz[0, i, j] = mapping[point]

print(indx_xyz)