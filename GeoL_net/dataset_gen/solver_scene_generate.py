import open3d as o3d
import numpy as np
import torch
from pytorch3d.loss import chamfer_distance
import torch.optim as optim
import torch.autograd as autograd

def sample_points_from_mesh(mesh_file, num_samples):
    # 从 obj 文件加载 mesh
    mesh = o3d.io.read_triangle_mesh(mesh_file)
    
    # 从 mesh 中均匀采样点
    pcd = mesh.sample_points_uniformly(number_of_points=num_samples)
    
    # 返回采样的点坐标
    return np.asarray(pcd.points)

def sample_points_from_point_cloud(ply_file, num_samples):
    # Load point cloud from ply file
    pcd = o3d.io.read_point_cloud(ply_file)
    
    # Sample points randomly from the point cloud
    sampled_indices = np.random.choice(len(pcd.points), num_samples, replace=True)
    sampled_points = np.asarray(pcd.points)[sampled_indices]
    
    # Return the sampled points
    return sampled_points

def compute_chamfer_distance(points1, points2, angle_degrees):
    angle_radians = torch.deg2rad(angle_degrees)
    #print("angle_radians.requires_grad:", angle_radians.requires_grad) 
    points1 = torch.from_numpy(points1)
    points2 = torch.from_numpy(points2)
    points1 = points1.to(torch.float32).requires_grad_(True)
    points2 = points2.to(torch.float32).requires_grad_(True)
    #print('type of points1', type(points1))    
    # Define rotation matrix around z-axis
    cos_angle = torch.cos(angle_radians)
    sin_angle = torch.sin(angle_radians)

    R = torch.stack([torch.stack([cos_angle, -sin_angle, torch.tensor(0.0)], dim=0),
                     torch.stack([sin_angle, cos_angle, torch.tensor(0.0)], dim=0),
                     torch.stack([torch.tensor(0.0), torch.tensor(0.0), torch.tensor(1.0)], dim=0)], dim=0)

    # Rotate points1 around z-axis
    rotated_points1 = torch.matmul(points1, R.T)

    # Compute Chamfer distance between rotated_points1 and points2
    chamfer_loss, _ = chamfer_distance(rotated_points1.unsqueeze(0), points2.unsqueeze(0))

    
    return chamfer_loss

def compute_chamfer_distance_batch(points1_list, points2_list, angle_degrees):
    angle_radians = torch.deg2rad(angle_degrees).unsqueeze(-1) # B*N*1
    #print("angle_radians.requires_grad:", angle_radians.requires_grad) 
    points1_list = torch.tensor(points1_list)
    points2_list = torch.tensor(points2_list)
    points1_list = points1_list.to(torch.float32).requires_grad_(True)
    points2_list = points2_list.to(torch.float32).requires_grad_(True)
    #print('type of points1', type(points1))    
    # Define rotation matrix around z-axis
    cos_angles = torch.cos(angle_radians)
    sin_angles = torch.sin(angle_radians)

    R = torch.stack([
        torch.stack([cos_angles, -sin_angles, torch.zeros_like(cos_angles)], dim=-1),
        torch.stack([sin_angles,  cos_angles, torch.zeros_like(sin_angles)], dim=-1),
        torch.stack([torch.zeros_like(cos_angles), torch.zeros_like(cos_angles), torch.ones_like(cos_angles)], dim=-1)
    ], dim=-2)  # Shape will be (B, N, 3, 3)

    # Rotate points1 around z-axis
    #rotated_points1_list = torch.matmul(points1_list, R.T)
    rotated_points1_list = torch.einsum('bnij,bnj->bni', R, points1_list)

    # Compute Chamfer distance between rotated_points1 and points2
    chamfer_loss, _ = chamfer_distance(rotated_points1_list, points2_list)

    
    return chamfer_loss 






def optimize_rotation(meshes, pcds, initial_angle, num_iterations=2000, learning_rate=1):
    # Initialize angle as a PyTorch tensor

    B = len(meshes)
    #angle = torch.tensor(initial_angle, dtype=torch.float32, requires_grad=True)
    #angles = torch.zeros(B, requires_grad=True)
    angles = torch.full((B,), initial_angle, dtype=torch.float32, requires_grad=True)
    #print("angle.requires_grad:", angle.requires_grad)
    # Define optimizer
    optimizer = optim.Adam([angles], lr=learning_rate)
    
    # Optimization loop
    for iter in range(num_iterations):
        chamfer_sum = 0

        #chamfer_loss = chamfer_distance(meshes, pcds)
        for i in range(B):
            # TODO: parallelly
            chamfer_dist = compute_chamfer_distance(meshes[i], pcds[i], angles[i])
            chamfer_sum += chamfer_dist

        optimizer.zero_grad()

            # Backpropagation
        chamfer_sum.backward()
        
            # Update angle
        optimizer.step()
        
            # Print progress
        if (iter+1) % 100 == 0:
            print("Iteration {}: Chamfer Distance = {}, Angle = {} degrees".format(iter+1, chamfer_sum, angles))
            print("angle grad:", angles.grad)


    # Return the optimized rotation angle
    return angles

def optimize_rotation_batch(meshes, pcds, initial_angle, num_iterations=2000, learning_rate=1):
    # Initialize angle as a PyTorch tensor

    B = len(meshes)
    #angle = torch.tensor(initial_angle, dtype=torch.float32, requires_grad=True)
    #angles = torch.zeros(B, requires_grad=True)
    angles = torch.full((B,), initial_angle, dtype=torch.float32, requires_grad=True)
    #print("angle.requires_grad:", angle.requires_grad)
    # Define optimizer
    optimizer = optim.Adam([angles], lr=learning_rate)
    
    # Optimization loop
    for iter in range(num_iterations):

        #chamfer_loss = chamfer_distance(meshes, pcds)
        chamfer_dist = compute_chamfer_distance_batch(meshes, pcds, angles)


        optimizer.zero_grad()

            # Backpropagation
        chamfer_dist.backward()
        
            # Update angle
        optimizer.step()
        
            # Print progress
        if (iter+1) % 1 == 0:
            print("Iteration {}: Chamfer Distance = {}, Angle = {} degrees".format(iter+1, chamfer_dist, angles))
            print("angle grad:", angles.grad)


    # Return the optimized rotation angle
    return angles

if __name__ == "__main__":
    mesh_files = ["display.obj", "display.obj"]
    ply_files = ["small_dataset_of/dataset_obj/laptop.ply","small_dataset_of/dataset_obj/laptop.ply"]

# 从 mesh 中采样点
    sampled_meshes = [sample_points_from_mesh(mesh_file, num_samples=512) for mesh_file in mesh_files]

    # 从点云文件中加载点云数据
    # Sample points from point cloud
    sampled_pcds = [sample_points_from_point_cloud(ply_file, num_samples=512) for ply_file in ply_files]
    #print("shape of points", sampled_meshes.shape)


    # 初始化角度为0
    initial_angle = 0

    # 调用优化器找到使得 Chamfer 距离最小的角度
    optimal_angle = optimize_rotation(sampled_meshes, sampled_pcds, initial_angle=initial_angle, learning_rate=0.1)
    print("Optimal rotation angle:", optimal_angle, "degrees")