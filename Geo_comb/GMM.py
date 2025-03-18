import numpy as np
from sklearn.mixture import GaussianMixture
import open3d as o3d
from matplotlib import cm

def fit_2d_gmm(xy, value):
    num_clusters, num_points, dim = xy.shape
    #xy_flattened = xy.reshape(num_clusters*num_points, dim)
    #value_flatttened = value.reshape(num_clusters*num_points)
    xy_flattened = xy[0].reshape(num_points, dim)
    value_flattened = np.sum(value, axis=0, keepdims=True)[0]
    # fit GMM
    gmm = GaussianMixture(n_components=num_clusters+1, covariance_type='full', random_state=42)
    gmm.fit(xy_flattened, value_flattened)

    return gmm

def sample_gmm_values(gmm, xy):
    num_points, dim = xy.shape
    values = np.zeros((1, num_points, 1))


    log_prob = gmm.score_samples(xy)
    values[0, :, 0] = np.exp(log_prob)
    return values

if __name__ == "__main__":
    pc_path_1 = "dataset/scene_RGBD_mask_v2_kinect_cfg/id747_1/cup_0001_red/mask_Front.ply"
    pc_path_2 = "dataset/scene_RGBD_mask_v2_kinect_cfg/id747_1/cup_0001_red/mask_Behind.ply"
    pc_path_3 = "dataset/scene_RGBD_mask_v2_kinect_cfg/id747_1/cup_0001_red/mask_Left.ply"
    pc_1 = o3d.io.read_point_cloud(pc_path_1)
    pc_2 = o3d.io.read_point_cloud(pc_path_2)
    pc_3 = o3d.io.read_point_cloud(pc_path_3)
    pc_points = np.asarray(pc_1.points)
    pc_points_1 = np.asarray(pc_1.points)[..., :2]
    pc_points_2 = np.asarray(pc_2.points)[..., :2]
    pc_points_3 = np.asarray(pc_3.points)[..., :2]
    value_1 = np.asarray(pc_1.colors)[:,1][:, None]
    value_2 = np.asarray(pc_2.colors)[:,1][:, None]
    value_3 = np.asarray(pc_3.colors)[:,1][:, None]

    xy = np.stack([pc_points_1, pc_points_2, pc_points_3], axis=0)
    value = np.stack([value_1, value_2, value_3], axis=0)

    num_clusters, num_points, dim = xy.shape

   #xy = np.random.rand(num_clusters, 2048, 2)
   #value = np.random.rand(num_clusters, 2048, 1)

    gmm_model = fit_2d_gmm(xy, value)
    #sampled_values = sample_gmm_values(gmm_model, xy[0]) # sample from gmm
    sampled_values = np.sum(value, axis=0, keepdims=True)
    # normalize the sampled values
    sampled_values = (sampled_values - sampled_values.min()) / (sampled_values.max() - sampled_values.min())

    # visualize the value 
    rgb_colors = np.zeros((num_clusters, num_points, 3))

    rgb_colors[0] = cm.turbo(sampled_values[0, :, 0])[:, :3]


    pc = o3d.geometry.PointCloud()
    pc.points = o3d.utility.Vector3dVector(pc_points)
    pc.colors = o3d.utility.Vector3dVector(rgb_colors[0])
    o3d.visualization.draw_geometries([pc])


    print("sampled_values shape", sampled_values.shape)

