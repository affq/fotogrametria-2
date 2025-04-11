import open3d as o3d
import numpy as np
import laspy

def las_to_o3d(plik) :
    las_pcd = laspy.read(plik)
    x = las_pcd.x
    y = las_pcd.y
    z = las_pcd.z

    r = las_pcd.red / max(las_pcd.red)
    g = las_pcd.green / max(las_pcd.green)
    b = las_pcd.blue / max(las_pcd.blue)

    las_points = np.vstack((x,y,z)).transpose()
    las_colors = np.vstack((r,g,b)).transpose()

    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(las_points)
    point_cloud.colors = o3d.utility.Vector3dVector(las_colors)
    return point_cloud

def find_outliers(point_cloud, neighbours = 1000, std_ratio = 3.0):
    filtered_point_cloud, ind = point_cloud.remove_statistical_outlier(nb_neighbors = neighbours, std_ratio = std_ratio)
    outliers = point_cloud.select_by_index(ind, invert = True)
    outliers.paint_uniform_color([1, 0, 0])
    return filtered_point_cloud, outliers


if __name__ == "__main__":
    plik = 'clouds\merged.las'
    point_cloud = las_to_o3d(plik)
    o3d.visualization.draw_geometries([point_cloud], window_name = "Point cloud")

    filtered_point_cloud, outliers = find_outliers(point_cloud)
    o3d.visualization.draw_geometries([filtered_point_cloud, outliers], window_name = "Point cloud with outliers")

    voxel_point_cloud = filtered_point_cloud.voxel_down_sample(voxel_size = 0.1)
    o3d.visualization.draw_geometries([voxel_point_cloud], window_name = "Voxelled point cloud")

    n_point_cloud = point_cloud.uniform_down_sample(every_k_points = 10)
    o3d.visualization.draw_geometries([n_point_cloud], window_name = "Point cloud with every n point")
    
