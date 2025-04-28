import open3d as o3d
import numpy as np
import laspy
import copy

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

def find_outliers(point_cloud, neighbours = 30, std_ratio = 2.0):
    filtered_point_cloud, ind = point_cloud.remove_statistical_outlier(nb_neighbors = neighbours, std_ratio = std_ratio)
    outliers = point_cloud.select_by_index(ind, invert = True)
    outliers.paint_uniform_color([1, 0, 0])
    return filtered_point_cloud, outliers

def visualize_point_clouds(reference_cloud, oriented_cloud, transformation):
    ori_temp = copy.deepcopy(oriented_cloud)
    ref_temp = copy.deepcopy(reference_cloud)
    ori_temp.paint_uniform_color([1, 0, 0])
    ref_temp.paint_uniform_color([0, 1, 0])
    ori_temp.transform(transformation)
    o3d.visualization.draw_geometries([ori_temp, ref_temp])

def measure_points(point_cloud):
    print("Pomiar punktów na chmurze punktów")
    print("Etapy pomiaru punktów: ")
    print(" (1.1) Pomiar punktu - shift + lewy przycisk myszy")
    print(" (1.2) Cofniecie ostatniego pomiaru - shift + prawy przycisk myszy")
    print(" (2) Koniec pomiaru - wciśnięcie klawisza Q")
    vis = o3d.visualization.VisualizerWithEditing()
    vis.create_window(window_name='Pomiar punktów')
    vis.add_geometry(point_cloud)
    vis.run()
    vis.destroy_window()
    print("Koniec pomiaru")
    print(vis.get_picked_points())
    return vis.get_picked_points()

def target_based_orientation(chmura_referencyjna, chmura_orientowana): 
    print('Orientacja chmur punktów metoda Target based') 
    visualize_point_clouds(chmura_referencyjna, chmura_orientowana,np.identity(4)) 

    print('Pomierz min. 3 punkty na chmurze referencyjnej: ') 
    pkt_ref = measure_points(chmura_referencyjna) 
    print('Pomierz min. 3 punkty orientowanej ') 
    pkt_ori = measure_points(chmura_orientowana)
    
    assert (len(pkt_ref) >= 3 and len(pkt_ori) >= 3) 
    assert (len(pkt_ref) == len(pkt_ori))

    wsp_pkt_ref = np.asarray(chmura_referencyjna.points)[pkt_ref] 
    wsp_pkt_ori = np.asarray(chmura_orientowana.points)[pkt_ori] 
    pcd_ref = o3d.geometry.PointCloud() 
    pcd_ref.points = o3d.utility.Vector3dVector(wsp_pkt_ref) 
    pcd_ori = o3d.geometry.PointCloud() 
    pcd_ori.points = o3d.utility.Vector3dVector(wsp_pkt_ori) 
    corr = np.asarray([[i, i] for i in range(len(wsp_pkt_ref))]) 
    p2p = o3d.pipelines.registration.TransformationEstimationPointToPoint() 
    trans = p2p.compute_transformation(pcd_ref, pcd_ori, o3d.utility.Vector2iVector(corr)) 
    
    return trans

def icp_orientation(source, target, threshold = 1.0, trans_init = np.identity(4)):
    print('Analiza dokładności wstępnej orientacji')
    evaluation = o3d.pipelines.registration.evaluate_registration(source, target, threshold, trans_init)
    print(evaluation)
    print("Orientacja ICP <Punkt do punktu>")
    reg_p2p = o3d.pipelines.registration.registration_icp(source, target, threshold, trans_init, o3d.pipelines.registration.TransformationEstimationPointToPoint())
    print(reg_p2p)
    print("Macierz transformacji:")
    print(reg_p2p.transformation)
    visualize_point_clouds(source, target, reg_p2p.transformation)
    information_reg_p2p = o3d.pipelines.registration.get_information_matrix_from_point_clouds(source, target, threshold, reg_p2p.transformation)
    return reg_p2p.transformation, information_reg_p2p

def calculate_normals(point_cloud):
    point_cloud.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.05, max_nn=30)
    )
    point_cloud.orient_normals_consistent_tangent_plane(100)
    point_cloud.normals = o3d.utility.Vector3dVector(-np.asarray(point_cloud.normals))
    return point_cloud


def ball_pivoting(point_cloud, promienie_kul = [0.1, 0.2, 0.4, 0.8]):
    point_cloud_with_normals = calculate_normals(point_cloud)
    tin = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(point_cloud_with_normals, o3d.utility.DoubleVector(promienie_kul))
    o3d.visualization.draw_geometries([point_cloud_with_normals, tin])
    return tin

def poisson(point_cloud):
    point_cloud = calculate_normals(point_cloud)
    o3d.visualization.draw_geometries([point_cloud], point_show_normal=True)
    with o3d.utility.VerbosityContextManager(o3d.utility.VerbosityLevel.Debug) as cm:
        tin, gestosc = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(point_cloud, depth=15)
    o3d.visualization.draw_geometries([tin])
    return tin, gestosc

def display_density(gestosc, tin):
    gestosc = np.asarray(gestosc)
    gestosc_colors = plt.get_cmap('plasma')((gestosc - gestosc.min()) / (gestosc.max() - gestosc.min()))
    gestosc_colors = gestosc_colors[:, :3]
    gestosc_mesh = o3d.geometry.TriangleMesh()
    gestosc_mesh.vertices = tin.vertices
    gestosc_mesh.triangles = tin.triangles
    gestosc_mesh.triangle_normals = tin.triangle_normals
    gestosc_mesh.vertex_colors = o3d.utility.Vector3dVector(gestosc_colors)
    o3d.visualization.draw_geometries([gestosc_mesh])

def filter_model_by_density(tin, gestosc, kwantyl = 0.01):
    vertices_to_remove = gestosc < np.quantile(gestosc, kwantyl)
    tin.remove_vertices_by_mask(vertices_to_remove)
    o3d.visualization.draw_geometries([tin])

def poisson_filtering(point_cloud):
    tin, density = poisson(point_cloud)
    display_density(density, tin)
    filter_model_by_density(tin, density, kwantyl = 0.01)
    return tin

reference = f'clouds\orig.laz'
results_reference = f'results\\reference'
reference_filtered = f'{results_reference}\\reference_filtered.pcd'
reference_voxel = f'{results_reference}\\reference_voxel.pcd'
reference_uniform = f'{results_reference}\\reference_uniform.pcd'

merged = f'clouds\merged.las'
results_merged = f'results\merged'
merged_filtered = f'{results_merged}\\merged_filtered.pcd'
merged_voxel = f'{results_merged}\\merged_voxel.pcd'
merged_uniform = f'{results_merged}\\merged_uniform.pcd'

'''
n_point_cloud = point_cloud.uniform_down_sample(every_k_points = 10)
# o3d.visualization.draw_geometries([n_point_cloud], window_name = "Point cloud with every n point")
o3d.io.write_point_cloud(merged_uniform, n_point_cloud)



n_point_cloud = point_cloud.uniform_down_sample(every_k_points = 10)
# o3d.visualization.draw_geometries([n_point_cloud], window_name = "Point cloud with every n point")
o3d.io.write_point_cloud(reference_uniform, n_point_cloud)

model = ball_pivoting(reference_voxel_as_o3d)
o3d.io.write_triangle_mesh(f"{results_reference}\\reference_model_pivot.ply", model)
'''

import matplotlib.pyplot as plt


# point_cloud = las_to_o3d(merged)
# filtered_point_cloud, outliers = find_outliers(point_cloud)
# voxel_point_cloud = filtered_point_cloud.voxel_down_sample(voxel_size = 0.1)
# o3d.io.write_point_cloud(merged_voxel, voxel_point_cloud)


# ref_point_cloud = las_to_o3d(reference)
# ref_filtered_point_cloud, ref_outliers = find_outliers(ref_point_cloud)
# ref_voxel_point_cloud = ref_filtered_point_cloud.voxel_down_sample(voxel_size = 0.1)
# o3d.io.write_point_cloud(reference_voxel, ref_voxel_point_cloud)


ref_voxel_point_cloud = o3d.io.read_point_cloud(reference_voxel)
voxel_point_cloud = o3d.io.read_point_cloud(merged_voxel)

# trans = target_based_orientation(ref_voxel_point_cloud, voxel_point_cloud)
# print(trans)


trans_init = np.array([
    [ 8.71959959e-02,  9.96190054e-01,  1.49507918e-03, -5.26069470e+05],
    [-9.96189790e-01,  8.71932734e-02,  1.79866705e-03,  5.95557058e+05],
    [ 1.66145337e-03, -1.64621918e-03,  9.99997265e-01, -4.27401737e+02],
    [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  1.00000000e+00]
])

# icp_result, information_reg_p2p = icp_orientation(voxel_point_cloud, ref_voxel_point_cloud, threshold=1.0, trans_init=trans_init)

def combine_point_clouds(ref_cloud, oriented_cloud, transformation):
    ori_temp = copy.deepcopy(oriented_cloud)
    ref_temp = copy.deepcopy(ref_cloud)
    ori_temp.transform(transformation)
    combined = ref_temp + ori_temp
    return combined

# trans_init_inv = np.linalg.inv(trans_init)
# combined_point_cloud = combine_point_clouds(ref_voxel_point_cloud, voxel_point_cloud, trans_init_inv)
# o3d.visualization.draw_geometries([combined_point_cloud])


pc = ball_pivoting(voxel_point_cloud, [0.5])
o3d.io.write_triangle_mesh(f"{results_merged}\\merged_model_pivot.ply", pc)

pc = poisson_filtering(voxel_point_cloud)
o3d.io.write_triangle_mesh(f"{results_merged}\\merged_model_poisson.ply", pc)