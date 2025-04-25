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

def target_based_orientation(reference_cloud, oriented_cloud, typ = 'Pomiar', Debug = 'False'):
    if typ == 'Pomiar':
        print('Pomierz min. 3 punkty na chmurze referencyjnej.')
        pkt_ref = measure_points(reference_cloud)
        print('Pomierz min. 3 punkty orientowanej.')
        pkt_ori = measure_points(oriented_cloud)
    elif typ == 'Plik':
        print('Wyznaczenia parametrów transformacji na podstawie punktów pozyskanych z plików tekstowych')
    else:
        print('Wyznaczenie parametrów na podstawie analizy deskryptorów')

    assert (len(pkt_ref) >= 3 and len(pkt_ori) >= 3)
    assert (len(pkt_ref) == len(pkt_ori))

    corr = np.zeros((len(pkt_ori), 2))
    corr[:, 0] = pkt_ori
    corr[:, 1] = pkt_ref

    p2p = o3d.pipelines.registration.TransformationEstimationPointToPoint()
    trans = p2p.compute_transformation(reference_cloud, oriented_cloud, o3d.utility.Vector2iVector(corr))
    visualize_point_clouds(reference_cloud, oriented_cloud, trans)

    if Debug == 'True':
        print(trans)
        visualize_point_clouds(reference_cloud, oriented_cloud, trans)
    
    return trans

def icp_orientation(source, target, threshold = 1.0, trans_init = np.identity(4), metoda = 'p2p'):
    print('Analiza dokładności wstępnej orientacji')
    evaluation = o3d.pipelines.registration.evaluate_registration(source, target, threshold, trans_init)
    print(evaluation)
    if metoda == 'p2p':
        print("Orientacja ICP <Punkt do punktu>")
        reg_p2p = o3d.pipelines.registration.registration_icp(source, target, threshold, trans_init, o3d.pipelines.registration.TransformationEstimationPointToPoint())
        print(reg_p2p)
        print("Macierz transformacji:")
        print(reg_p2p.transformation)
        visualize_point_clouds(source, target, reg_p2p.transformation)
        information_reg_p2p = o3d.pipelines.registration.get_information_matrix_from_point_clouds(source, target, threshold, reg_p2p.transformation)
        return reg_p2p.transformation, information_reg_p2p
    elif metoda == 'p2pl':
        print('Wyznaczanie normalnych')
        source.normals = o3d.utility.Vector3dVector(np.zeros((1, 3))) # Jeżeli istnieją normalne to są zerowane
        source.estimate_normals()
        target.normals = o3d.utility.Vector3dVector(np.zeros((1, 3))) # Jeżeli istnieją normalne to są zerowane
        target.estimate_normals()
        print("Orientacja ICP <Punkt do płaszczyzny>")
        reg_p2pl = o3d.pipelines.registration.registration_icp(source, target, threshold, trans_init, o3d.pipelines.registration.TransformationEstimationPointToPlane())
        print(reg_p2pl)
        print("Macierz transformacji:")
        print(reg_p2pl.transformation)
        visualize_point_clouds(source, target, reg_p2pl.transformation)
        information_reg_p2pl = o3d.pipelines.registration.get_information_matrix_from_point_clouds(source, target, threshold, reg_p2pl.transformation)
        return reg_p2pl.transformation,information_reg_p2pl
    elif metoda == 'cicp':
        reg_cicp = o3d.pipelines.registration.registration_colored_icp(source, target, threshold, trans_init)
        print(reg_cicp)
        print("Macierz transformacji:")
        print(reg_cicp.transformation)
        visualize_point_clouds(source, target, reg_cicp.transformation)
        information_reg_cicp = o3d.pipelines.registration.get_information_matrix_from_point_clouds(source, target, threshold, reg_cicp.transformation)
        return reg_cicp.transformation, information_reg_cicp
    else:
        print('Nie wybrano odpowiedniego sposobu transformacji')

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
point_cloud = las_to_o3d(merged)
# o3d.visualization.draw_geometries([point_cloud], window_name = "Point cloud")

filtered_point_cloud, outliers = find_outliers(point_cloud)
# o3d.visualization.draw_geometries([filtered_point_cloud, outliers], window_name = "Point cloud with outliers")
both = filtered_point_cloud + outliers
o3d.io.write_point_cloud(merged_filtered, both)

voxel_point_cloud = filtered_point_cloud.voxel_down_sample(voxel_size = 0.1)
# o3d.visualization.draw_geometries([voxel_point_cloud], window_name = "Voxelled point cloud")
o3d.io.write_point_cloud(merged_voxel, voxel_point_cloud)

n_point_cloud = point_cloud.uniform_down_sample(every_k_points = 10)
# o3d.visualization.draw_geometries([n_point_cloud], window_name = "Point cloud with every n point")
o3d.io.write_point_cloud(merged_uniform, n_point_cloud)




point_cloud = las_to_o3d(reference)
# o3d.visualization.draw_geometries([point_cloud], window_name = "Point cloud")

filtered_point_cloud, outliers = find_outliers(point_cloud)
# o3d.visualization.draw_geometries([filtered_point_cloud, outliers], window_name = "Point cloud with outliers")
both = filtered_point_cloud + outliers
o3d.io.write_point_cloud(reference_filtered, both)

voxel_point_cloud = filtered_point_cloud.voxel_down_sample(voxel_size = 0.1)
# o3d.visualization.draw_geometries([voxel_point_cloud], window_name = "Voxelled point cloud")
o3d.io.write_point_cloud(reference_voxel, voxel_point_cloud)

n_point_cloud = point_cloud.uniform_down_sample(every_k_points = 10)
# o3d.visualization.draw_geometries([n_point_cloud], window_name = "Point cloud with every n point")
o3d.io.write_point_cloud(reference_uniform, n_point_cloud)
'''
reference_voxel_as_o3d = o3d.io.read_point_cloud(reference_voxel)
merged_voxel_as_o3d = o3d.io.read_point_cloud(merged_voxel)
'''
target_based_orientation_result = target_based_orientation(reference_voxel_as_o3d, merged_voxel_as_o3d)
icp_orientation_result = icp_orientation(reference_voxel_as_o3d, merged_voxel_as_o3d)
'''

def calculate_normals(point_cloud):
    point_cloud.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.05, max_nn=30)
    )
    point_cloud.orient_normals_consistent_tangent_plane(100)
    point_cloud.normals = o3d.utility.Vector3dVector(-np.asarray(point_cloud.normals))
    return point_cloud


def ball_pivoting(point_cloud, promienie_kul = [0.005, 0.01, 0.02, 0.04]):
    point_cloud_with_normals = calculate_normals(point_cloud)
    tin = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(point_cloud_with_normals, o3d.utility.DoubleVector(promienie_kul))
    o3d.visualization.draw_geometries([point_cloud_with_normals, tin])
    return tin

'''
model = ball_pivoting(merged_voxel_as_o3d)
o3d.io.write_triangle_mesh(f"{results_merged}\merged_model_pivot.ply", model)

model = ball_pivoting(reference_voxel_as_o3d)
o3d.io.write_triangle_mesh(f"{results_reference}\\reference_model_pivot.ply", model)
'''

def poisson(point_cloud):
    point_cloud = calculate_normals(point_cloud)
    o3d.visualization.draw_geometries([point_cloud], point_show_normal=True)
    with o3d.utility.VerbosityContextManager(o3d.utility.VerbosityLevel.Debug) as cm:
        tin, gestosc = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(point_cloud, depth=15)
    o3d.visualization.draw_geometries([tin])
    return tin, gestosc

import matplotlib.pyplot as plt

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

tin = poisson_filtering(merged_voxel_as_o3d)
o3d.io.write_triangle_mesh(f"{results_merged}\merged_model_poisson.ply", tin)