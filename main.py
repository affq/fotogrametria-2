# Import potrzebnych bibliotek
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

    chmura_punktow = o3d.geometry.PointCloud()
    chmura_punktow.points = o3d.utility.Vector3dVector(las_points)
    chmura_punktow.colors = o3d.utility.Vector3dVector(las_colors)
    return chmura_punktow

def wyznaczanie_obserwacji_odstajacych(chmura_punktow, liczba_sasiadow = 1000, std_ratio = 3.0):
    chmura_punktow_odfiltrowana, ind = chmura_punktow.remove_statistical_outlier(nb_neighbors = liczba_sasiadow, std_ratio = std_ratio)
    punkty_odstajace = chmura_punktow.select_by_index(ind, invert = True)
    punkty_odstajace.paint_uniform_color([1, 0, 0])
    return chmura_punktow_odfiltrowana, punkty_odstajace

def regularyzacja_chmur_punktow(chmura_punktow, odleglosc_miedzy_wokselami = 0.1):
    chmura_punktow_woksele = chmura_punktow.voxel_down_sample(voxel_size = odleglosc_miedzy_wokselami)
    return chmura_punktow_woksele

def usuwanie_co_ntego(chmura_punktow, n = 10):
    chmura_punktow_co_nty = chmura_punktow.uniform_down_sample(every_k_points = n)
    return chmura_punktow_co_nty

if __name__ == "__main__":
    plik = 'merged.las'
    chmura_punktow = las_to_o3d(plik)
    # o3d.visualization.draw_geometries([chmura_punktow], window_name = "Chmura punktów")

    chmura_punktow_odfiltrowana, punkty_odstajace = wyznaczanie_obserwacji_odstajacych(chmura_punktow)
    # o3d.visualization.draw_geometries([chmura_punktow_odfiltrowana, punkty_odstajace], window_name = "Chmura punktów z obserwacjami odstającymi")

    chmura_punktow_woksele = regularyzacja_chmur_punktow(chmura_punktow_odfiltrowana, odleglosc_miedzy_wokselami = 0.1)
    o3d.visualization.draw_geometries([chmura_punktow_woksele], window_name = "Chmura punktów woksele")

    # chmura_punktow_co_nty = usuwanie_co_ntego(chmura_punktow, n = 10)
    # o3d.visualization.draw_geometries([chmura_punktow_co_nty], window_name = "Chmura punktów co n-tego")
    
