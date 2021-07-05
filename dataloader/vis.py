import copy
import numpy as np
import open3d as o3d

if __name__ == "__main__":



    # Load saved point cloud and visualize it
    #pcd_load = o3d.io.read_point_cloud("./back.ply")
    pcd_load = o3d.io.read_point_cloud("./KITTI_original.ply")

    o3d.visualization.draw_geometries([pcd_load])

    # convert Open3D.o3d.geometry.PointCloud to numpy array
    xyz_load = np.asarray(pcd_load.points)
    print('xyz_load')
    print(xyz_load)

    '''
    # save z_norm as an image (change [0,1] range to [0,255] range with uint8 type)
    img = o3d.geometry.Image((z_norm * 255).astype(np.uint8))
    o3d.io.write_image("./sync.png", img)
    o3d.visualization.draw_geometries([img])
    '''