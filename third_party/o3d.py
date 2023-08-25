import open3d as o3d

# Load the point cloud from a file (replace 'path_to_point_cloud.ply' with the actual path)
point_cloud = o3d.io.read_point_cloud('/misc/lmbraid19/sommerl/datasets/CO3D_Preprocess/pcls/car/106_12650_23736/pcl_clean.ply')
def pick_points(pcd):
    print("")
    print(
        "1) Please pick at least three correspondences using [shift + left click]"
    )
    print("   Press [shift + right click] to undo point picking")
    print("2) Afther picking points, press q for close the window")
    vis = o3d.visualization.VisualizerWithEditing()
    vis.create_window()
    vis.add_geometry(pcd)
    vis.run()  # user picks points
    vis.destroy_window()
    print("")
    return vis.get_picked_points()

print(pick_points(point_cloud))