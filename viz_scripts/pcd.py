import open3d as o3d

pcd = o3d.io.read_point_cloud("/home/ubuntu/TJH/Work/aff_ws/SGS-SLAM/data/afford/test/init.pcd")
o3d.visualization.draw([pcd])  # 如果是 open3d >= 0.16
