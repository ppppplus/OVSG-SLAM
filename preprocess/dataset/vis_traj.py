import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def read_poses_from_txt(file_path):
    poses = []
    with open(file_path, 'r') as f:
        for line in f:
            vals = list(map(float, line.strip().split()))
            if len(vals) != 16:
                continue  # 跳过非法行
            pose = np.array(vals).reshape((4, 4))
            poses.append(pose)
    return poses

def plot_camera_trajectory(poses):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # 提取相机位置（平移部分）
    xs, ys, zs = [], [], []
    for pose in poses:
        t = pose[:3, 3]
        xs.append(t[0])
        ys.append(t[1])
        zs.append(t[2])

    ax.plot(xs, ys, zs, label='Camera Trajectory', color='blue', linewidth=2)
    ax.scatter(xs, ys, zs, color='red', s=5)

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('Camera Trajectory')
    ax.legend()
    ax.view_init(elev=30, azim=45)
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    poses = read_poses_from_txt('/home/ubuntu/TJH/Work/aff_ws/SGS-SLAM/data/Replica/room0/traj.txt')
    plot_camera_trajectory(poses)
