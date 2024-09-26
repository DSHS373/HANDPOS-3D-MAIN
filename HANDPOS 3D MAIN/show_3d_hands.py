import numpy as np
import matplotlib.pyplot as plt
from utils import DLT
from pathlib import Path
from mpl_toolkits.mplot3d import Axes3D  # Moved to top for clarity

def read_keypoints(filename):
    with open(filename, 'r') as fin:
        kpts = []
        for line in fin:
            if not line.strip():  # Skip empty lines
                continue
            line = line.split()
            line = [float(s) for s in line]
            line = np.reshape(line, (21, -1))
            kpts.append(line)
    kpts = np.array(kpts)
    return kpts

def visualize_3d(p3ds):
    """Apply coordinate rotations to point z axis as up"""
    Rz = np.array([
        [0., -1., 0.],
        [1.,  0., 0.],
        [0.,  0., 1.]
    ])
    Rx = np.array([
        [1.,  0.,  0.],
        [0., -1.,  0.],
        [0.,  0., -1.]
    ])

    # Apply rotations
    p3ds_rotated = (Rz @ Rx @ p3ds.transpose(0, 2, 1)).transpose(0, 2, 1)

    """Now visualize in 3D"""
    thumb_f = [[0,1],[1,2],[2,3],[3,4]]
    index_f = [[0,5],[5,6],[6,7],[7,8]]
    middle_f = [[0,9],[9,10],[10,11],[11,12]]
    ring_f = [[0,13],[13,14],[14,15],[15,16]]
    pinkie_f = [[0,17],[17,18],[18,19],[19,20]]
    fingers = [pinkie_f, ring_f, middle_f, index_f, thumb_f]
    fingers_colors = ['red', 'blue', 'green', 'black', 'orange']

    # Define the output directory
    output_dir = Path('figs')
    output_dir.mkdir(parents=True, exist_ok=True)  # Create if it doesn't exist

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    for i, kpts3d in enumerate(p3ds_rotated):
        if i % 2 == 0:
            continue  # Skip every 2nd frame

        for finger, finger_color in zip(fingers, fingers_colors):
            for _c in finger:
                ax.plot(
                    xs=[kpts3d[_c[0], 0], kpts3d[_c[1], 0]],
                    ys=[kpts3d[_c[0], 1], kpts3d[_c[1], 1]],
                    zs=[kpts3d[_c[0], 2], kpts3d[_c[1], 2]],
                    linewidth=4,
                    c=finger_color
                )

        # Draw axes
        ax.plot([0,5], [0,0], [0,0], linewidth=2, color='red')
        ax.plot([0,0], [0,5], [0,0], linewidth=2, color='blue')
        ax.plot([0,0], [0,0], [0,5], linewidth=2, color='black')

        # Customize axes
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_zticks([])

        ax.set_xlim3d(-50, 50)
        ax.set_xlabel('x')
        ax.set_ylim3d(-50, 50)
        ax.set_ylabel('y')
        ax.set_zlim3d(-100, 0)
        ax.set_zlabel('z')

        # Define the filepath using pathlib
        filepath = output_dir / f'fig_{i}.png'

        # Save the figure
        try:
            plt.savefig(filepath)
            print(f"Plot saved to {filepath}")
        except Exception as e:
            print(f"Failed to save plot {filepath}: {e}")

        # Pause and clear the axes for the next frame
        plt.pause(0.01)
        ax.cla()

    plt.close(fig)  # Close the figure after all frames are saved

if __name__ == '__main__':
    p3ds = read_keypoints('kpts_3d.dat')
    visualize_3d(p3ds)
