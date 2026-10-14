import argparse

import cv2
import matplotlib
import matplotlib.pyplot as plt
import numpy as np


def render_plot(points):
    """Render points on a new figure."""
    pts = np.array(points)

    fig = plt.figure(figsize=(4, 4))
    ax = fig.add_subplot(111, projection="3d")

    ax.plot(pts[:, 0], pts[:, 1], pts[:, 2], "r-", linewidth=2)

    ax.set_xlim(0, 1)
    ax.set_ylim(-1, 1)
    ax.set_zlim(0, 1)

    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])

    fig.canvas.draw()

    width, height = fig.canvas.get_width_height()

    img = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
    img = img.reshape(height, width, 4)

    img = img[:, :, :3]

    plt.close(fig)

    return img


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=str, default="exp/tabletop/20260619_131843")
    parser.add_argument("--height", type=int, default=360)
    parser.add_argument("--width", type=int, default=640)
    args = parser.parse_args()
    matplotlib.use("Agg")

    images = np.load(f"{args.path}/_wrist_rgb.npy")
    poses = np.load(f"{args.path}/_tool_pose_b.npy")

    trajectory = []

    T = min(len(images), len(poses))

    for t in range(T):
        wrist = cv2.cvtColor(images[t], cv2.COLOR_RGB2BGR)

        trajectory.append(poses[t, :3])

        plot_img = cv2.cvtColor(render_plot(trajectory), cv2.COLOR_RGB2BGR)

        combined = np.hstack(
            [cv2.resize(wrist, (args.width, args.height)), cv2.resize(plot_img, (args.width, args.height))]
        )

        cv2.imshow("Wrist + Trajectory", combined)

        if cv2.waitKey(1) == 27:
            break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
