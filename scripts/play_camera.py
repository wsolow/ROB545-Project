"""play_camera.py.

Show the live RealSense camera feed in an OpenCV window. Press V to start / stop recording
a video; close the window or press Ctrl-C to quit.

Written by Will Solow, 2026.
"""

import argparse

from kinova_lfd.envs.base.cameras import CameraEnv, RealsenseCameraCfg
from kinova_lfd.envs.base.viewers.camera_viewer import CameraViewer

# add argparse arguments
parser = argparse.ArgumentParser(description="Live RealSense camera viewer.")
parser.add_argument("--width", type=int, default=640, help="Stream width (px).")
parser.add_argument("--height", type=int, default=480, help="Stream height (px).")
parser.add_argument("--fps", type=int, default=30, help="Stream frame rate.")
parser.add_argument("--video_dir", type=str, default="exp/camera_videos", help="Where recorded videos go.")

# parse the arguments
args_cli = parser.parse_args()


def main() -> None:
    """Create the RealSense camera env and show its live feed."""
    env = CameraEnv(RealsenseCameraCfg(width=args_cli.width, height=args_cli.height, fps=args_cli.fps))
    env.reset()

    viewer = CameraViewer(env, preview_fps=args_cli.fps, preview_height=args_cli.height, video_dir=args_cli.video_dir)
    viewer.run()


if __name__ == "__main__":
    main()
