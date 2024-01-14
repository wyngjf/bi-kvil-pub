from robot_utils.cv.io.io import image_to_video, image_to_gif
from robot_utils.py.filesystem import create_path
from pathlib import Path
import click


@click.command(context_settings=dict(help_option_names=["-h", "--help"]))
@click.option("--path", "-p", type=str, help="the absolute path to the images")
@click.option("--mode", "-m", type=str, help="rgb, depth, mask, depth_color")
@click.option("--pattern", "-pt", type=str, help="patterns to look up image files")
@click.option("--fps", "-f", type=int, help="fps")
@click.option("--codec", "-c", type=str, help="XVID, MPEG, MJPG, mp4v")
def main(path, mode, pattern, fps, codec):
    path = Path(path)
    if not path.is_dir():
        raise FileNotFoundError
    video_file = create_path(path.parent / "video") / f"{path.stem}.mp4"
    gif_file = path.parent / "video" / f"{path.stem}.gif"
    image_to_video(path, video_file, mode, pattern=[pattern if pattern else ""], fps=fps, codec=codec)
    image_to_gif(path, gif_file=gif_file, pattern=[pattern if pattern else ""], duration=3, loop=0)


if __name__ == "__main__":
    main()


# TODO: E.g.
#  python vil/perception/to_video.py -m rgb -f 10 -c mp4v -p /home/gao/dataset/kvil/demo/pour/viz/kvil/20230803_102755/flow/uv
