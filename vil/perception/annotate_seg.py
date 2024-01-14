import copy
import os

import cv2
import click
from pathlib import Path
from robot_utils.py.filesystem import get_ordered_subdirs, get_ordered_files, validate_path
from robot_utils.serialize.dataclass import dump_data_to_yaml
from robot_utils.py.interact import ask_checkbox_with_all, ask_list
from robot_utils.cv.io.io_cv import load_rgb
from robot_utils import console
from pathos.multiprocessing import ProcessPool
from vil.cfg.preprocessing import SegmentationConfig, ClipCfg, VideoClipConfig


def annotate(folder: Path, occlude_obj: str):
    console.rule(f"annotate {folder}")
    if occlude_obj is not None:
        occlude_obj_list = occlude_obj.split(",")
        console.log(f"[bold blue]occluded objects include: {occlude_obj_list}")
    else:
        occlude_obj_list = None
    pool = ProcessPool(os.cpu_count())
    image_files = get_ordered_files(folder / "images", pattern=[".jpg"], ex_pattern=["right_"])
    images = pool.map(load_rgb, image_files)
    ic(len(images))
    i_max = len(images) - 1

    i = 0
    pause = True

    c = SegmentationConfig()
    c.manual = dict(
        motion_seg=copy.deepcopy(VideoClipConfig()),
        kvil=copy.deepcopy(VideoClipConfig())
    )

    while True:
        cv2.imshow("main", images[i])
        k = cv2.waitKey(1) & 0xFF
        if k == 27 or k == ord("q"):
            break

        elif k == ord("p"):
            pause = not pause

        elif k == ord("j"):
            pause = True
            i -= 1

        elif k == ord("k"):
            pause = True
            i += 1

        elif k == ord("a"):
            c.manual["motion_seg"].video.s = i
        elif k == ord("s"):
            c.manual["kvil"].video.s = i
        elif k == ord("d"):
            c.manual["kvil"].video.e = i
        elif k == ord("f"):
            c.manual["motion_seg"].video.e = i
        elif k == ord("o") or k == ord("i") or k == ord("u"):
            if occlude_obj_list is None:
                continue
            if k == ord("o"):
                idx = 0
            elif k == ord("i"):
                idx = 1
            elif k == ord("u"):
                idx = 2
            else:
                idx = 0
            occ_obj = occlude_obj_list[idx]
            c.manual["motion_seg"].occlude[occ_obj] = ClipCfg()
            c.manual["motion_seg"].occlude[occ_obj].s = i
            c.manual["kvil"].occlude[occ_obj] = ClipCfg()
            c.manual["kvil"].occlude[occ_obj].s = i
        elif k == ord("c"):
            ic(c)

        i = min(max(0, i), i_max)
        if i < i_max and not pause:
            i += 1

    dump_data_to_yaml(SegmentationConfig, c, folder / "seg.yaml")
    cv2.destroyAllWindows()


@click.command(context_settings=dict(help_option_names=['-h', '--help']))
@click.option("--path",  "-p", type=str, help="the absolute path of the skill, parent of recordings")
@click.option("--occlude_obj",  "-o", type=str, help="the name of the occluded object")
def main(path, occlude_obj):
    path = validate_path(path, throw_error=True)[0]
    if not (path / "recordings").is_dir():
        console.log("[bold red]You may forget to put your recordings in the 'recording' folder")
        exit()
    folder_list = get_ordered_subdirs(path / "recordings")
    ic(len(folder_list))
    folder_name_list = [folder.stem for folder in folder_list]
    selected = ask_checkbox_with_all("select folder:", folder_name_list)
    folder_list = [folder_list[folder_name_list.index(f)] for f in selected]

    for folder in folder_list:
        annotate(folder, occlude_obj)


if __name__ == "__main__":
    main()
