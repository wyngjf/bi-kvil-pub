import click
from robot_utils.py.filesystem import validate_path
from vil.kvil.kvil import KVIL
from vil.cfg import init_gui


@click.command(context_settings=dict(help_option_names=['-h', '--help']))
@click.option("--path",         "-p",   type=str,      default="",    help="the absolute path to demonstration root")
@click.option("--append",       "-a",   type=str,      default="",    help="append to the processing path for different experiments")
@click.option("--n_demos",      "-n",   type=int,      default=3,     help="the absolute path to demonstration root")
@click.option("--viz_debug",    "-vd",  is_flag=True,                 help="whether to visualize intermediate results")
@click.option("--force_redo",   "-f",   is_flag=True,                 help="whether to force redo everything")
@click.option("--reload",       "-r",   is_flag=True,                 help="whether to reload kvil constraints")
@click.option("--global_view",  "-g",   is_flag=True,                 help="whether to visualize in global view")
@click.option("--hand_group",   "-hg",  is_flag=True,                 help="whether to generate hand group config")
@click.option("--delete_prev",  "-d",   is_flag=True,                 help="whether to clean up previous data")
def main(path, append, n_demos, force_redo, viz_debug, reload, global_view, hand_group, delete_prev):
    init_gui()
    path, _ = validate_path(path, throw_error=True)
    kvil = KVIL(path=path, n_demos=n_demos, force_redo=force_redo,
                viz_debug=viz_debug, append=append, reload=reload,
                enable_hand_group=hand_group, delete_previous_data=delete_prev)
    if reload:
        kvil.reload()
    else:
        kvil.pce()
    if global_view:
        kvil.show_global_scene()
    kvil.show_constraints()


if __name__ == "__main__":
    main()
