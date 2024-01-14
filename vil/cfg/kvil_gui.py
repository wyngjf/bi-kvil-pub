import polyscope as ps


def init_gui():
    ps.set_program_name("K-VIL")
    ps.set_autocenter_structures(False)
    ps.set_autoscale_structures(False)

    # ps.set_length_scale(1.)
    ps.set_up_dir("neg_y_up")
    ps.set_navigation_style("free")
    ps.set_ground_plane_mode("none")
    ps.init()
    ps.set_give_focus_on_show(True)
