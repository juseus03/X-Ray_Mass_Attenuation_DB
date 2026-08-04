from dataclasses import dataclass

import numpy as np
from imgui_bundle import hello_imgui, imgui, immapp, implot

from xray_attenuation import CLI


# =============================================================================
# AppState
# =============================================================================
@dataclass
class AppState:
    version: str = "0.0.1"
    counter: int = 0
    f: float = 0.0
    cli: CLI = CLI()

    spectrum_list = cli.data.get_spectrum_list()
    current_spectrum_idx = 0

    def get_current_base_spectrum(self):
        return self.spectrum_list[self.current_spectrum_idx]


# =============================================================================
# GUI Functions
# =============================================================================


def gui_commands(app_state: AppState):
    imgui.separator_text("Spectrum")
    imgui.text("Energy")
    imgui.same_line()

    items = app_state.spectrum_list
    combo_preview_value = items[app_state.current_spectrum_idx]

    if imgui.begin_combo("keV", combo_preview_value):
        for n in range(len(app_state.spectrum_list)):
            is_selected = app_state.current_spectrum_idx == n
            _, is_selected = imgui.selectable(items[n], is_selected)
            if is_selected:
                app_state.current_spectrum_idx = n
            if is_selected:
                imgui.set_item_default_focus()
        imgui.end_combo()


def gui_plot(app_sate: AppState):
    static = gui_plot

    if not hasattr(static, "base_energy"):
        static.base_energy = app_sate.get_current_base_spectrum()

    if not hasattr(static, "limits"):
        static.ylimits = [1e-10, 1e-5]

    yflags = implot.AxisFlags_.lock_max | implot.AxisFlags_.lock_min

    current_spectrum_lbl = app_sate.get_current_base_spectrum()

    if app_sate.cli.spectrum_df is None or static.base_energy != current_spectrum_lbl:
        app_sate.cli.add_base_spectrum(int(current_spectrum_lbl))

    energy = app_sate.cli.spectrum_df["Energy[keV]"].to_numpy().flatten()
    intensity = app_sate.cli.spectrum_df[current_spectrum_lbl].to_numpy().flatten()

    implot.begin_plot("Spectrum", immapp.em_to_vec2(41, 41))

    implot.setup_axes("Energy [keV]", "Intensity [a. u.]", 0, yflags)
    implot.setup_axis_scale(implot.ImAxis_.y1, implot.Scale_.log10)
    implot.setup_axes_limits(0, 100, static.ylimits[0], static.ylimits[1])

    implot.plot_line("", energy, intensity)

    implot.end_plot()


# =============================================================================
# GUI - Backbones
# =============================================================================
def create_default_docking_splits() -> list[hello_imgui.DockingSplit]:
    # Define the default docking splits,
    # i.e. the way the screen space is split in different target zones for the dockable windows
    # We want to split "MainDockSpace" (which is provided automatically) into three zones, like this:
    #
    #    ___________________________________________
    #    |        |                                |
    #    | Command|                                |
    #    | Space  |    MainDockSpace               |
    #    |        |                                |
    #    |        |                                |
    #    |        |                                |
    #    -------------------------------------------
    #    |     OtherInfo                           |
    #    -------------------------------------------
    #

    # Uncomment the next line if you want to always start with this layout.
    # Otherwise, modifications to the layout applied by the user layout will be remembered.
    # runner_params.docking_params.layout_condition = (
    #     hello_imgui.DockingLayoutCondition.application_start
    # )

    # Then, add a space named "MiscSpace" whose height is 25% of the app height.
    # This will split the preexisting default dockspace "MainDockSpace" in two parts.
    split_main_misc = hello_imgui.DockingSplit()
    split_main_misc.initial_dock = "MainDockSpace"
    split_main_misc.new_dock = "OtherInfoSPace"
    split_main_misc.direction = imgui.Dir.down
    split_main_misc.ratio = 0.25

    # Then, add a space to the left which occupies a column whose width is 25% of the app width
    split_main_command = hello_imgui.DockingSplit()
    split_main_command.initial_dock = "MainDockSpace"
    split_main_command.new_dock = "CommandSpace"
    split_main_command.direction = imgui.Dir.left
    split_main_command.ratio = 0.25

    splits = [split_main_misc, split_main_command]
    return splits


def create_dockable_windows(app_state: AppState) -> list[hello_imgui.DockableWindow]:
    # A features demo window named "FeaturesDemo" will be placed in "CommandSpace".
    # Its Gui is provided by "gui_window_demo_features"
    configurations_window = hello_imgui.DockableWindow()
    configurations_window.label = "Configuration"
    configurations_window.dock_space_name = "CommandSpace"
    configurations_window.gui_function = lambda: gui_commands(app_state)

    # A Log window named "Logs" will be placed in "MiscSpace". It uses the HelloImGui logger gui
    other_information_window = hello_imgui.DockableWindow()
    other_information_window.label = "Other Information"
    other_information_window.dock_space_name = "OtherInfoSPace"
    other_information_window.gui_function = hello_imgui.log_gui

    # A window will be placed in "MainDockSpace"
    main_plot_window = hello_imgui.DockableWindow()
    main_plot_window.label = "Plot"
    main_plot_window.dock_space_name = "MainDockSpace"
    main_plot_window.imgui_window_flags = imgui.WindowFlags_.menu_bar
    main_plot_window.gui_function = lambda: gui_plot(app_state)

    dockable_windows = [
        configurations_window,
        other_information_window,
        main_plot_window,
    ]
    return dockable_windows


def create_default_layout(app_state: AppState) -> hello_imgui.DockingParams:
    docking_params = hello_imgui.DockingParams()
    # By default, the layout name is already "Default"
    # docking_params.layout_name = "Default"
    docking_params.docking_splits = create_default_docking_splits()
    docking_params.dockable_windows = create_dockable_windows(app_state)
    return docking_params


def main():

    app_state = AppState()

    runner_params = hello_imgui.RunnerParams()
    runner_params.app_window_params.window_title = "X-ray Attenuation"
    runner_params.imgui_window_params.menu_app_title = "X-ray Attenuation"
    runner_params.app_window_params.window_geometry.size = (1000, 900)
    runner_params.app_window_params.restore_previous_geometry = False
    runner_params.app_window_params.borderless = False
    runner_params.app_window_params.borderless_movable = False
    runner_params.app_window_params.borderless_resizable = False
    runner_params.app_window_params.borderless_closable = False

    # runner_params.docking_params.layout_condition = (
    #     hello_imgui.DockingLayoutCondition.application_start
    # )

    runner_params.imgui_window_params.default_imgui_window_type = (
        hello_imgui.DefaultImGuiWindowType.provide_full_screen_dock_space
    )

    runner_params.imgui_window_params.enable_viewports = False
    runner_params.docking_params = create_default_layout(app_state)
    # runner_params.ini_folder_type = hello_imgui.IniFolderType.app_user_config_folder # Use this for deployment
    runner_params.ini_folder_type = hello_imgui.IniFolderType.current_folder
    runner_params.ini_filename = "xray_attenuation/xray_attenuation.ini"

    # hello_imgui.run(runner_params)
    addons = immapp.AddOnsParams()
    addons.with_implot = True
    immapp.run(runner_params, addons)


if __name__ == "__main__":
    main()
