import logging
import re
from dataclasses import dataclass, field

import numpy as np
from imgui_bundle import hello_imgui, imgui, immapp, implot

from xray_attenuation import CLI, Filter

LOG_LEVEL = "DEBUG"

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL),
    format="|%(asctime)s| (%(levelname)s) (%(name)s) - %(message)s",
)
logger.info("Logging level set to: %s", LOG_LEVEL)


# =============================================================================
# AppState
# =============================================================================
@dataclass
class AppState:
    """Class for controlling the GUI"""

    version: str = "0.0.1"
    counter: int = 0
    f: float = 0.0
    cli: CLI = CLI()

    spectrum_list = cli.data.get_spectrum_list()
    current_spectrum_idx = 41

    material_list = cli.data.get_materials_list()

    pattern = re.compile(r"\s*\([A-Z][a-z]?\)$")
    filters: list[Filter] = field(default_factory=list[Filter])

    def get_current_base_spectrum(self):
        return self.spectrum_list[self.current_spectrum_idx]

    def register_filter(self, name: str, thickness: float):

        thickness = np.round(thickness, 5)
        name = self.pattern.sub("", name)

        n, is_compound = self.cli.get_single_material_name(name)
        energy = float(self.get_current_base_spectrum())
        self.cli.add_filter(n, energy, thickness, is_compound)

        self.filters.append(Filter(name, thickness, is_compound))

        logging.debug("Register filter: %s %s %s", n, thickness, is_compound)

    def remove_filter(self, idx: int):

        old_filter = self.filters[idx]
        self.cli.remove_filter(idx)

        self.filters.pop(idx)
        logging.debug(
            "Filter %s removed: %s - %s", idx, old_filter.name, old_filter.thickness
        )

    def clean_filters_list(self):
        self.filters = []
        logging.debug("Cleaned filter list")


# =============================================================================
# GUI Functions
# =============================================================================


def gui_commands(app_state: AppState):
    static = gui_commands

    if not hasattr(static, "material_idx"):
        static.material_idx = 12

    if not hasattr(static, "thickness"):
        static.thickness = 0.1

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

    imgui.separator_text("Filter Configuration")
    materials = app_state.material_list

    if imgui.begin_combo("##filters", materials[static.material_idx]):
        for n, m in enumerate(materials):
            is_selected = static.material_idx == n
            _, is_selected = imgui.selectable(m, is_selected)
            if is_selected:
                static.material_idx = n
            if is_selected:
                imgui.set_item_default_focus()
        imgui.end_combo()

    cbx_size = imgui.get_item_rect_size()

    changed, static.thickness = imgui.input_float(
        "Thickness [cm]", static.thickness, 0.001, 0.1
    )

    if changed:
        static.thickness = static.thickness if static.thickness > 0 else 0

    if imgui.button("+Add", imgui.ImVec2(cbx_size.x, 0)):
        app_state.register_filter(materials[static.material_idx], static.thickness)

    imgui.separator_text("Filters")

    for i, f in enumerate(app_state.filters):
        imgui.push_id(i)
        imgui.align_text_to_frame_padding()
        imgui.text(f"{f.name} - {f.thickness} cm")
        imgui.same_line()
        if imgui.button("-", immapp.em_to_vec2(4, 0)):
            app_state.remove_filter(i)

        imgui.pop_id()


def gui_plot(app_sate: AppState):
    static = gui_plot

    if not hasattr(static, "base_energy"):
        static.base_energy = app_sate.get_current_base_spectrum()

    if not hasattr(static, "limits"):
        static.ylimits = [1e-10, 1e-5]

    yflags = implot.AxisFlags_.lock_max | implot.AxisFlags_.lock_min

    current_spectrum_lbl = app_sate.get_current_base_spectrum()

    if app_sate.cli.spectrum_df is None or static.base_energy != current_spectrum_lbl:
        app_sate.clean_filters_list()
        app_sate.cli.add_base_spectrum(int(current_spectrum_lbl))
        static.base_energy = app_sate.get_current_base_spectrum()

    energy = app_sate.cli.spectrum_df["Energy[keV]"].to_numpy().flatten()

    intensity_cols = app_sate.cli.spectrum_df.columns[1:]

    implot.begin_plot("Spectrum", immapp.em_to_vec2(41, 41))
    implot.setup_axes("Energy [keV]", "Intensity [a. u.]", 0, yflags)
    implot.setup_axis_scale(implot.ImAxis_.y1, implot.Scale_.log10)
    implot.setup_axes_limits(0, 100, static.ylimits[0], static.ylimits[1])

    for i, c in enumerate(intensity_cols):
        intensity = app_sate.cli.spectrum_df[c].to_numpy().flatten()

        if i == 0:
            lbl = f"{c} kV"
        else:
            flt = app_sate.filters[i - 1]
            lbl = f"+ {flt.name} {flt.thickness} cm"

        imgui.push_id(i)
        implot.plot_line(lbl, energy, intensity)
        imgui.pop_id()

    implot.end_plot()


# =============================================================================
# GUI - Backbones
# =============================================================================
def create_default_docking_splits() -> list[hello_imgui.DockingSplit]:
    # Define the default docking splits,
    # i.e. the way the screen space is split in different target zones
    # for the dockable windows
    # We want to split "MainDockSpace" (which is provided automatically) into three
    # zones like this:
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

    # Add a space named "MiscSpace" whose height is 25% of the app height.
    # This will split the preexisting default dockspace "MainDockSpace" in two parts.
    split_main_misc = hello_imgui.DockingSplit()
    split_main_misc.initial_dock = "MainDockSpace"
    split_main_misc.new_dock = "OtherInfoSPace"
    split_main_misc.direction = imgui.Dir.down
    split_main_misc.ratio = 0.25

    # Add a space to the left which occupies a column whose width
    # is 30% of the app width
    split_main_command = hello_imgui.DockingSplit()
    split_main_command.initial_dock = "MainDockSpace"
    split_main_command.new_dock = "CommandSpace"
    split_main_command.direction = imgui.Dir.left
    split_main_command.ratio = 0.30

    splits = [split_main_misc, split_main_command]
    return splits


def create_dockable_windows(app_state: AppState) -> list[hello_imgui.DockableWindow]:
    # A features demo window named "FeaturesDemo" will be placed in "CommandSpace".
    # Its Gui is provided by "gui_window_demo_features"
    configurations_window = hello_imgui.DockableWindow()
    configurations_window.label = "Configuration"
    configurations_window.dock_space_name = "CommandSpace"
    configurations_window.gui_function = lambda: gui_commands(app_state)

    # A Log window named "Logs" will be placed in "MiscSpace"
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
    # Use this for deployment
    # runner_params.ini_folder_type = hello_imgui.IniFolderType.app_user_config_folder
    runner_params.ini_folder_type = hello_imgui.IniFolderType.current_folder
    runner_params.ini_filename = "xray_attenuation/xray_attenuation.ini"

    # hello_imgui.run(runner_params)
    addons = immapp.AddOnsParams()
    addons.with_implot = True
    immapp.run(runner_params, addons)


if __name__ == "__main__":
    main()
