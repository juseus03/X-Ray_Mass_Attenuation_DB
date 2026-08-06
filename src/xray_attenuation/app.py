"""imgui_bundle / hello_imgui front end for the X-ray attenuation package"""

import logging
import re
from collections.abc import Callable
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
# Helpers
# =============================================================================


@dataclass
class PhysicsQuantity:
    """One derived quantity shown in the "Other Information" panel

    Attributes:
        name (str): display text without the unit, also the plot legend entry
        compute (Callable): Function for calculation through CLI
        unit (str): bare unit, rendered dimmed next to the value, e.g. "keV"
        fmt (str): display format
        show_in_plot (bool):  "Show in plot" checkbox
    """

    name: str
    compute: Callable[[CLI], float | None]
    unit: str = ""
    fmt: str = "{:.2f}"
    value: float = 0.0
    show_in_plot: bool = False

    def refresh(self, cli: CLI) -> None:
        value = self.compute(cli)
        self.value = value if value is not None else 0.0

    @property
    def text(self) -> str:
        return self.fmt.format(self.value)


def _make_physics_info() -> dict[str, PhysicsQuantity]:
    """The panel contents, in display order."""
    return {
        "filtered_fraction": PhysicsQuantity(
            "Photons removed",
            lambda cli: (
                (1 - f) * 100
                if (f := cli.get_total_filtered_fraction()) is not None
                else None
            ),
            unit="%",
        ),
        "hvl": PhysicsQuantity(
            "Half-value layer", lambda cli: cli.get_hvl(), unit="mm Al"
        ),
        "mean_energy": PhysicsQuantity(
            "Mean energy",
            lambda cli: cli.get_mean_energy_spectrum(),
            unit="keV",
        ),
        "eeff": PhysicsQuantity(
            "Effective energy",
            lambda cli: cli.get_effective_energy(),
            unit="keV",
        ),
    }


# =============================================================================
# AppState
# =============================================================================
@dataclass
class AppState:
    """Holds the GUI state and forwards every user action to the CLI layer

    Attributes:
        cli (CLI): the calculation layer. Owns the spectrum dataframe and the
            authoritative filter stack
        spectrum_list (list[str]): available tube voltages, as the column labels of
            the spectra table (e.g. "9" ... "100").
        current_spectrum_idx (int): index into ``spectrum_list`` of the selected
            tube voltage.
        material_list (list[str]): selectable materials as *display* names, i.e.
            elements as "Aluminum (Al)" and compounds under their plain name.
        pattern (re.Pattern): matches the trailing " (Symbol)" of an element display
            name, so it can be stripped before the name is resolved against the
            database
        filters (list[Filter]): the active filter stack. A read-only view of
            ``cli.filters``; drives the filter panel and the plot labels
        curve_colors (list[imgui.ImVec4]): the color ImPlot gave each spectrum curve,
            in plot order — the unfiltered spectrum first, then one per filter.
            Written by ``gui_plot``, read by ``gui_commands``
    """

    cli: CLI = CLI()

    spectrum_list = cli.data.get_spectrum_list()
    current_spectrum_idx = 41  # Starts at 50 keV

    material_list = cli.data.get_materials_list()

    pattern = re.compile(r"\s*\([A-Z][a-z]?\)$")

    physics_info: dict[str, PhysicsQuantity] = field(default_factory=_make_physics_info)

    curve_colors: list[imgui.ImVec4] = field(default_factory=list)

    @property
    def filters(self) -> list[Filter]:
        """The active filter stack, owned by the CLI layer

        Returns:
            list[Filter]: ``cli.filters``, read live. ``CLI.remove_filter`` rebinds
                that attribute instead of mutating it, so this must not be cached
        """
        return self.cli.filters

    def get_current_base_spectrum(self) -> str:
        """Returns the label of the currently selected base spectrum

        Returns:
            str: the tube voltage as the spectra table's column label, e.g. "50".
        """
        return self.spectrum_list[self.current_spectrum_idx]

    def register_filter(self, name: str, thickness: float) -> None:
        """Adds a filter to the stack and applies it to the current spectrum

        Args:
            name (str): *display* name taken from ``material_list``, e.g.
                "Aluminum (Al)" for an element or "Water, Liquid" for a compound
            thickness (float): filter thickness in cm

        Note:
            The energy the filter is applied at comes from the currently selected
            combo entry, so the caller must have brought the base spectrum in sync
            with that selection first
        """

        thickness = np.round(thickness, 5)
        name = self.pattern.sub("", name)

        try:
            n, is_compound = self.cli.get_single_material_name(name)
        except TypeError:
            logger.warning("Filter not registered. %s not in database", name)
            return

        energy = float(self.get_current_base_spectrum())
        self.cli.add_filter(n, energy, thickness, is_compound)

        logger.debug("Register filter: %s %s %s", n, thickness, is_compound)

        self.update_phyics_info()

    def remove_filter(self, idx: int) -> None:
        """Removes one filter from the stack and recomputes the spectrum

        Args:
            idx (int): position of the filter in the filter list
        """

        if not 0 <= idx < len(self.filters):
            logger.warning("Filter %s doesn't exist, nothing removed", idx)
            return

        old_filter = self.filters[idx]
        self.cli.remove_filter(idx)

        logger.debug(
            "Filter %s removed: %s - %s", idx, old_filter.name, old_filter.thickness
        )
        self.update_phyics_info()

    def update_phyics_info(self) -> None:
        """Triggers the re-calculation of the physical information values"""
        for quantity in self.physics_info.values():
            quantity.refresh(self.cli)

    def get_total_filter_stack(self) -> str:

        answ = ""

        # material:thickness dic
        added = {}

        for f in self.filters:
            symbol = self.cli.data.get_material_symbol(f.name)
            symbol = f.name if symbol == "" else symbol

            if added.get(symbol) is None:
                added[symbol] = f.thickness * 10
            else:
                added[symbol] += f.thickness * 10

        for key, val in added.items():
            if answ == "":
                answ = f"{val:.3f} mm {key}"
            else:
                answ += f" + {val:.3f} mm {key}"

        return answ


# =============================================================================
# GUI Functions
# =============================================================================


def gui_commands(app_state: AppState) -> None:
    """Renders the "Configuration" panel

    Draws three sections: the tube voltage selector, the filter entry form
    (material combo, thickness input and an "+Add" button) and the current filter
    stack with a "-" button per entry.

    Args:
        app_state (AppState): the GUI state to read and mutate
    """
    static = gui_commands

    if not hasattr(static, "material_idx"):
        static.material_idx = 12  # Starts at Al

    if not hasattr(static, "thickness"):
        static.thickness = 0.1

    imgui.separator_text("Source")

    imgui.text("Tube voltage")
    txt_size = imgui.get_item_rect_size().x + 10

    imgui.same_line()

    slider_value = app_state.spectrum_list[app_state.current_spectrum_idx]
    changed, app_state.current_spectrum_idx = imgui.slider_int(
        "kV",
        app_state.current_spectrum_idx,
        0,
        len(app_state.spectrum_list) - 1,
        slider_value,
    )

    imgui.indent(txt_size)
    imgui.text_disabled("W-anode, 150 um Be window")
    imgui.unindent(txt_size)

    imgui.separator_text("Add filter")
    materials = app_state.material_list

    avail_w = imgui.get_content_region_avail().x
    widget_w = int(avail_w / 4.5)

    imgui.align_text_to_frame_padding()
    imgui.text("Material")
    imgui.same_line()

    imgui.indent(widget_w)

    if imgui.begin_combo("##filters", materials[static.material_idx]):
        for n, m in enumerate(materials):
            is_selected = static.material_idx == n
            _, is_selected = imgui.selectable(m, is_selected)
            if is_selected:
                static.material_idx = n
            if is_selected:
                imgui.set_item_default_focus()
        imgui.end_combo()
    imgui.unindent(widget_w)

    cbx_size = imgui.get_item_rect_size()

    imgui.align_text_to_frame_padding()
    imgui.text("Thickness")
    imgui.same_line()

    imgui.indent(widget_w)

    changed, static.thickness = imgui.input_float("[mm]", static.thickness, 0.001, 0.1)

    if changed:
        static.thickness = static.thickness if static.thickness > 0 else 0.001

    if imgui.button("Add to stack", imgui.ImVec2(cbx_size.x, 0)):
        if static.thickness <= 0:
            logger.info("Filter thicknes %s <= 0 ", static.thickness)
        else:
            # Thickness should be in cm when passed to the CLI
            app_state.register_filter(
                materials[static.material_idx], static.thickness * 1e-1
            )

    imgui.unindent(widget_w)

    imgui.separator_text("Filter stack")

    # Table layout for better organization and indent

    width = imgui.get_content_region_avail().x

    tbl_flags = imgui.TableFlags_.borders_outer

    if imgui.begin_table("tbl1", 3, tbl_flags):
        imgui.table_setup_column(
            "Material", imgui.TableColumnFlags_.width_fixed, width * 0.6
        )
        imgui.table_setup_column(
            "Thickness", imgui.TableColumnFlags_.width_fixed, width * 0.2
        )
        imgui.table_headers_row()
        for i, f in enumerate(app_state.filters):
            imgui.push_id(i)
            imgui.table_next_row()
            imgui.table_next_column()
            color = (
                app_state.curve_colors[i + 1]
                if i + 1 < len(app_state.curve_colors)
                else imgui.ImVec4(0, 0, 0, 0)
            )
            implot.item_icon(color)
            imgui.align_text_to_frame_padding()
            imgui.same_line()
            imgui.text_wrapped(f"{f.name}")

            imgui.table_next_column()
            imgui.align_text_to_frame_padding()
            imgui.text_wrapped(f"{f.thickness * 10:.2f} mm")

            imgui.table_next_column()
            # Button fills the full space
            if imgui.button("-", immapp.em_to_vec2(-1, 0)):
                app_state.remove_filter(i)
            imgui.pop_id()
        imgui.end_table()

    imgui.text_disabled("Total")
    imgui.same_line()
    imgui.text_wrapped(app_state.get_total_filter_stack())


def gui_plot(app_state: AppState) -> None:
    """Renders the "Plot" panel

    Draws a log-Y ImPlot of the unfiltered base spectrum plus one curve per filter.

    Args:
        app_state (AppState): the GUI state to read and mutate
    """
    static = gui_plot

    if not hasattr(static, "base_energy"):
        static.base_energy = app_state.get_current_base_spectrum()

    if not hasattr(static, "ylimits"):
        static.ylimits = [1e-10, 1e-5]

    ## Plot checkboxes
    current_info = app_state.physics_info["mean_energy"]
    _, current_info.show_in_plot = imgui.checkbox(
        current_info.name, current_info.show_in_plot
    )
    imgui.same_line()
    imgui.indent(immapp.em_to_vec2(10, 0).x)

    current_info = app_state.physics_info["eeff"]
    _, current_info.show_in_plot = imgui.checkbox(
        current_info.name, current_info.show_in_plot
    )
    imgui.unindent(immapp.em_to_vec2(10, 0).x)

    ## Spectra plot

    yflags = implot.AxisFlags_.lock_max | implot.AxisFlags_.lock_min

    current_spectrum_lbl = app_state.get_current_base_spectrum()

    if app_state.cli.spectrum_df is None or static.base_energy != current_spectrum_lbl:
        app_state.cli.add_base_spectrum(int(current_spectrum_lbl))
        static.base_energy = app_state.get_current_base_spectrum()
        app_state.update_phyics_info()

    energy = app_state.cli.spectrum_df["Energy[keV]"].to_numpy().flatten()

    intensity_cols = app_state.cli.spectrum_df.columns[1:]

    # One label per column: the base spectrum, then the cumulative filters. strict
    # asserts that correspondence rather than letting a curve be mislabelled
    labels = [f"{app_state.cli.max_kv} kV"]
    labels += [f"+ {f.name} {f.thickness * 10:.3f} mm" for f in app_state.filters]

    spec_fill = implot.Spec(flags=0, fill_alpha=0.25)

    if implot.begin_plot("Spectrum", imgui.ImVec2(-1, -1)):
        implot.setup_axes("Photon energy [keV]", "Intensity [a. u.]", 0, yflags)
        implot.setup_axis_scale(implot.ImAxis_.y1, implot.Scale_.log10)
        implot.setup_axes_limits(0, 100, static.ylimits[0], static.ylimits[1])
        implot.setup_legend(implot.Location_.north_east)

        curve_colors = []

        for i, (lbl, c) in enumerate(zip(labels, intensity_cols, strict=True)):
            intensity = app_state.cli.spectrum_df[c].to_numpy().flatten()

            imgui.push_id(i)
            implot.plot_line(lbl, energy, intensity, spec=implot.Spec(line_weight=3))
            # plot_shaded reuses this item's id, so this is the color of both
            curve_colors.append(implot.get_last_item_color())
            implot.plot_shaded(lbl, energy, intensity, spec=spec_fill)
            imgui.pop_id()

        app_state.curve_colors = curve_colors

        ylims = implot.get_plot_limits().y
        log_min, log_max = np.log10(ylims.min), np.log10(ylims.max)
        # 90% of the visible height. The axis is log-scaled, so the fraction is taken
        # in log space rather than as 0.9 * ymax
        y_annotation = 10 ** (log_min + 0.93 * (log_max - log_min))
        line_height = imgui.get_text_line_height_with_spacing()

        markers = [
            q for q in app_state.physics_info.values() if q.show_in_plot and q.value
        ]

        for i, quantity in enumerate(markers):
            implot.plot_inf_lines(
                quantity.name,
                np.array([quantity.value]),
                spec=implot.Spec(flags=implot.ItemFlags_.no_legend, line_weight=2),
            )
            # The markers land close together on a filtered spectrum, so each label
            # drops one text line below the previous one
            implot.annotation(
                quantity.value,
                y_annotation,
                implot.get_last_item_color(),
                imgui.ImVec2(+15, 15 + i * 2 * line_height),
                False,
                quantity.name,
            )

        implot.end_plot()


def _gui_quantity_cell(quantity: PhysicsQuantity, value_width: float) -> None:
    """Renders one quantity inside the current table cell

    Lays out ``name`` flush left, then the unit and the value flush right. The value
    keeps a fixed width so the unit does not shift when a value gains or loses a digit.

    Args:
        quantity (PhysicsQuantity): the quantity to render
        value_width (float): reserved width for the value, in pixels
    """
    right = imgui.get_cursor_pos_x() + imgui.get_content_region_avail().x
    spacing = imgui.get_style().item_spacing.x

    imgui.text_unformatted(quantity.name)

    imgui.same_line()
    unit_width = imgui.calc_text_size(quantity.unit).x
    imgui.set_cursor_pos_x(right - value_width - spacing - unit_width)
    imgui.push_style_color(
        imgui.Col_.text, imgui.get_style_color_vec4(imgui.Col_.text_disabled)
    )
    imgui.text_unformatted(quantity.unit)
    imgui.pop_style_color()

    imgui.same_line()
    imgui.set_cursor_pos_x(right - imgui.calc_text_size(quantity.text).x)
    imgui.text_unformatted(quantity.text)


def gui_info(app_state: AppState) -> None:
    """Renders the "Beam quality" panel as a 2x2 grid of quantities

    Args:
        app_state (AppState): Current application state
    """
    quantities = list(app_state.physics_info.values())
    # Widest value the quantities can reach, so the columns never shift
    value_width = imgui.calc_text_size("000.00").x

    flags = imgui.TableFlags_.borders_inner_v | imgui.TableFlags_.borders_h

    imgui.push_style_var(imgui.StyleVar_.cell_padding, immapp.em_to_vec2(0.6, 0.6))
    if imgui.begin_table("tbl2", 2, flags):
        for row in range(0, len(quantities), 2):
            imgui.table_next_row()
            for quantity in quantities[row : row + 2]:
                imgui.table_next_column()
                _gui_quantity_cell(quantity, value_width)
        imgui.end_table()
    imgui.pop_style_var()


# =============================================================================
# GUI - Backbones
# =============================================================================
def create_default_docking_splits() -> list[hello_imgui.DockingSplit]:
    """Defines how the screen space is split into target zones for the windows

    "MainDockSpace" is provided automatically by hello_imgui and is split into
    three zones::

        ___________________________________________
        |        |                                |
        | Command|                                |
        | Space  |    MainDockSpace               |
        |        |                                |
        |        |                                |
        |        |                                |
        -------------------------------------------
        |     OtherInfo                           |
        -------------------------------------------

    Returns:
        list[hello_imgui.DockingSplit]: the splits, in application order. "OtherInfo"
            is carved out first so that "CommandSpace" spans the full height
    """

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
    """Builds the three dockable windows and binds them to their render callbacks

    - "Configuration" in "CommandSpace", rendered by ``gui_commands``
    - "Other Information" in "OtherInfoSPace", rendered by ``hello_imgui.log_gui``
    - "Plot" in "MainDockSpace", rendered by ``gui_plot``

    Args:
        app_state (AppState): the GUI state the callbacks are bound against

    Returns:
        list[hello_imgui.DockableWindow]: the windows, in render order
    """
    configurations_window = hello_imgui.DockableWindow()
    configurations_window.label = "Configuration"
    configurations_window.dock_space_name = "CommandSpace"
    configurations_window.gui_function = lambda: gui_commands(app_state)
    configurations_window.can_be_closed = False

    other_information_window = hello_imgui.DockableWindow()
    other_information_window.label = "Beam quality"
    other_information_window.dock_space_name = "OtherInfoSPace"
    other_information_window.gui_function = lambda: gui_info(app_state)
    other_information_window.can_be_closed = False

    main_plot_window = hello_imgui.DockableWindow()
    main_plot_window.label = "Plot"
    main_plot_window.dock_space_name = "MainDockSpace"
    main_plot_window.imgui_window_flags = imgui.WindowFlags_.menu_bar
    main_plot_window.gui_function = lambda: gui_plot(app_state)
    main_plot_window.can_be_closed = False

    dockable_windows = [
        configurations_window,
        other_information_window,
        main_plot_window,
    ]
    return dockable_windows


def create_default_layout(app_state: AppState) -> hello_imgui.DockingParams:
    """Assembles the default docking layout from its splits and windows

    Args:
        app_state (AppState): the GUI state the window callbacks are bound against

    Returns:
        hello_imgui.DockingParams: the layout, named "Default". It is only applied
            the first time the app runs; afterwards the saved .ini file wins
    """
    docking_params = hello_imgui.DockingParams()
    # By default, the layout name is already "Default"
    # docking_params.layout_name = "Default"
    docking_params.docking_splits = create_default_docking_splits()
    docking_params.dockable_windows = create_dockable_windows(app_state)
    return docking_params


def main() -> None:
    """Entrypoint for the GUI

    Builds the ``AppState``, configures the hello_imgui runner (window, docking
    layout and .ini location) and hands over to ``immapp.run`` with the ImPlot
    add-on enabled. Blocks until the user closes the window.
    """

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
