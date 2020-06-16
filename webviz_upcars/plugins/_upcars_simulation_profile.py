import itertools
from collections import namedtuple
from random import random
import dash_html_components as html
import dash_core_components as dcc
from dash.dependencies import Input, Output, State
from webviz_config import WebvizPluginABC

import plotly.graph_objs as go
import webviz_core_components as wcc
import dash
from dash.exceptions import PreventUpdate
from webviz_subsurface._abbreviations.reservoir_simulation import (
    simulation_vector_description,
)

from .._util.palette import PALETTE
from .._util.fmu_input import (
    get_table_df,
    get_ensemble_df,
    get_summary_df,
    get_multiple_table_df,
)

TERMINALCOLORS = {
    "HEADER": "\033[95m",
    "OKBLUE": "\033[94m",
    "OKGREEN": "\033[92m",
    "WARNING": "\033[93m",
    "FAIL": "\033[91m",
    "ENDC": "\033[0m",
    "BOLD": "\033[1m",
    "UNDERLINE": "\033[4m",
}

UDF_VECTOR = {
    "FUPVINJ": "Pore Volume Injected",
    "FUDP": "BHP Differential Pressure",
}

# Create named tuple
TraceLabel = namedtuple("TraceLabel", "curve_name legend_name meta")
TraceStyle = namedtuple("TraceStyle", "opacity color")
PlotOptions = namedtuple("PlotOptions", "reference ensemble profile krpc")


def eclipse_vector_description(keyword):
    return UDF_VECTOR.get(keyword, simulation_vector_description(keyword))


def create_trace_dict(values, label, style, axis_idx, showlegend=False):
    """
    Create dictionary of trace
    label is namedtuple contains curve_name, legend_name and meta
    style is namedtuple contains opacity and color
    """
    return {
        "x": values[0],
        "y": values[1],
        "legendgroup": label.legend_name,
        "hovertext": label.curve_name,
        "hoverinfo": "y+x+text",
        "name": label.legend_name,
        "type": "scattergl",
        "xaxis": f"x{axis_idx}",
        "yaxis": f"y{axis_idx}",
        "showlegend": showlegend,
        "meta": label.meta,
        "mode": "lines",
        "opacity": style.opacity,
        "line": {"color": style.color, "width": 2.0},
        "marker": {"size": 0 if showlegend else 10},
    }


def krpc_table_key(table_type):
    dict_table = {
        "SWOF": {"saturation": "Sw", "kr1": "krw", "kr2": "krow", "pc": "pcow"},
        "SGOF": {"saturation": "Sg", "kr1": "krg", "kr2": "krog", "pc": "pcog"},
    }
    keys = dict_table.get(table_type, dict_table["SWOF"])
    return keys["saturation"], keys["kr1"], keys["kr2"], keys["pc"]


# pylint:disable=too-many-locals
def create_layout(
    sat_axis_title=None, profile_x_axis="", profile_y_axis=None, log_relperm=False
):
    # config = {
    #     'krpc_height': 400,
    #     'profile_height': 300,
    #     'spacing_height': 100.0,
    #     'profile_col_count': 2
    # }

    krpc_height = 400  # 600.0
    profile_height = 300  # 450.0
    spacing_height = 100.0
    axis_format_dict = {
        "gridcolor": "LightGray",
        "gridwidth": 1,
        "linecolor": "black",
        "linewidth": 1,
        "mirror": True,
        "showgrid": True,
        "showline": True,
        "zeroline": True,
        "zerolinecolor": "LightGray",
    }
    count_krpc_row = 1 if sat_axis_title else 0
    count_profile_row = int((len(profile_y_axis) + 1) / 2)
    figure_height = (
        count_krpc_row * krpc_height
        + count_profile_row * profile_height
        + (count_krpc_row + count_profile_row - 1) * spacing_height
    )

    _dict = {
        "height": figure_height,
        "paper_bgcolor": "white",
        "plot_bgcolor": "white",
        "uirevision": str(random()),  # nosec
    }

    # Build bottom-up, start with profile
    chart_y1 = [
        i * (profile_height + spacing_height) / figure_height
        for i in range(count_profile_row)
    ]
    chart_y2 = [y1 + profile_height / figure_height for y1 in chart_y1]
    if count_krpc_row:
        if len(chart_y1) == 0:
            chart_y1 = [0]
            chart_y2 = [1]
        else:
            chart_y1.append(chart_y2[-1] + spacing_height / figure_height)
            chart_y2.append(chart_y1[-1] + krpc_height / figure_height)

    chart_y1.reverse()
    chart_y2.reverse()

    # Axis 1 - 3 is for KrPc
    # Axis 4 - xxx is for Eclipse profile
    if count_krpc_row:
        chart_x1 = [i * (0.8 / 3 + 0.1) for i in range(3)]
        chart_x2 = [x1 + 0.8 / 3 for x1 in chart_x1]
        for idx, title in enumerate(
            ["Relative Permeability", "Fractional Flow", "Capillary Pressure"]
        ):
            _dict[f"xaxis{idx+1}"] = {
                "anchor": f"y{idx+1}",
                "domain": [chart_x1[idx], chart_x2[idx]],
                "title": {"text": f"<b>{sat_axis_title}</b>"},
            }
            _dict[f"xaxis{idx+1}"].update(axis_format_dict)
            _dict[f"yaxis{idx+1}"] = {
                "anchor": f"x{idx+1}",
                "domain": [chart_y1[0], chart_y2[0]],
                "title": {"text": f"<b>{title}</b>"},
                "type": "log" if log_relperm and idx == 0 else "linear",
            }
            _dict[f"yaxis{idx+1}"].update(axis_format_dict)
            if idx > 0:
                _dict[f"xaxis{idx+1}"]["matches"] = "x"
    if count_profile_row:
        profile_x_axis = eclipse_vector_description(profile_x_axis)
        if profile_y_axis is None:
            profile_y_axis = []
        else:
            profile_y_axis = [eclipse_vector_description(x) for x in profile_y_axis]
        chart_x1 = [i * 0.55 for i in range(2)]
        chart_x2 = [_x1 + 0.45 for _x1 in chart_x1]
        for idx, title in enumerate(profile_y_axis):
            row, col = divmod(idx, 2)
            _dict[f"xaxis{idx+4}"] = {
                "anchor": f"y{idx+4}",
                "domain": [chart_x1[col], chart_x2[col]],
                "title": {"text": f"<b>{profile_x_axis}</b>"},
            }
            _dict[f"xaxis{idx+4}"].update(axis_format_dict)
            _dict[f"yaxis{idx+4}"] = {
                "anchor": f"x{idx+4}",
                "domain": [
                    chart_y1[count_krpc_row + row],
                    chart_y2[count_krpc_row + row],
                ],
                "title": {"text": f"<b>{title}</b>"},
            }
            _dict[f"yaxis{idx+4}"].update(axis_format_dict)
            if idx > 0:
                _dict[f"xaxis{idx+4}"]["matches"] = "x4"
    return _dict


# pylint:enable=too-many-locals


def toggle_relperm_axis(figure, semilog):
    if semilog:
        figure["layout"]["yaxis"]["type"] = "log"
    else:
        figure["layout"]["yaxis"]["type"] = "linear"
    return figure


def warning(message):
    print(f"{TERMINALCOLORS['WARNING']}{message}{TERMINALCOLORS['ENDC']}")


# pylint:disable=too-many-instance-attributes
class UpCaRsSimulationProfile(WebvizPluginABC):
    """
    Webviz plugin for displaying relative permeability and capillary pressure
    and corresponding Eclipse simulation profile
    Input:
        - ensembles: List of ensemble simulation
        - ensembles_idx: Ensemble index which is corresponding to list of
                         ensemble simulation. Needed to link krpc with ensembles
        - eclipse_references: List of Eclipse simulation run
        - column_keys: List of Eclipse summary keywords that are of interest
        - krpc_ensembles: CSV table generated using swof2csv,
                          containing SWOF/SGOF table for all ensembles
        - krpc_references: CSV table generated using swof2csv, containing
                           SWOF/SGOF table for individual case, assumed to have
                           same structure as krpc_ensembles CSV table
    Interactive:
    - Choose between SWOF and SGOF if the data is available
    - Choose SATNUM index if more than 1 available
    - Toggle y-axis for relative permeability : linear/logarithmic
    - Change opacity of ensemble curves
    - Choose x-axis parameter for simulation profile
    - Choose one or more y-axis parameters for simulation profile
    """

    # pylint:disable=too-many-arguments, too-many-branches, too-many-statements
    def __init__(
        self,
        app,
        x_axis=None,
        y_axis=None,
        ensembles=None,
        reference_cases=None,
        column_keys=None,
        krpc_ensembles=None,
        ensembles_idx=None,
        krpc_references=None,
    ):
        super().__init__()

        # Get setting from shared_settings
        shared_settings = app.webviz_settings["shared_settings"]

        self.plot_profile = ensembles or reference_cases
        self.plot_krpc = krpc_ensembles or krpc_references
        self.plot_ensembles = ensembles or krpc_ensembles
        self.plot_references = reference_cases or krpc_references
        self.x_axis = x_axis
        self.y_axis = [] if y_axis is None else y_axis

        self.column_keys = column_keys
        self.krpc_csv_tables = None
        self.references_tuple = ()
        self.case_tuple = ()
        self.ensemble_paths = ()
        self.colors = PALETTE["tableau"]
        if not (self.plot_profile or self.plot_krpc):
            raise ValueError(
                "Nothing to visualize.\n Please specify at least one Eclipse case or krpc table"
            )

        keywords = []
        if self.plot_profile:
            if ensembles is None:
                self.ensemble_paths = ()
                self.df_ens = None
            else:
                self.ensemble_paths = tuple(
                    (ensemble, shared_settings["scratch_ensembles"][ensemble])
                    for ensemble in ensembles
                )
                self.df_ens = get_ensemble_df(self.ensemble_paths, self.column_keys)
                keywords.extend(self.df_ens.columns)
            if reference_cases is None:
                warning("[UpCaRs Container] User didn't specify any reference cases")
                self.df_ref = None
            else:
                self.references_tuple = tuple(
                    (reference, shared_settings["realizations"][reference])
                    for reference in reference_cases
                )
                self.df_ref = get_summary_df(self.references_tuple, self.column_keys)
                keywords.extend(self.df_ref.columns)
            # Get all columns
            keywords.remove("REAL")
            keywords.remove("ENSEMBLE")
            keywords = sorted(list(set(keywords)))
            self.keywords_options = [
                {
                    "label": "{} ({})".format(
                        UDF_VECTOR.get(val, simulation_vector_description(val)), val
                    ),
                    "value": val,
                }
                for val in keywords
            ]
            if x_axis in keywords:
                self.x_axis = x_axis
            else:
                self.x_axis = keywords[0]
            self.y_axis = [key for key in y_axis if key in keywords]
        else:
            self.keywords_options = []
            self.x_axis = None
            self.y_axis = []

        if self.plot_krpc:
            if krpc_references:
                self.case_tuple = tuple(
                    (case, shared_settings["krpc_csv_tables"][case])
                    for case in krpc_references
                )
                self.df_ref_krpc = get_multiple_table_df(self.case_tuple)
            else:
                self.df_ref_krpc = None

            if krpc_ensembles:
                self.krpc_csv_tables = shared_settings["krpc_csv_tables"][
                    krpc_ensembles
                ]
                self.df_ens_krpc = get_table_df(self.krpc_csv_tables)
                # Create Iter column based on ENSEMBLE column
                self.df_ens_krpc["Iter"] = self.df_ens_krpc.apply(
                    lambda row: int(row["ENSEMBLE"][-1]), axis=1
                )
                if ensembles_idx is not None:
                    self.df_ens_krpc = self.df_ens_krpc[
                        self.df_ens_krpc["Iter"].isin(ensembles_idx)
                    ]
                if ensembles is None:
                    self.df_ens_krpc["ENSEMBLE"] = "iter-" + self.df_ens_krpc[
                        "Iter"
                    ].astype(str)
                    ensembles = self.df_ens_krpc["ENSEMBLE"].unique()
                else:
                    if len(ensembles) != len(ensembles_idx):
                        raise ValueError(
                            "Specified number of ensembles does not match with "
                            "number of ensemble index"
                        )
                    dict_ens = dict(zip(ensembles_idx, ensembles))
                    self.df_ens_krpc["ENSEMBLE"] = self.df_ens_krpc.apply(
                        lambda row: dict_ens.get(row["Iter"], None), axis=1
                    )
            else:
                self.df_ens_krpc = None
                ensembles = []

            self.satnum_list = []
            self.table_type = []

            if krpc_ensembles:
                self.satnum_list.extend(self.df_ens_krpc["satnum"].unique())
                self.table_type.extend(self.df_ens_krpc["type"].unique())
            if krpc_references:
                self.satnum_list.extend(self.df_ref_krpc["satnum"].unique())
                self.table_type.extend(self.df_ref_krpc["type"].unique())
            self.satnum_list = list(set(self.satnum_list))
            self.table_type = list(set(self.table_type))

        else:
            self.satnum_list = [None]
            self.table_type = [None]

        self.set_callbacks(app)

    # pylint:enable=too-many-arguments, too-many-branches, too-many-statements

    @property
    def tour_steps(self):
        return [
            {
                "id": self.uuid("layout"),
                "content": (
                    "Dashboard displaying saturation functions and "
                    "the corresponding flow response"
                ),
            },
            {
                "id": self.uuid("plot"),
                "content": (
                    "Chart showing saturation function "
                    "(relative permeability, fractional flow and "
                    "capillary pressure) and/or Eclipse simulation "
                    "profile."
                    "Different options can be set in the menu to the "
                    "left."
                    "Individual case can be highlighted by "
                    "clicking on any of the line in the chart."
                    "You can also toggle data on/off by clicking at "
                    "the legend."
                ),
            },
            {
                "id": self.uuid("x_axis_selector"),
                "content": ("Select Eclipse vector to be used in x-axis"),
            },
            {
                "id": self.uuid("y_axis_selector"),
                "content": (
                    "Select Eclipse vector to be used in y-axis. "
                    "Each vector will be visualized in a separate plot"
                ),
            },
            {
                "id": self.uuid("y_axis_selector"),
                "content": (
                    "Select Eclipse vector to be used in y-axis. "
                    "Each vector will be visualized in a separate plot"
                ),
            },
            {
                "id": self.uuid("satnum_selector"),
                "content": ("Select index of saturation function to be plotted"),
            },
            {
                "id": self.uuid("fluid_selector"),
                "content": (
                    "Select type of fluid system (SWOF: oil-water, "
                    "SGOF: oil-gas) to be plotted"
                ),
            },
            {
                "id": self.uuid("visc1_input"),
                "content": (
                    "Specify viscosity of water (in oil-water system) or "
                    "gas (in oil-gas system), to be used in "
                    "fractional flow calculation"
                ),
            },
            {
                "id": self.uuid("visc2_input"),
                "content": (
                    "Specify viscosity of oil, to be used in "
                    "fractional flow calculation"
                ),
            },
            {
                "id": self.uuid("axis_selector"),
                "content": (
                    "Switch between linear and logarithmic for "
                    "relative permeability axis."
                ),
            },
            {
                "id": self.uuid("opacity_selector"),
                "content": ("Specify the opacity of ensemble line plotting"),
            },
            {
                "id": self.uuid("reset"),
                "content": ("Clears the currently selected case"),
            },
        ]

    @property
    def layout(self):
        return wcc.FlexBox(
            id=self.uuid("layout"),
            children=[
                # Hidden object
                dcc.Store(id=self.uuid("reset_flag")),
                # Control
                html.Div(
                    id=self.uuid("control"),
                    style={"flex": "1"},
                    children=[
                        html.Label(
                            id=self.uuid("x_axis_selector"),
                            style={"display": "block" if self.plot_profile else "none"},
                            children=[
                                html.Span("X-axis", style={"font-weight": "bold"}),
                                dcc.Dropdown(
                                    id=self.uuid("x_axis"),
                                    clearable=False,
                                    options=self.keywords_options,
                                    value=self.x_axis,
                                    optionHeight=60,
                                ),
                            ],
                        ),
                        html.Label(
                            id=self.uuid("y_axis_selector"),
                            style={"display": "block" if self.plot_profile else "none"},
                            children=[
                                html.Span("Y-axis", style={"font-weight": "bold"}),
                                dcc.Dropdown(
                                    id=self.uuid("y_axis"),
                                    multi=True,
                                    options=self.keywords_options,
                                    value=self.y_axis,
                                    optionHeight=60,
                                ),
                            ],
                        ),
                        html.Label(
                            id=self.uuid("satnum_selector"),
                            style={"display": "block" if self.plot_krpc else "none"},
                            children=[
                                html.Span("SATNUM", style={"font-weight": "bold"}),
                                dcc.Dropdown(
                                    id=self.uuid("satnum"),
                                    clearable=False,
                                    options=[
                                        {"label": f"{val}", "value": val}
                                        for val in self.satnum_list
                                    ],
                                    value=self.satnum_list[0],
                                    disabled=len(self.satnum_list) == 1,
                                ),
                            ],
                        ),
                        html.Label(
                            id=self.uuid("fluid_selector"),
                            style={"display": "block" if self.plot_krpc else "none"},
                            children=[
                                html.Span("Type", style={"font-weight": "bold"}),
                                dcc.Dropdown(
                                    id=self.uuid("type"),
                                    options=[
                                        {"label": val, "value": val}
                                        for val in self.table_type
                                    ],
                                    value=self.table_type[0],
                                    clearable=False,
                                    disabled=len(self.table_type) == 1,
                                ),
                            ],
                        ),
                        html.Label(
                            id=self.uuid("visc1_input"),
                            style={"display": "block" if self.plot_krpc else "none"},
                            children=[
                                html.Span(
                                    "Water/Gas Viscosity", style={"font-weight": "bold"}
                                ),
                                dcc.Input(
                                    id=self.uuid("visc1"),
                                    style={"width": "97%"},
                                    value=1.0,
                                    type="number",
                                    debounce=True,
                                    placeholder="Water/Gas viscosity",
                                ),
                            ],
                        ),
                        html.Label(
                            id=self.uuid("visc2_input"),
                            style={"display": "block" if self.plot_krpc else "none"},
                            children=[
                                html.Span(
                                    "Oil Viscosity", style={"font-weight": "bold"}
                                ),
                                dcc.Input(
                                    id=self.uuid("visc2"),
                                    style={"width": "97%"},
                                    value=1.0,
                                    type="number",
                                    debounce=True,
                                    placeholder="Oil Viscosity",
                                ),
                            ],
                        ),
                        html.Label(
                            id=self.uuid("axis_selector"),
                            style={"display": "block" if self.plot_krpc else "none"},
                            children=[
                                html.Span(
                                    "Rel. Perm Plot", style={"font-weight": "bold"}
                                ),
                                dcc.RadioItems(
                                    id=self.uuid("axis"),
                                    options=[
                                        {"label": "Linear", "value": "linear",},
                                        {"label": "Semi-Log", "value": "log",},
                                    ],
                                    value="linear",
                                    labelStyle={"display": "inline-block"},
                                ),
                            ],
                        ),
                        html.Label(
                            id=self.uuid("opacity_selector"),
                            style={
                                "display": "block" if self.plot_ensembles else "none"
                            },
                            children=[
                                html.Span("Opacity", style={"font-weight": "bold"}),
                                dcc.Slider(
                                    id=self.uuid("opacity"),
                                    min=0.0,
                                    max=1.0,
                                    value=0.3,
                                    step=0.1,
                                    marks={
                                        val: {"label": f"{val:.1f}"}
                                        for val in [x * 0.2 for x in range(6)]
                                    },
                                ),
                            ],
                        ),
                        html.Button("Clear selected", id=self.uuid("reset")),
                    ],
                ),
                # Figures
                html.Div(
                    id=self.uuid("plot"),
                    style={"flex": "4"},
                    children=[wcc.Graph(id=self.uuid("figure"))],
                ),
            ],
        )

    def add_webvizstore(self):
        return [
            (
                get_ensemble_df,
                [
                    {
                        "ensemble_path": self.ensemble_paths,
                        "column_keys": self.column_keys,
                    }
                ],
            ),
            (
                get_summary_df,
                [
                    {
                        "case_paths": self.references_tuple,
                        "column_keys": self.column_keys,
                    }
                ],
            ),
            (get_table_df, [{"csv_table": self.krpc_csv_tables}]),
            (get_multiple_table_df, [{"tables": self.case_tuple}]),
        ]

    @staticmethod
    def simulation_profile_traces(data_frame, data_type, line_opacity, color_dict):
        """
        Create list of trace dictionary
        data_frame is a pandas dataframe with columns:
        ENSEMBLE, REAL, X_AXIS_COL, Y_AXIS1, Y_AXIS2, ... Y_AXIS_N

        style is namedtuple contains opacity and color
        """
        if data_frame is None:
            return []
        result = []
        for ens in data_frame["ENSEMBLE"].unique():
            color = color_dict[ens]
            style = TraceStyle(line_opacity, color)
            df_ens = data_frame[data_frame["ENSEMBLE"] == ens]
            for real in df_ens["REAL"].unique():
                df_real = df_ens[df_ens["REAL"] == real]
                for idx_param, param in enumerate(data_frame.columns[3:]):
                    result.append(
                        create_trace_dict(
                            [df_real[data_frame.columns[2]], df_real[param]],
                            TraceLabel(
                                f"Real: {real}" if data_type == "ens" else ens,
                                ens,
                                f"{ens}/{real}/{data_type}",
                            ),
                            style,
                            idx_param + 4,
                        ),
                    )
        return result

    @staticmethod
    def saturation_function_traces(
        data_frame, table_type, line_opacity, data_type, color_dict
    ):
        """
        Create list of trace dictionary
        """
        if data_frame is None:
            return []
        satfun_key = krpc_table_key(table_type)
        result = []
        for ens in data_frame["ENSEMBLE"].unique():
            color = color_dict.get(ens, None)
            style = TraceStyle(line_opacity, color)
            df_ens = data_frame[data_frame["ENSEMBLE"] == ens]
            for real in df_ens["REAL"].unique():
                df_real = df_ens[df_ens["REAL"] == real]
                result.append(
                    create_trace_dict(
                        [df_real[satfun_key[0]], df_real[satfun_key[1]]],
                        TraceLabel(
                            f"{satfun_key[1]} {ens}, Real {real}",
                            ens,
                            f"{ens}/{real}/{data_type}",
                        ),
                        style,
                        1,
                    )
                )

                result.append(
                    create_trace_dict(
                        [df_real[satfun_key[0]], df_real[satfun_key[2]]],
                        TraceLabel(
                            f"{satfun_key[2]} {ens}, Real {real}",
                            ens,
                            f"{ens}/{real}/{data_type}",
                        ),
                        style,
                        1,
                    )
                )

                result.append(
                    create_trace_dict(
                        [df_real[satfun_key[0]], df_real["fract_flow"]],
                        TraceLabel(
                            f"Fractional flow {ens}, Real {real}",
                            ens,
                            f"{ens}/{real}/{data_type}",
                        ),
                        style,
                        2,
                    )
                )

                result.append(
                    create_trace_dict(
                        [df_real[satfun_key[0]], df_real[satfun_key[3]]],
                        TraceLabel(
                            f"{satfun_key[3]} {ens}, Real {real}",
                            ens,
                            f"{ens}/{real}/{data_type}",
                        ),
                        style,
                        3,
                    )
                )
        return result

    def assign_series(self):
        """
        Create dictionary of tracer and assign color
        Reason for dummy tracer is to ensure legend opacity remains 1.0 despite
        the opacity setting from user

        Return values: color_dict and list of dummy tracer dictionary
        """
        color_dict = {}
        tracer_list = []
        color_list = itertools.cycle(self.colors)
        data_frames = []
        if self.plot_profile:
            data_frames.extend([self.df_ens, self.df_ref])
            axis_idx = 4
        if self.plot_krpc:
            data_frames.extend([self.df_ens_krpc, self.df_ref_krpc])
            axis_idx = 1

        for data_frame in data_frames:
            if data_frame is not None:
                for ens in data_frame["ENSEMBLE"].unique():
                    if color_dict.get(ens, None) is None:
                        color_dict[ens] = next(color_list)
                        tracer_list.append(
                            create_trace_dict(
                                [[None], [None]],
                                TraceLabel(ens, ens, ""),
                                TraceStyle(1.0, color_dict[ens]),
                                axis_idx,
                                True,
                            )
                        )
        return color_dict, tracer_list

    def set_callbacks(self, app):
        @app.callback(
            Output(self.uuid("plot"), "children"),
            [
                Input(self.uuid("x_axis"), "value"),
                Input(self.uuid("y_axis"), "value"),
                Input(self.uuid("satnum"), "value"),
                Input(self.uuid("type"), "value"),
                Input(self.uuid("visc1"), "value"),
                Input(self.uuid("visc2"), "value"),
            ],
            [State(self.uuid("opacity"), "value"), State(self.uuid("axis"), "value"),],
        )
        # pylint: disable=too-many-locals, too-many-arguments
        def _plot_figure(
            x_axis, y_axis, satnum, table_type, visc1, visc2, opacity, axis_type
        ):
            sat = ""
            if self.plot_krpc:
                sat, kr1, kr2, _ = krpc_table_key(table_type)
            layout = create_layout(sat, x_axis, y_axis, axis_type == "log")
            color_dict, data = self.assign_series()
            if self.plot_profile and len(y_axis) > 0:
                # Prepare Eclipse profile plot
                for data_frame, line_opacity, data_type in zip(
                    [self.df_ens, self.df_ref], [opacity, 1.0], ["ens", "ref"]
                ):
                    if data_frame is not None:
                        data.extend(
                            self.simulation_profile_traces(
                                data_frame[["ENSEMBLE", "REAL", x_axis] + y_axis],
                                data_type,
                                line_opacity,
                                color_dict,
                            )
                        )

            if self.plot_krpc:
                for data_frame, line_opacity, data_type in zip(
                    [self.df_ens_krpc, self.df_ref_krpc], [opacity, 1.0], ["ens", "ref"]
                ):
                    if data_frame is not None:
                        # Calculate fractional flow
                        data_frame["fract_flow"] = data_frame.apply(
                            lambda row: (row[kr1] / visc1)
                            / (row[kr1] / visc1 + row[kr2] / visc2)
                            if row[kr1] + row[kr2] > 1e-20
                            else 0.0,
                            axis=1,
                        )
                        data_frame = data_frame[data_frame["satnum"] == satnum]
                        data.extend(
                            self.saturation_function_traces(
                                data_frame,
                                table_type,
                                line_opacity,
                                data_type,
                                color_dict,
                            )
                        )
            return wcc.Graph(
                figure=go.Figure(data=data, layout=layout), id=self.uuid("figure")
            )

        # pylint: enable=too-many-locals

        @app.callback(
            [
                Output(self.uuid("figure"), "figure"),
                Output(self.uuid("reset_flag"), "data"),
            ],
            [
                Input(self.uuid("opacity"), "value"),
                Input(self.uuid("axis"), "value"),
                Input(self.uuid("figure"), "clickData"),
                Input(self.uuid("reset"), "n_clicks"),
            ],
            [
                State(self.uuid("figure"), "figure"),
                State(self.uuid("reset_flag"), "data"),
            ],
        )
        def _update_style(opacity, toggle_axis, click_data, _, figure, reset_mode):
            ctx = dash.callback_context.triggered
            if not (ctx and "data" in figure and "layout" in figure):
                raise PreventUpdate
            sender = ctx[0]["prop_id"].split(".")[0]

            if reset_mode is None or sender == self.uuid("reset"):
                reset_mode = True
            elif sender == self.uuid("figure"):
                reset_mode = False

            if sender in [
                self.uuid("opacity"),
                self.uuid("figure"),
                self.uuid("reset"),
            ]:
                if click_data and not reset_mode:
                    curve_idx = click_data["points"][0]["curveNumber"]
                    selected_meta = figure["data"][curve_idx]["meta"]
                else:
                    selected_meta = ""
                for trace in figure["data"]:
                    if trace["meta"] == selected_meta:
                        trace["opacity"] = 1.0
                        trace["mode"] = "lines+markers"
                    else:
                        if trace["meta"].endswith("/ens"):
                            trace["opacity"] = opacity
                        else:
                            trace["opacity"] = 1.0
                        trace["mode"] = "lines"
            elif sender == self.uuid("axis"):
                toggle_relperm_axis(figure, toggle_axis == "log")
                figure["layout"]["uirevision"] = str(random())  # nosec
            return figure, reset_mode
