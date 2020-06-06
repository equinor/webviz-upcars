import itertools
from random import random

import dash_html_components as html
import dash_core_components as dcc
from dash.dependencies import Input, Output, State
from webviz_config import WebvizPluginABC
import plotly.graph_objs as go
import webviz_core_components as wcc
import dash
from dash.exceptions import PreventUpdate

# from ._upcars_udf import *
from ._upcars_udf import (
    bcolors,
    create_trace_dict,
    krpc_table_key,
    get_table_df,
    get_multiple_table_df,
    get_ensemble_df,
    get_summary_df,
)


def warning(message):
    print(f"{bcolors.WARNING}{message}{bcolors.ENDC}")


class UpCaRsSimulationProfile(WebvizPluginABC):
    """
    Plot relative permeability, capillary pressure and Eclipse simulation profile
    Webviz container for displaying relative permeability and capillary pressure and corresponding Eclipse simulation profile
    Input:
        - ensembles: List of ensemble simulation
        - ensembles_idx: Ensemble index which is corresponding to list of ensemble simulation. Needed to link krpc with ensembles
        - eclipse_references: List of Eclipse simulation run
        - column_keys: List of Eclipse summary keywords that are of interest
        - krpc_ensembles: CSV table generated using swof2csv, containing SWOF/SGOF table for all ensembles
        - krpc_references: CSV table generated using swof2csv, containing SWOF/SGOF table for individual case, assumed to have
                     same structure as krpc_ensembles CSV table
    Interactive:
    - Choose between SWOF and SGOF if the data is available
    - Choose SATNUM index if more than 1 available
    - Toggle y-axis for relative permeability : linear/logarithmic
    - Change opacity of ensemble curves
    - Choose x-axis parameter for simulation profile
    - Choose one or more y-axis parameters for simulation profile
    """

    def __init__(
        self,
        app,
        x_axis=None,
        y_axis=[],
        ensembles=[],
        reference_cases=None,
        column_keys=None,
        krpc_ensembles=None,
        ensembles_idx=None,
        krpc_references=None,
    ):
        super().__init__()
        # Get setting from shared_settings
        self.shared_settings = app.webviz_settings["shared_settings"]
        self.plot_profile = ensembles or reference_cases
        self.plot_krpc = krpc_ensembles or krpc_references
        self.plot_ensembles = ensembles or krpc_ensembles
        self.plot_references = reference_cases or krpc_references
        self.column_keys = column_keys
        self.x_axis = x_axis
        self.y_axis = y_axis
        self.krpc_ensembles = krpc_ensembles
        self.krpc_csv_tables = None
        self.references_tuple = ()
        self.case_tuple = ()
        self.ensemble_paths = ()
        self.colors = [
            "rgb(31, 119, 180)",
            "rgb(255, 127, 14)",
            "rgb(44, 160, 44)",
            "rgb(214, 39, 40)",
            "rgb(148, 103, 189)",
            "rgb(140, 86, 75)",
            "rgb(227, 119, 194)",
            "rgb(127, 127, 127)",
            "rgb(188, 189, 34)",
            "rgb(23, 190, 207)",
        ]
        if not (self.plot_profile or self.plot_krpc):
            raise ValueError(
                "Nothing to visualize.\n Please specify at least one Eclipse case or krpc table"
            )

        self.ensembles = ensembles
        keywords = []
        if self.plot_profile:
            if self.ensembles == []:
                self.ensemble_paths = ()
                self.df_ens = None
            else:
                self.ensemble_paths = tuple(
                    (ensemble, self.shared_settings["scratch_ensembles"][ensemble])
                    for ensemble in ensembles
                )
                self.df_ens = get_ensemble_df(self.ensemble_paths, self.column_keys)
                keywords.extend(self.df_ens.columns)
            if reference_cases is None:
                warning("[UpCaRs Container] User didn't specify any reference cases")
                self.df_ref = None
            else:
                self.references_tuple = tuple(
                    (reference, self.shared_settings["realizations"][reference])
                    for reference in reference_cases
                )
                self.df_ref = get_summary_df(self.references_tuple, self.column_keys)
                keywords.extend(self.df_ref.columns)
            # Get all columns
            keywords.remove("REAL")
            keywords.remove("ENSEMBLE")
            self.keywords = sorted(list(set(keywords)))
            self.keywords_options = [
                {"label": val, "value": val} for val in self.keywords
            ]

            if x_axis in self.keywords:
                self.x_axis = x_axis
            else:
                self.x_axis = self.keywords[0]
            self.y_axis = [key for key in y_axis if key in self.keywords]
        else:
            self.keywords_options = []
            self.keywords = [None]
            self.x_axis = None
            self.y_axis = []

        if self.plot_krpc:
            if krpc_references:
                self.case_tuple = tuple(
                    (case, self.shared_settings["krpc_csv_tables"][case])
                    for case in krpc_references
                )
                self.df_ref_krpc = get_multiple_table_df(self.case_tuple)
            else:
                self.df_ref_krpc = None

            if self.krpc_ensembles:
                self.krpc_csv_tables = self.shared_settings["krpc_csv_tables"][
                    self.krpc_ensembles
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
                if self.ensembles == []:
                    self.df_ens_krpc["ENSEMBLE"] = "iter-" + self.df_ens_krpc[
                        "Iter"
                    ].astype(str)
                    self.ensembles = self.df_ens_krpc["ENSEMBLE"].unique()
                else:
                    if len(self.ensembles) != len(ensembles_idx):
                        raise ValueError(
                            "Specified number of ensembles does not match with number of ensemble index"
                        )
                    dict_ens = dict(zip(ensembles_idx, self.ensembles))
                    self.df_ens_krpc["ENSEMBLE"] = self.df_ens_krpc.apply(
                        lambda row: dict_ens.get(row["Iter"], None), axis=1
                    )
                df = self.df_ens_krpc
            else:
                self.df_ens_krpc = None
                self.ensembles = []
                df = self.df_ref_krpc

            self.satnum_list = []
            self.table_type = []

            if self.krpc_ensembles:
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
                                    style={"width": "98%"},
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
                                    style={"width": "98%"},
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
                        html.Button("Resets", id=self.uuid("reset")),
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

    def toggle_relperm_axis(self, figure, semilog):
        if semilog:
            figure["layout"]["yaxis"]["type"] = "log"
        else:
            figure["layout"]["yaxis"]["type"] = "linear"
        return figure

    def create_dummy_trace_dict(self, legend_name, color, xaxis, yaxis):
        return {
            "x": [None],
            "y": [None],
            "legendgroup": legend_name,
            "name": legend_name,
            "mode": "lines",
            "type": "scattergl",
            "xaxis": xaxis,
            "yaxis": yaxis,
            "opacity": 1.0,
            "showlegend": True,
            "meta": "dummy",
            "line": {"color": color},
        }

    def create_layout(self, sat_axis_title=None, profile_x_axis="", profile_y_axis=[]):
        krpc_height = 400  # 600.0
        profile_height = 300  # 450.0
        spacing_height = 100.0
        profile_col_count = 2

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
        count_profile_row = int(
            (len(profile_y_axis) + profile_col_count - 1) / profile_col_count
        )
        count_total_row = count_krpc_row + count_profile_row

        figure_height = (
            count_krpc_row * krpc_height
            + count_profile_row * profile_height
            + (count_total_row - 1) * spacing_height
        )

        _dict = {
            "height": figure_height,
            "paper_bgcolor": "white",
            "plot_bgcolor": "white",
            "uirevision": str(random()),
        }

        # Build bottom-up, start with profile
        y1 = [
            i * (profile_height + spacing_height) / figure_height
            for i in range(count_profile_row)
        ]
        y2 = [_y1 + profile_height / figure_height for _y1 in y1]
        if count_krpc_row:
            if len(y1) == 0:
                y1 = [0]
                y2 = [1]
            else:
                y1.append(y2[-1] + spacing_height / figure_height)
                y2.append(y1[-1] + krpc_height / figure_height)

        y1.reverse()
        y2.reverse()

        # Axis 1 - 3 is for KrPc
        # Axis 4 - xxx is for Eclipse profile
        if count_krpc_row:
            spacing_col = 0.2 / 2
            chart_width = (1.0 - 2 * spacing_col) / 3
            x1 = [i * (chart_width + spacing_col) for i in range(3)]
            x2 = [_x1 + chart_width for _x1 in x1]
            for idx, title in enumerate(
                ["Relative Permeability", "Fractional Flow", "Capillary Pressure"]
            ):
                _dict[f"xaxis{idx+1}"] = {
                    "anchor": f"y{idx+1}",
                    "domain": [x1[idx], x2[idx]],
                    "title": {"text": f"<b>{sat_axis_title}</b>"},
                }
                _dict[f"xaxis{idx+1}"].update(axis_format_dict)
                _dict[f"yaxis{idx+1}"] = {
                    "anchor": f"x{idx+1}",
                    "domain": [y1[0], y2[0]],
                    "title": {"text": f"<b>{title}</b>"},
                }
                _dict[f"yaxis{idx+1}"].update(axis_format_dict)
                if idx > 0:
                    _dict[f"xaxis{idx+1}"]["matches"] = "x"
        if count_profile_row:
            spacing_col = 0.2 / profile_col_count
            chart_width = (
                1.0 - (profile_col_count - 1) * spacing_col
            ) / profile_col_count
            x1 = [i * (chart_width + spacing_col) for i in range(profile_col_count)]
            x2 = [_x1 + chart_width for _x1 in x1]
            for idx, title in enumerate(profile_y_axis):
                row, col = divmod(idx, profile_col_count)
                _dict[f"xaxis{idx+4}"] = {
                    "anchor": f"y{idx+4}",
                    "domain": [x1[col], x2[col]],
                    "title": {"text": f"<b>{profile_x_axis}</b>"},
                }
                _dict[f"xaxis{idx+4}"].update(axis_format_dict)
                _dict[f"yaxis{idx+4}"] = {
                    "anchor": f"x{idx+4}",
                    "domain": [y1[count_krpc_row + row], y2[count_krpc_row + row]],
                    "title": {"text": f"<b>{title}</b>"},
                }
                _dict[f"yaxis{idx+4}"].update(axis_format_dict)
                if idx > 0:
                    _dict[f"xaxis{idx+4}"]["matches"] = "x4"
        return _dict

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
        def plot_figure(
            x_axis, y_axis, satnum, table_type, visc1, visc2, opacity, axis_type
        ):
            if not dash.callback_context.triggered:
                raise PreventUpdate
            sat = ""
            if self.plot_krpc:
                sat, kr1, kr2, pc = krpc_table_key(table_type)

            layout = self.create_layout(sat, x_axis, y_axis)
            data = []
            color_list = itertools.cycle(self.colors)

            color_dict = {}

            if self.plot_profile:
                # Prepare Eclipse profile plot
                for df, line_opacity, data_type in zip(
                    [self.df_ens, self.df_ref], [opacity, 1.0], ["ens", "ref"]
                ):
                    if df is not None and len(y_axis) > 0:
                        for idx_ens, ens in enumerate(df["ENSEMBLE"].unique()):
                            color = color_dict.get(ens, None)
                            if color is None:
                                color = next(color_list)
                                color_dict[ens] = color
                                showlegend = True
                            else:
                                showlegend = False
                            df_ens = df[df["ENSEMBLE"] == ens]
                            for idx_real, real in enumerate(df_ens["REAL"].unique()):
                                df_ens_real = df_ens[df_ens["REAL"] == real]
                                showlegend = showlegend and idx_real == 0
                                for idx_param, param in enumerate(y_axis):
                                    showlegend = showlegend and idx_param == 0
                                    if showlegend:
                                        data.append(
                                            self.create_dummy_trace_dict(
                                                ens, color, "x4", "y4"
                                            )
                                        )
                                    data.append(
                                        create_trace_dict(
                                            df_ens_real[x_axis],
                                            df_ens_real[param],
                                            f"Realization: {real}"
                                            if data_type == "ens"
                                            else ens,
                                            ens,
                                            line_opacity,
                                            False,
                                            color,
                                            f"{ens}/{real}/{data_type}",
                                            f"x{idx_param+4}",
                                            f"y{idx_param+4}",
                                        )
                                    )

            if self.plot_krpc:
                for df, line_opacity, data_type in zip(
                    [self.df_ens_krpc, self.df_ref_krpc], [opacity, 1.0], ["ens", "ref"]
                ):
                    if df is not None:
                        # Calculate fractional flow
                        df["fract_flow"] = df.apply(
                            lambda row: (row[kr1] / visc1)
                            / (row[kr1] / visc1 + row[kr2] / visc2),
                            axis=1,
                        )
                        df = df[df["satnum"] == satnum]
                        for idx_ens, ens in enumerate(df["ENSEMBLE"].unique()):
                            color = color_dict.get(ens, None)
                            if color is None:
                                color = next(color_list)
                                color_dict[ens] = color
                                showlegend = True
                            else:
                                showlegend = False

                            df_ens = df[df["ENSEMBLE"] == ens]
                            for idx_real, real in enumerate(df_ens["REAL"].unique()):
                                df_real = df_ens[df_ens["REAL"] == real]

                                if showlegend and idx_real == 0:
                                    data.append(
                                        self.create_dummy_trace_dict(
                                            ens, color, "x", "y"
                                        )
                                    )

                                data.append(
                                    create_trace_dict(
                                        *df_real[[sat, kr1]].T.values,
                                        f"{kr1} {ens}, Real {real}",
                                        ens,
                                        line_opacity,
                                        False,
                                        color,
                                        f"{ens}/{real}/{data_type}",
                                        "x1",
                                        "y1",
                                    )
                                )
                                data.append(
                                    create_trace_dict(
                                        *df_real[[sat, "fract_flow"]].T.values,
                                        f"Fractional flow {ens}, Real {real}",
                                        ens,
                                        line_opacity,
                                        False,
                                        color,
                                        f"{ens}/{real}/{data_type}",
                                        "x2",
                                        "y2",
                                    )
                                )

                                data.append(
                                    create_trace_dict(
                                        *df_real[[sat, kr2]].T.values,
                                        f"{kr2} {ens}, Real {real}",
                                        ens,
                                        line_opacity,
                                        False,
                                        color,
                                        f"{ens}/{real}/{data_type}",
                                        "x1",
                                        "y1",
                                    )
                                )
                                data.append(
                                    create_trace_dict(
                                        *df_real[[sat, pc]].T.values,
                                        f"{pc} {ens} Real {real}",
                                        ens,
                                        line_opacity,
                                        False,
                                        color,
                                        f"{ens}/{real}/{data_type}",
                                        "x3",
                                        "y3",
                                    )
                                )
            return wcc.Graph(
                figure=go.Figure(data=data, layout=layout), id=self.uuid("figure")
            )

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
        def _update_style(opacity, toggle_axis, clickData, reset, figure, reset_mode):
            ctx = dash.callback_context.triggered
            if not ctx:
                raise PreventUpdate
            sender = ctx[0]["prop_id"].split(".")[0]
            if reset_mode is None:
                reset_mode = True
            if sender == self.uuid("figure"):
                reset_mode = False
            elif sender == self.uuid("reset"):
                reset_mode = True

            if sender in [
                self.uuid("opacity"),
                self.uuid("figure"),
                self.uuid("reset"),
            ]:
                if clickData and not reset_mode:
                    curve_idx = clickData["points"][0]["curveNumber"]
                    selected_meta = figure["data"][curve_idx]["meta"]
                    reference_opacity = 0.2
                    ensemble_opacity = min(0.2, 0.5 * opacity)
                else:
                    selected_meta = ""
                    reference_opacity = 1.0
                    ensemble_opacity = opacity
                if "data" in figure:
                    for trace in figure["data"]:
                        if trace["meta"] == "dummy":
                            pass
                        elif trace["meta"] != selected_meta:
                            if trace["name"] in self.ensembles:
                                trace["opacity"] = ensemble_opacity
                            else:
                                trace["opacity"] = reference_opacity
                        else:
                            trace["opacity"] = 1.0
            elif sender == self.uuid("axis"):
                if "layout" in figure:
                    self.toggle_relperm_axis(figure, toggle_axis == "log")
                    figure["layout"]["uirevision"] = str(random())
            return figure, reset_mode
