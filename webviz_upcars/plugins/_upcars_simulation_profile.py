from uuid import uuid4

import dash_html_components as html
import dash_core_components as dcc
from dash.dependencies import Input, Output, State
from webviz_config import WebvizPluginABC
import plotly.graph_objs as go
from plotly.subplots import make_subplots
import pandas as pd
from webviz_config.webviz_store import webvizstore
from webviz_config.common_cache import CACHE
import webviz_core_components as wcc
import itertools
import io
import zipfile
import base64
import traceback
from _upcars_udf import *
import dash
from dash.exceptions import PreventUpdate
from random import random
import numpy as np


def warning(message):
    print(f'{bcolors.WARNING}{message}{bcolors.ENDC}')


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
            x_axis,
            y_axis,
            ensembles=[],
            reference_cases=None,
            column_keys=None,
            krpc_ensembles=None,
            ensembles_idx=[],
            krpc_references=None
    ):
        super().__init__()
        # Get setting from shared_settings
        shared_settings = app.webviz_settings["shared_settings"]
        self.make_uids()
        self.plot_profile = ensembles or reference_cases
        self.plot_krpc = krpc_ensembles or krpc_references
        self.plot_ensembles = ensembles or krpc_ensembles
        self.plot_references = reference_cases or krpc_references
        self.x_axis = x_axis
        self.y_axis = y_axis
        if not (self.plot_profile or self.plot_krpc):
            raise ValueError(
                "Nothing to visualize.\nPlease specify at least one Eclipse case or krpc table")

        self.ensembles = ensembles
        keywords = []
        if self.plot_profile:
            if self.ensembles == []:
                self.ensemble_paths = None
                self.df_ens = None
            else:
                self.ensemble_paths = tuple(
                    (ensemble, shared_settings["scratch_ensembles"][ensemble])
                    for ensemble in ensembles
                )
                self.df_ens = get_ensemble_df(self.ensemble_paths, column_keys)
                keywords.extend(self.df_ens.columns)
            if reference_cases is None:
                warning(
                    "[UpCaRs Container] User didn't specify any reference cases")
                self.df_ref = None
            else:
                references_tuple = tuple(
                    (reference, shared_settings["realizations"][reference])
                    for reference in reference_cases
                )
                self.df_ref = get_summary_df(references_tuple, column_keys)
                keywords.extend(self.df_ref.columns)
                #print (self.df_ref.columns)
            # Get all columns
            keywords.remove("REAL")
            keywords.remove("ENSEMBLE")
            self.keywords = sorted(list(set(keywords)))
            self.keywords_options = [
                {'label': val, 'value': val} for val in self.keywords]

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
                case_tuple = tuple(
                    (case, shared_settings["krpc_csv_tables"][case])
                    for case in krpc_references)
                self.df_ref_krpc = get_multiple_table_df(case_tuple)
            else:
                self.df_ref_krpc = None

            if krpc_ensembles:
                self.df_ens_krpc = get_table_df(
                    shared_settings["krpc_csv_tables"][krpc_ensembles])
                if ensembles_idx is not None:
                    self.df_ens_krpc = self.df_ens_krpc[self.df_ens_krpc['Iter'].isin(
                        ensembles_idx)]
                if self.ensembles == []:
                    self.df_ens_krpc["ENSEMBLE"] = "iter-" + \
                        self.df_ens_krpc["Iter"].astype(str)
                    self.ensembles = self.df_ens_krpc["ENSEMBLE"].unique()
                else:
                    if len(self.ensembles) != len(ensembles_idx):
                        raise ValueError(
                            "Specified number of ensembles does not match with number of ensemble index")
                    dict_ens = {key: value for key, value in zip(
                        ensembles_idx, self.ensembles)}
                    self.df_ens_krpc["ENSEMBLE"] = self.df_ens_krpc.apply(
                        lambda row: dict_ens.get(row["Iter"], None), axis=1)
                df = self.df_ens_krpc
            else:
                self.df_ens_krpc = None
                self.ensembles = []
                df = self.df_ref_krpc

            self.satnum_list = []
            self.table_type = []
            if krpc_ensembles:
                self.satnum_list.extend(self.df_ens_krpc['satnum'].unique())
                self.table_type.extend(self.df_ens_krpc['type'].unique())
            if krpc_references:
                self.satnum_list.extend(self.df_ref_krpc['satnum'].unique())
                self.table_type.extend(self.df_ref_krpc['type'].unique())

            self.satnum_list = list(set(self.satnum_list))
            self.table_type = list(set(self.table_type))
        else:
            self.satnum_list = [None]
            self.table_type = [None]

        self.set_callbacks(app)

    def make_uids(self):
        self.uid = f'{uuid4()}'
        self.plot_id = f'plot-{self.uid}'
        self.figure_id = f'figure-{self.uid}'

        self.x_axis_id = f'keyword-xaxis-{self.uid}'
        self.y_axis_id = f'keyword-yaxis-{self.uid}'

        self.toggle_axis_id = f'toggle_axis-{self.uid}'
        self.satnum_id = f'satnum_{self.uid}'
        self.table_type_id = f'table_type_{self.uid}'
        self.opacity_id = f'opacity-{self.uid}'
        self.reset_id = f'reset-{self.uid}'
        self.reset_flag_id = f'reset-flag-{self.uid}'

        self.visc1_id = f'visc1_{self.uid}'
        self.visc2_id = f'visc2_{self.uid}'

    @property
    def visc1_input(self):
        """Input box to specify viscosity of fluid1"""
        return html.Div(
            style={"paddingBottom": "5px"},
            children=[
                html.Label("Water/Gas Viscosity"),
                dcc.Input(
                    style={'width': '100%'},
                    id=self.visc1_id,
                    value=1.0,
                    type="number",
                    debounce=True,
                    placeholder="Water/Gas viscosity"
                ),
            ]
        )

    @property
    def visc2_input(self):
        """Input box to specify viscosity of fluid1"""
        return html.Div(
            style={"paddingBottom": "5px"},
            children=[
                html.Label("Oil Viscosity"),
                dcc.Input(
                    style={'width': '100%'},
                    value=1.0,
                    id=self.visc2_id,
                    type="number",
                    debounce=True,
                    placeholder="Oil viscosity"
                ),
            ]
        )


    @property
    def x_axis_selector(self):
        """Dropdown to select x-axis"""
        return html.Div(
            style={"paddingBottom": "5px"},
            children=[
                html.Label("X-axis"),
                dcc.Dropdown(
                    id=self.x_axis_id,
                    options=self.keywords_options,
                    value=self.x_axis,
                    clearable=False,
                    style={'height': '39px'},
                ),
            ]
        )

    @property
    def y_axis_selector(self):
        """Dropdown to select y-axis"""
        return html.Div(
            style={"paddingBottom": "5px"},
            children=[
                html.Label('Y-axis'),
                dcc.Dropdown(
                    id=self.y_axis_id,
                    options=self.keywords_options,
                    multi=True,
                    value=self.y_axis,
                    #style={'height': '39px'},
                ),
            ]
        )

    @property
    def toggle_type(self):
        """Dropdown to choose fluid combination type"""
        return html.Div(
            style={"paddingBottom": "5px"},
            children=[
                html.Label("Type"),
                dcc.Dropdown(
                    id=self.table_type_id,
                    options=[{'label': val, 'value': val}
                             for val in self.table_type],
                    value=self.table_type[0],
                    clearable=False,
                    disabled=len(self.table_type) == 1,
                    style={'height': '39px'},
                ),
            ]
        )

    @property
    def toggle_satnum(self):
        """Dropdown to choose satnum"""
        return html.Div(
            style={"paddingBottom": "5px"},
            children=[
                html.Label("SATNUM"),
                dcc.Dropdown(
                    id=self.satnum_id,
                    clearable=False,
                    options=[{'label': f'{val}', 'value': val}
                             for val in self.satnum_list],
                    value=self.satnum_list[0],
                    disabled=len(self.satnum_list) == 1,
                    style={'height': '39px'},
                ),
            ]
        )

    @property
    def toggle_axis(self):
        """Checkbox to toggle axis"""
        return html.Div(
            style={"paddingBottom": "5px",
                   "display": "block" if self.plot_krpc else "none"},
            children=[
                html.Label('Options'),
                dcc.Checklist(
                    id=self.toggle_axis_id,
                    options=[
                        {'label': 'Semilog Relative Permeability', 'value': 'log'}],
                    value=[],
                    labelStyle={'display': 'inline-block',
                                'height': '39px',
                                },
                ),
            ]
        )

    @property
    def opacity_selector(self):
        """Slider to adjust opacity"""
        return html.Div(
            style={"paddingBottom": "5px"},
            children=[
                html.Label('Opacity'),
                dcc.Slider(
                    id=self.opacity_id,
                    min=0.0,
                    max=1.0,
                    step=0.05,
                    value=0.25,
                    marks={
                        val: {'label': f'{val:.1f}'}
                        for val in [x*0.2 for x in range(6)]
                    },
                )
            ]
        )

    @property
    def reset_button(self):
        """Buton to reset selection"""
        return html.Div(
            children=[
                html.Button('Reset',
                            id=self.reset_id,
                            )
            ]
        )

    @property
    def layout(self):
        if self.plot_krpc and self.plot_profile:
            layout_style = "1fr 1fr 1fr 1fr"
        elif self.plot_krpc:
            layout_style = "1fr 1fr 1fr"
        else:
            layout_style = "1fr 1fr"

        return html.Div([
                        dcc.Store(id=self.reset_flag_id),
                        html.Div(
                            style=set_grid_layout(layout_style),
                            children=[
                                html.Div([self.x_axis_selector, self.y_axis_selector], style={
                                         'display': 'block' if self.plot_profile else 'none'}),
                                html.Div([self.toggle_satnum, self.toggle_type], style={
                                         'display': 'block' if self.plot_krpc else 'none'}),
                                html.Div([self.visc1_input, self.visc2_input], style={
                                         'display': 'block' if self.plot_krpc else 'none'}),
                                html.Div([self.toggle_axis, self.opacity_selector], style={
                                         'display': 'block' if self.plot_ensembles else 'none'}),
                            ]
                        ),
                        html.Div(
                            children=[
                                self.reset_button,
                                html.Div(
                                    id=self.plot_id,
                                    children=[
                                        wcc.Graph(
                                            id=self.figure_id, figure={}),
                                    ]
                                )
                            ]
                        )
                        ])

    def toggle_relperm_axis(self, figure, semilog):
        if semilog:
            figure["layout"]["yaxis"]["type"] = "log"
        else:
            figure["layout"]["yaxis"]["type"] = "linear"
        return figure

    def create_dummy_trace_dict(self, legend_name, color, xaxis, yaxis):
        return {
            'x': [None],
            'y': [None],
            'legendgroup': legend_name,
            'name': legend_name,
            'mode': 'lines',
            'type': 'scattergl',
            'xaxis': xaxis,
            'yaxis': yaxis,
            'opacity': 1.0,
            'showlegend': True,
            'meta': 'dummy',
            'line': {'color': color},
        }

    def create_layout(self,
                      sat_axis_title=None,
                      profile_x_axis="",
                      profile_y_axis=[]
                      ):
        krpc_height = 400  # 600.0
        profile_height = 300  # 450.0
        spacing_height = 100.0
        profile_col_count = 2

        axis_format_dict = {
            'gridcolor': 'LightGray',
            'gridwidth': 1,
            'linecolor': 'black',
            'linewidth': 1,
            'mirror': True,
            'showgrid': True,
            'showline': True,
            'zeroline': True,
            'zerolinecolor': 'LightGray',
        }

        count_krpc_row = 1 if sat_axis_title else 0
        count_profile_row = int(
            (len(profile_y_axis)+profile_col_count-1)/profile_col_count)
        count_total_row = count_krpc_row + count_profile_row
        if count_total_row == 0:
            print(f"{bcolors.OKBLUE}There is nothing to plot{bcolors.ENDC}")

        figure_height = count_krpc_row * krpc_height + count_profile_row * \
            profile_height + (count_total_row-1)*spacing_height

        _dict = {'height': figure_height,
                 'paper_bgcolor': 'white',
                 'plot_bgcolor': 'white',
                 'uirevision': str(random())}

        # Build bottom-up, start with profile
        y1 = [i*(profile_height + spacing_height) /
              figure_height for i in range(count_profile_row)]
        y2 = [_y1 + profile_height/figure_height for _y1 in y1]

        warning(f'y1: {y1}')
        warning(f'y2: {y2}')
        if count_krpc_row:
            if len(y1) == 0:
                y1 = [0]
                y2 = [1]
            else:
                y1.append(y2[-1] + spacing_height/figure_height)
                y2.append(y1[-1] + krpc_height/figure_height)

        y1.reverse()
        y2.reverse()

        # Axis 1 - 3 is for KrPc
        # Axis 4 - xxx is for Eclipse profile
        if count_krpc_row:
            spacing_col = 0.2/2
            chart_width = (1.0 - 2*spacing_col)/3
            x1 = [i*(chart_width + spacing_col) for i in range(3)]
            x2 = [_x1 + chart_width for _x1 in x1]
            for idx, title in enumerate(['Relative Permeability', 'Fractional Flow', 'Capillary Pressure']):
                _dict[f'xaxis{idx+1}'] = {
                    'anchor': f'y{idx+1}',
                    'domain': [x1[idx], x2[idx]],
                    'title': {'text': f'<b>{sat_axis_title}</b>'},
                }
                _dict[f'xaxis{idx+1}'].update(axis_format_dict)
                _dict[f'yaxis{idx+1}'] = {
                    'anchor': f'x{idx+1}',
                    'domain': [y1[0], y2[0]],
                    'title': {'text': f'<b>{title}</b>'},
                }
                _dict[f'yaxis{idx+1}'].update(axis_format_dict)
                if idx > 0:
                    _dict[f'xaxis{idx+1}']['matches'] = 'x'
        if count_profile_row:
            spacing_col = 0.2/profile_col_count
            chart_width = (1.0 - (profile_col_count-1) *
                           spacing_col)/profile_col_count
            x1 = [i*(chart_width + spacing_col)
                  for i in range(profile_col_count)]
            x2 = [_x1 + chart_width for _x1 in x1]
            for idx, title in enumerate(profile_y_axis):
                row, col = divmod(idx, profile_col_count)
                _dict[f'xaxis{idx+4}'] = {
                    'anchor': f'y{idx+4}',
                    'domain': [x1[col], x2[col]],
                    'title': {'text': f'<b>{profile_x_axis}</b>'},
                }
                _dict[f'xaxis{idx+4}'].update(axis_format_dict)
                _dict[f'yaxis{idx+4}'] = {
                    'anchor': f'x{idx+4}',
                    'domain': [y1[count_krpc_row + row], y2[count_krpc_row + row]],
                    'title': {'text': f'<b>{title}</b>'},
                }
                _dict[f'yaxis{idx+4}'].update(axis_format_dict)
                if idx > 0:
                    _dict[f'xaxis{idx+4}']['matches'] = 'x4'
        return _dict

    def set_callbacks(self, app):
        @app.callback(self.plugin_data_output,
                      [self.plugin_data_requested],
                      [
                          State(self.figure_id, 'figure'),
                          State(self.x_axis_id, 'value'),
                          State(self.y_axis_id, 'value'),
                          State(self.satnum_id, 'value'),
                          State(self.table_type_id, 'value'),
                      ])
        def _user_download_data(data_requested, figure, x_axis, y_axis, satnum, table_type):
            warning('User Download Data')

            # TODO: Extract data directly from pandas dataframe
            if (not data_requested) or (not "data" in figure):
                return ''

            #keys = self.dict_table.get(table_type, self.dict_table['SWOF'])

            data = figure["data"]
            entries = set()
            for i, trace in enumerate(figure["data"]):
                warning(f'{trace["meta"]} : {trace.get("visible",True)}')
                if trace.get('visible', True) and trace["meta"] != "dummy":
                    entries.add(trace['meta'])

            file_list = []
            for entry in entries:
                ens, real, type = entry.split("/")
                if self.plot_krpc:
                    pass

            byte_io = io.BytesIO()
            with zipfile.ZipFile(byte_io, 'w', compression=zipfile.ZIP_DEFLATED) as zipped_data:
                for data in file_list:
                    zipped_data.writestr(data["filename"], data["content"])
            byte_io.seek(0)
            return base64.b64encode(byte_io.read()).decode("ascii")

        @app.callback(Output(self.plot_id, 'children'),
                      [Input(self.x_axis_id, 'value'),
                       Input(self.y_axis_id, 'value'),
                       Input(self.satnum_id, 'value'),
                       Input(self.table_type_id, 'value'),
                       Input(self.visc1_id, 'value'),
                       Input(self.visc2_id, 'value'),
                       ],
                      [
            State(self.opacity_id, 'value'),
            State(self.toggle_axis_id, 'value'),
        ])
        def plot_figure(x_axis, y_axis, satnum, table_type, visc1, visc2, opacity, axis_type):
            warning("Plot Figure")
            # if not dash.callback_context.triggered:
            #     raise PreventUpdate
            sat = ""
            if self.plot_krpc:
                sat, kr1, kr2, pc = krpc_table_key(table_type)

            layout = self.create_layout(sat, x_axis, y_axis)
            data = []
            color_list = itertools.cycle(palette.tableau_flip)
            #color_list = itertools.cycle(palette.tableau_20)

            color_dict = {}

            if self.plot_profile:
                # Prepare Eclipse profile plot
                for df, line_opacity, data_type in zip([self.df_ens, self.df_ref],
                                                       [opacity, 1.0], ["ens", "ref"]):
                    if df is not None and len(y_axis) > 0:
                        for idx_ens, ens in enumerate(df['ENSEMBLE'].unique()):
                            color = color_dict.get(ens, None)
                            if color is None:
                                color = next(color_list)
                                color_dict[ens] = color
                                showlegend = True
                            else:
                                showlegend = False
                            df_ens = df[df['ENSEMBLE'] == ens]
                            for idx_real, real in enumerate(df_ens['REAL'].unique()):
                                df_ens_real = df_ens[df_ens['REAL'] == real]
                                showlegend = showlegend and idx_real == 0
                                for idx_param, param in enumerate(y_axis):
                                    showlegend = showlegend and idx_param == 0
                                    if showlegend:
                                        data.append(self.create_dummy_trace_dict(
                                            ens, color, 'x4', 'y4'))
                                    data.append(create_trace_dict(df_ens_real[x_axis],
                                                                  df_ens_real[param],
                                                                  f'Realization: {real}' if data_type == "ens" else ens,
                                                                  ens,
                                                                  line_opacity,
                                                                  False,
                                                                  color,
                                                                  f'{ens}/{real}/{data_type}',
                                                                  f'x{idx_param+4}',
                                                                  f'y{idx_param+4}'))

            if self.plot_krpc:
                for df, line_opacity, data_type in zip([self.df_ens_krpc, self.df_ref_krpc],
                                                       [opacity, 1.0], ["ens", "ref"]):
                    if df is not None:
                        # Calculate fractional flow
                        df['fract_flow'] = df.apply(lambda row: (
                            row[kr1]/visc1)/(row[kr1]/visc1 + row[kr2]/visc2), axis=1)
                        df = df[df['satnum'] == satnum]
                        for idx_ens, ens in enumerate(df['ENSEMBLE'].unique()):
                            color = color_dict.get(ens, None)
                            if color is None:
                                color = next(color_list)
                                color_dict[ens] = color
                                showlegend = True
                            else:
                                showlegend = False

                            df_ens = df[df['ENSEMBLE'] == ens]
                            for idx_real, real in enumerate(df_ens['Realization'].unique()):
                                df_real = df_ens[df_ens['Realization'] == real]

                                if showlegend and idx_real == 0:
                                    data.append(self.create_dummy_trace_dict(
                                        ens, color, 'x', 'y'))

                                data.append(create_trace_dict(*df_real[[sat, kr1]].T.values,
                                                              f'{kr1} {ens}, Real {real}',
                                                              ens,
                                                              line_opacity,
                                                              False,
                                                              color,
                                                              f'{ens}/{real}/{data_type}',
                                                              'x1', 'y1')
                                            )
                                data.append(create_trace_dict(*df_real[[sat, 'fract_flow']].T.values,
                                                              f'Fractional flow {ens}, Real {real}',
                                                              ens,
                                                              line_opacity,
                                                              False,
                                                              color,
                                                              f'{ens}/{real}/{data_type}',
                                                              'x2', 'y2')
                                            )

                                data.append(create_trace_dict(*df_real[[sat, kr2]].T.values,
                                                              f'{kr2} {ens}, Real {real}',
                                                              ens,
                                                              line_opacity,
                                                              False,
                                                              color,
                                                              f'{ens}/{real}/{data_type}', 'x1', 'y1'))
                                data.append(create_trace_dict(*df_real[[sat, pc]].T.values,
                                                              f'{pc} {ens} Real {real}',
                                                              ens,
                                                              line_opacity,
                                                              False,
                                                              color,
                                                              f'{ens}/{real}/{data_type}', 'x3', 'y3'))

            return wcc.Graph(figure=go.Figure(data=data, layout=layout), id=self.figure_id)

        @app.callback([Output(self.figure_id, 'figure'),
                       Output(self.reset_flag_id, 'data')
                       ],
                      [Input(self.opacity_id, 'value'),
                       Input(self.toggle_axis_id, 'value'),
                       Input(self.figure_id, "clickData"),
                       Input(self.reset_id, "n_clicks")
                       ],
                      [State(self.figure_id, 'figure'),
                       State(self.reset_flag_id, 'data')])
        def _update_style(opacity, toggle_axis, clickData, reset, figure, reset_mode):
            ctx = dash.callback_context.triggered
            if not ctx:
                raise PreventUpdate
            sender = ctx[0]['prop_id'].split('.')[0]
            if reset_mode is None:
                reset_mode = True
            if sender == self.figure_id:
                reset_mode = False
            elif sender == self.reset_id:
                reset_mode = True

            if sender in [self.opacity_id, self.figure_id, self.reset_id]:
                if clickData and not reset_mode:
                    curve_idx = clickData["points"][0]["curveNumber"]
                    selected_meta = figure["data"][curve_idx]["meta"]
                    reference_opacity = 0.3
                    ensemble_opacity = min(0.2, 0.5*opacity)
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
            elif sender == self.toggle_axis_id:
                if "layout" in figure:
                    self.toggle_relperm_axis(figure, len(toggle_axis) > 0)
                    figure["layout"]["uirevision"] = str(random())
            return figure, reset_mode
