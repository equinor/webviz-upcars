from pathlib import Path

import dash_html_components as html
import webviz_core_components as wcc
import plotly.graph_objs as go

from webviz_config import WebvizPluginABC
from .._util.palette import PALETTE
from .._util.fmu_input import get_table_df, load_parameters


def create_best_trace(subplot_idx, label, value, legend=False):
    return {
        "type": "scatter",
        "x": value,
        "y": [label],
        "xaxis": f"x{subplot_idx}",
        "yaxis": f"y{subplot_idx*2}",
        "name": "Best realization",
        "showlegend": legend,
        "mode": "markers",
        "marker": {"symbol": "x", "color": "black"},
        "legendgroup": "Best realization",
    }


def create_hist_trace(subplot_idx, data, color, name, showlegend):
    return [
        {
            "legendgroup": name,
            "marker": {"color": color, "line": {"width": 2}},
            "bingroup": subplot_idx,
            "nbinsx": 10,
            "name": name,
            "opacity": 0.3,
            "showlegend": showlegend,
            "type": "histogram",
            "x": data,
            "xaxis": f"x{subplot_idx}",
            "yaxis": f"y{subplot_idx*2-1}",
        },
        {
            "legendgroup": name,
            "name": name,
            "marker": {"color": color},
            "notched": True,
            "showlegend": False,
            "type": "box",
            "x": data,
            "xaxis": f"x{subplot_idx}",
            "yaxis": f"y{subplot_idx*2}",
        },
    ]


def calculate_chart_coordinate(n_rows, n_cols, box_plot_height, hist_plot_height):
    """
    Calculate chart coordinate point
    --------------   y3
    |  box_plot  |
    --------------   y2
    |  hist_plot |
    --------------   y1
    x1          x2
    """
    spacing_col = 0.2 / n_cols
    spacing_row = 0.3 / n_rows
    chart_height = (1.0 - (n_rows - 1) * spacing_row) / n_rows
    chart_width = (1.0 - (n_cols - 1) * spacing_col) / n_cols
    # Normalize for use in dictionary
    box_height = box_plot_height / (box_plot_height + hist_plot_height) * chart_height
    hist_height = hist_plot_height / (box_plot_height + hist_plot_height) * chart_height

    chart_x1 = [i * (chart_width + spacing_col) for i in range(n_cols)]
    chart_x2 = [x + chart_width for x in chart_x1]

    chart_y1 = [i * (chart_height + spacing_row) for i in range(n_rows)]
    chart_y2 = [y + hist_height for y in chart_y1]
    chart_y3 = [y + box_height for y in chart_y2]

    return [chart_x1, chart_x2], [chart_y1, chart_y2, chart_y3]


def create_hist_layout(parameters, n_cols=4, box_plot_height=100, hist_plot_height=300):
    n_rows = int((len(parameters) + n_cols - 1) / n_cols)
    figure_height = n_rows * (box_plot_height + hist_plot_height)
    chart_x, chart_y = calculate_chart_coordinate(
        n_rows, n_cols, box_plot_height, hist_plot_height
    )

    chart_dict = {
        "barmode": "overlay",
        "height": figure_height,
        "paper_bgcolor": "white",
        "plot_bgcolor": "white",
    }
    for idx, param in enumerate(parameters):
        row, col = divmod(idx, n_cols)
        chart_dict[f"xaxis{idx+1}"] = {
            "anchor": f"y{2*idx+1}",
            "domain": [chart_x[0][col], chart_x[1][col]],
            # 'type': "log" if idx == 0 else "linear",
            "title": {"text": param},
        }
        # Histogram
        chart_dict[f"yaxis{2*idx+1}"] = {
            "anchor": f"x{idx+1}",
            "domain": [chart_y[0][n_rows - row - 1], chart_y[1][n_rows - row - 1]],
        }

        # Box plot
        chart_dict[f"yaxis{2*idx+2}"] = {
            "anchor": f"x{idx+1}",
            "domain": [chart_y[1][n_rows - row - 1], chart_y[2][n_rows - row - 1]],
            "type": "category",
            "showticklabels": False,
        }
    return chart_dict


class UpCaRsParameterDistribution(WebvizPluginABC):
    """### ParameterDistribution

This container shows parameter distributions for FMU ensembles.
Parameters are visualized per ensemble as a histogram, and as a boxplot showing
the parameter ranges for each ensemble.
Input can be given either as an aggregated csv files with parameter information
or as an ensemble name defined in `shared_settings`.

* `csvfile`: Aggregated csvfile with 'REAL', 'ENSEMBLE' and parameter columns
* `ensembles`: Which ensembles in `shared_settings` to visualize.
"""

    def __init__(
        self,
        app,
        csvfile: Path = None,
        ensembles: list = None,
        best_realization: list = None,
    ):

        super().__init__()
        self.csvfile = csvfile if csvfile else None
        self.color = PALETTE["tableau"]
        if csvfile and ensembles:
            raise ValueError(
                'Incorrect arguments. Either provide a "csvfile" or "ensembles".'
            )
        if csvfile:
            self.parameters = get_table_df(csvfile)
            self.parameters.rename(columns=lambda x: x[x.find(":") + 1 :], inplace=True)
            self.parameters = self.parameters.loc[
                :, ~self.parameters.columns.duplicated()
            ]
            self.ensembles = list(self.parameters["ENSEMBLE"].unique())
        elif ensembles:
            self.ensembles = tuple(
                (ens, app.webviz_settings["shared_settings"]["scratch_ensembles"][ens])
                for ens in ensembles
            )
            self.parameters = load_parameters(self.ensembles)
        else:
            raise ValueError(
                'Incorrect arguments. Either provide a "csvfile" or "ensembles".'
            )

        if best_realization is not None:
            if len(best_realization) != len(self.ensembles):
                raise ValueError(
                    "Number of ensembles differs with number of best realization"
                )
            self.best_realization = best_realization
        else:
            self.best_realization = None

        self.parameter_columns = [
            col
            for col in list(self.parameters.columns)
            if (
                col not in ["REAL", "ENSEMBLE"]
                and len(self.parameters[col].unique()) > 1
                and f"LOG10_{col}" not in self.parameters.columns
            )
        ]

    @property
    def tour_steps(self):
        return [
            {
                "id": self.uuid("layout-param"),
                "content": ("Dashboard displaying distribution of input parameters"),
            },
            {
                "id": self.uuid("graph"),
                "content": (
                    "Visualization of currently selected parameter as histogram "
                    "series and distribution range per ensemble."
                ),
            },
        ]

    def plot_histogram_dict(self):
        layout = create_hist_layout(self.parameter_columns)
        data = []
        ens_list = self.parameters["ENSEMBLE"].unique()
        color_list = self.color
        for idx_param, param in enumerate(self.parameter_columns):
            parameters_df = self.parameters[param]
            for idx_ens, ens in enumerate(ens_list):
                if self.best_realization:
                    data.append(
                        create_best_trace(
                            idx_param + 1,
                            ens,
                            parameters_df[
                                (self.parameters["ENSEMBLE"] == ens)
                                & (
                                    self.parameters["REAL"]
                                    == self.best_realization[idx_ens]
                                )
                            ],
                            idx_param == 0 and idx_ens == 0,
                        )
                    )
                data.extend(
                    create_hist_trace(
                        idx_param + 1,
                        parameters_df[self.parameters["ENSEMBLE"] == ens].values,
                        color_list[idx_ens],
                        ens,
                        idx_param == 0,
                    )
                )
        fig = go.Figure(data=data, layout=layout)
        return fig

    @property
    def layout(self):
        return html.Div(
            id=self.uuid("layout-param"),
            children=[
                wcc.Graph(id=self.uuid("graph"), figure=self.plot_histogram_dict()),
            ],
        )

    def add_webvizstore(self):
        return [
            (get_table_df, [{"csv_table": self.csvfile}])
            if self.csvfile
            else (load_parameters, [{"ensemble_paths": self.ensembles}])
        ]


# @CACHE.memoize(timeout=CACHE.TIMEOUT)
# def load_ensemble_set(ensemble_paths: tuple):
#     return EnsembleSet(
#         "EnsembleSet",
#         [ScratchEnsemble(ens_name, ens_path) for ens_name, ens_path in ensemble_paths],
#     )


# @CACHE.memoize(timeout=CACHE.TIMEOUT)
# @webvizstore
# def read_csv(csv_file) -> pd.DataFrame:
#     return pd.read_csv(csv_file, index_col=None)


# @CACHE.memoize(timeout=CACHE.TIMEOUT)
# @webvizstore
# def load_parameters(ensemble_paths: tuple) -> pd.DataFrame:
#     return load_ensemble_set(ensemble_paths).parameters
