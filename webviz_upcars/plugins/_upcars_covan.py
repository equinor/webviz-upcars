from pathlib import Path
import dash_html_components as html
import webviz_core_components as wcc
from webviz_config import WebvizPluginABC
import plotly.graph_objs as go
from plotly.subplots import make_subplots
from .._util.fmu_input import get_table_df


class UpCaRsCovan(WebvizPluginABC):
    """### Plot for Linearized Co-variance analysis

    This container shows relative permeability and its confidence interval.
    Input is an aggregated csv file generated from ERT Covan Workflow
    """

    def __init__(
        # self, app, csv_relperm: Path = None, csv_reference: Path = None,
        self,
        csv_relperm: Path = None,
        csv_reference: Path = None,
    ):
        super().__init__()
        self.csv_relperm = csv_relperm
        self.csv_reference = csv_reference
        if not (csv_relperm and csv_reference):
            raise ValueError(
                "Incorrect argument. Please provide path to csv files from covan analysis."
            )
        self.curves = get_table_df(csv_relperm).round(4)
        self.reference = get_table_df(csv_reference)
        self.gas_oil_mode = "krg" in self.curves.columns
        self.obs_params = []
        for col in self.reference.columns:
            if ":BASE" in col:
                obs = col[:-5]
                if not obs in self.obs_params:
                    self.obs_params.append(obs)

    @property
    def tour_steps(self):
        return [
            {
                "id": self.uuid("layout-covan"),
                "content": (
                    "Dashboard displaying result from covariance analysis "
                    "of relative permeability curve"
                ),
            },
            {
                "id": self.uuid("reference"),
                "content": (
                    "Visualization of objective functions being used as "
                    "input paramteter for covariance analysis. The standard "
                    "deviation is taken from average error between base case "
                    "and reference (fine-scale) case"
                ),
            },
            {
                "id": self.uuid("relperm"),
                "content": (
                    "Visualization of relative permeability confidence interval. "
                    "A wide range (in vertical direction) shows that there is "
                    "not much confidence in the value."
                ),
            },
        ]

    @property
    def layout(self):
        return html.Div(
            id=self.uuid("layout-covan"),
            children=[
                wcc.Graph(id=self.uuid("reference"), figure=self.plot_reference()),
                wcc.Graph(id=self.uuid("relperm"), figure=self.plot_relperm()),
            ],
        )

    def plot_reference(self):
        nparams = len(self.obs_params)
        nrow = int((nparams + 1) / 2)
        ncol = 2
        fig = make_subplots(rows=nrow, cols=ncol)
        ref_line = {"color": "blue", "width": 2}
        base_line = {"color": "black", "width": 2}
        area_line = {"color": "red", "dash": "dash", "width": 1}
        fill_color = "rgba(255, 170 , 170, 0.3)"

        row = 1
        col = 1
        for i, obs in enumerate(self.obs_params):
            fig.add_trace(
                go.Scatter(
                    x=self.reference["Elapsed Days"],
                    y=self.reference[f"{obs}:BASE"],
                    line=base_line,
                    name="Base Case",
                    legendgroup="Base",
                    showlegend=i == 0,
                ),
                row=row,
                col=col,
            )
            fig.add_trace(
                go.Scatter(
                    x=self.reference["Elapsed Days"],
                    y=self.reference[f"{obs}:REF"],
                    line=ref_line,
                    name="Reference Case (Fine Scale)",
                    legendgroup="Reference",
                    showlegend=i == 0,
                ),
                row=row,
                col=col,
            )
            fig.add_trace(
                go.Scatter(
                    x=self.reference["Elapsed Days"],
                    y=self.reference[f"{obs}:BASE"] - self.reference[f"{obs}:SD"],
                    line=area_line,
                    name="Confidence Interval",
                    legendgroup="Interval",
                    hoverinfo="skip",
                    showlegend=i == 0,
                ),
                row=row,
                col=col,
            )
            fig.add_trace(
                go.Scatter(
                    x=self.reference["Elapsed Days"],
                    y=self.reference[f"{obs}:BASE"] + self.reference[f"{obs}:SD"],
                    line=area_line,
                    fill="tonexty",
                    fillcolor=fill_color,
                    name="Confidence Interval",
                    legendgroup="Interval",
                    hoverinfo="skip",
                    showlegend=False,
                ),
                row=row,
                col=col,
            )
            fig.update_yaxes(row=row, col=col, title=obs)
            col += 1
            if col > 2:
                col = 1
                row += 1
        fig.update_xaxes(title="Elapsed days")
        fig.update_layout(height=600, hovermode="x unified")

        return fig

    def plot_relperm(self):
        if self.gas_oil_mode:
            plot_titles = ["Gas Relative Permeability", "Oil Relative Permeability"]
            sat_col = "Sg"
            kr1_col = "krg"
            dkr1_col = "dkrg"
        else:
            plot_titles = ["Water Relative Permeability", "Oil Relative Permeability"]
            sat_col = "Sw"
            kr1_col = "krw"
            dkr1_col = "dkrw"
        fig = make_subplots(rows=1, cols=2, subplot_titles=plot_titles)
        curve_line = {"color": "black", "width": 2}
        area_line = {"color": "red", "dash": "dash", "width": 2}
        fill_color = "rgba(255, 170, 170, 0.3)"
        # Water Relative Permeability
        fig.add_trace(
            go.Scatter(
                x=self.curves[sat_col],
                y=self.curves[kr1_col] + self.curves[dkr1_col],
                hoverinfo="skip",
                line=area_line,
            ),
            row=1,
            col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=self.curves[sat_col],
                y=self.curves[kr1_col] - self.curves[dkr1_col],
                hoverinfo="skip",
                line=area_line,
                fill="tonexty",
                fillcolor=fill_color,
            ),
            row=1,
            col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=self.curves[sat_col],
                y=self.curves[kr1_col],
                name=kr1_col,
                line=curve_line,
                hovertext=["dkr = {}".format(dkr) for dkr in self.curves[dkr1_col]],
            ),
            row=1,
            col=1,
        )

        # Oil Relative Permeability
        fig.add_trace(
            go.Scatter(
                x=self.curves[sat_col],
                y=self.curves["kro"] + self.curves["dkro"],
                hoverinfo="skip",
                line=area_line,
            ),
            row=1,
            col=2,
        )
        fig.add_trace(
            go.Scatter(
                x=self.curves[sat_col],
                y=self.curves["kro"] - self.curves["dkro"],
                hoverinfo="skip",
                fill="tonexty",
                fillcolor=fill_color,
                line=area_line,
            ),
            row=1,
            col=2,
        )
        fig.add_trace(
            go.Scatter(
                x=self.curves[sat_col],
                y=self.curves["kro"],
                name="kro",
                line=curve_line,
                hovertext=["dkr = {}".format(dkr) for dkr in self.curves["dkro"]],
            ),
            row=1,
            col=2,
        )

        fig.update_xaxes(range=[0.0, 1.0], dtick=0.2, title="Saturation, fraction")
        fig.update_yaxes(
            range=[0.0, 1.0], dtick=0.2, title="Relative Permeability, fraction"
        )

        fig.update_layout(
            showlegend=False, height=800,
        )
        return fig

    def add_webvizstore(self):
        return [
            (
                get_table_df,
                [{"csv_table": self.csv_relperm}, {"csv_table": self.csv_reference},],
            )
        ]
