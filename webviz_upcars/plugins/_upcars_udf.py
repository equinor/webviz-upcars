from webviz_config.webviz_store import webvizstore
from webviz_config.common_cache import CACHE
import pandas as pd
import plotly.graph_objs as go


try:
    from fmu.ensemble import EnsembleSet, ScratchEnsemble
    from ecl.summary import EclSum
except ImportError:
    pass


class palette:
    tableau = [
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
    tableau_light = [
        "rgb(174, 199, 232)",
        "rgb(255, 187, 120)",
        "rgb(152, 223, 138)",
        "rgb(255, 152, 150)",
        "rgb(197, 176, 213)",
        "rgb(196, 156, 148)",
        "rgb(247, 182, 210)",
        "rgb(199, 199, 199)",
        "rgb(219, 219, 141)",
        "rgb(158, 218, 229)",
    ]
    tabelau_medium = [
        "rgb(114, 158, 206)",
        "rgb(255, 158, 74)",
        "rgb(103, 191, 92)",
        "rgb(237, 102, 93)",
        "rgb(173, 139, 201)",
        "rgb(168, 120, 110)",
        "rgb(237, 151, 202)",
        "rgb(162, 162, 162)",
        "rgb(205, 204, 93)",
        "rgb(109, 204, 218)",
    ]
    tableau_flip = [
        "rgb(31, 119, 180)",
        "rgb(174, 199, 232)",
        "rgb(255, 127, 14)",
        "rgb(255, 187, 120)",
        "rgb(44, 160, 44)",
        "rgb(152, 223, 138)",
        "rgb(214, 39, 40)",
        "rgb(255, 152, 150)",
        "rgb(148, 103, 189)",
        "rgb(197, 176, 213)",
        "rgb(140, 86, 75)",
        "rgb(196, 156, 148)",
        "rgb(227, 119, 194)",
        "rgb(247, 182, 210)",
        "rgb(127, 127, 127)",
        "rgb(199, 199, 199)",
        "rgb(188, 189, 34)",
        "rgb(219, 219, 141)",
        "rgb(23, 190, 207)",
        "rgb(158, 218, 229)",
    ]
    tableau_20 = [
        "rgb(255, 187, 120)",
        "rgb(255, 127, 14 )",
        "rgb(174, 199, 232)",
        "rgb(44, 160, 44  )",
        "rgb(31, 119, 180 )",
        "rgb(255, 152, 150)",
        "rgb(214, 39, 40  )",
        "rgb(197, 176, 213)",
        "rgb(152, 223, 138)",
        "rgb(148, 103, 189)",
        "rgb(247, 182, 210)",
        "rgb(227, 119, 194)",
        "rgb(196, 156, 148)",
        "rgb(140, 86, 75  )",
        "rgb(127, 127, 127)",
        "rgb(219, 219, 141)",
        "rgb(199, 199, 199)",
        "rgb(188, 189, 34 )",
        "rgb(158, 218, 229)",
        "rgb(23, 190, 207 )",
    ]
    colorblind = [
        "rgb(0, 107, 164  )",
        "rgb(255, 128, 14 )",
        "rgb(171, 171, 171)",
        "rgb(89, 89, 89   )",
        "rgb(95, 158, 209 )",
        "rgb(200, 82, 0   )",
        "rgb(137, 137, 137)",
        "rgb(162, 200, 236)",
        "rgb(255, 188, 121)",
        "rgb(207, 207, 207)",
    ]


@CACHE.memoize(timeout=CACHE.TIMEOUT)
@webvizstore
def get_table_df(csv_table: tuple) -> pd.DataFrame:
    if csv_table is None:
        return pd.DataFrame()
    return pd.read_csv(csv_table)


@CACHE.memoize(timeout=CACHE.TIMEOUT)
@webvizstore
def get_multiple_table_df(tables: tuple) -> pd.DataFrame:
    table_list = []
    for name, path in tables:
        table = pd.read_csv(path)
        table.insert(0, "REAL", 0)
        table.insert(0, "ENSEMBLE", name)
        table_list.append(table)
    if table_list:
        return pd.concat(table_list, sort=False)
    return pd.DataFrame()


@CACHE.memoize(timeout=CACHE.TIMEOUT)
def load_ensemble_set(ensemble_paths: tuple):
    return EnsembleSet(
        "EnsembleSet",
        [ScratchEnsemble(ens_name, ens_path) for ens_name, ens_path in ensemble_paths],
    )


@CACHE.memoize(timeout=CACHE.TIMEOUT)
@webvizstore
def get_ensemble_df(ensemble_path: tuple, column_keys: tuple) -> pd.DataFrame:
    ensset = load_ensemble_set(ensemble_path)
    df = ensset.get_smry(column_keys=column_keys)
    df.rename(columns={"FUPVINJ": "Pore Volume Injected"}, inplace=True)
    return df


@CACHE.memoize(timeout=CACHE.TIMEOUT)
@webvizstore
def get_summary_df(case_paths: tuple, column_keys: tuple) -> pd.DataFrame:
    smrylist = []
    for case_name, case_path in case_paths:
        smry = EclSum(
            case_path.replace(".DATA", ".SMSPEC"),
            include_restart=False,
            lazy_load=False,
        ).pandas_frame(None, column_keys)
        smry.rename(columns={"FUPVINJ": "Pore Volume Injected"}, inplace=True)
        smry.insert(0, "REAL", 0)
        smry.insert(0, "ENSEMBLE", case_name)

        smrylist.append(smry)
    if smrylist:
        # pd.concat(smrylist, sort=False).to_csv("/mnt/c/home_office/dump.csv", index=False)
        df = pd.concat(smrylist, sort=False)
        df = df.loc[:, ~df.columns.duplicated()]
        return df
    return pd.DataFrame()


@CACHE.memoize(timeout=CACHE.TIMEOUT)
@webvizstore
def load_parameters(ensemble_paths: tuple) -> pd.DataFrame:
    return load_ensemble_set(ensemble_paths).parameters


def set_grid_layout(columns):
    return {
        "display": "grid",
        "alignContent": "space-around",
        "justifyContent": "space-between",
        "gridTemplateColumns": f"{columns}",
        "gridColumnGap": "10px",
    }


class bcolors:
    HEADER = "\033[95m"
    OKBLUE = "\033[94m"
    OKGREEN = "\033[92m"
    WARNING = "\033[93m"
    FAIL = "\033[91m"
    ENDC = "\033[0m"
    BOLD = "\033[1m"
    UNDERLINE = "\033[4m"


def eclsum_keyword2title(keyword):
    """
    Decipher Eclipse summary keyword to proper title
    :param keyword: Eclipse summary keyword
    :return: Proper title
    Modify "special" dict for keywords that doesn't follow "standard convention"
    """
    special = {
        "FWCT": "Field watercut",
        "FUDP": "BHP INJ1 - PROD1",
        "FPR": "Field Pressure",
        "FOSAT": "Oil saturation",
        "FGSAT": "Gas saturation",
        "FUPVINJ": "PV Injected",
        "FGOR": "Field GOR",
        "TIME": "Elapsed time",
    }
    _title = special.get(keyword, None)
    if _title is None:
        dict_list = [
            {"F": "Field ", "W": "Well ", "B": "Block ", "C": "Connection "},
            {"O": "Oil ", "W": "Water ", "G": "Gas "},
            {"P": "Production ", "I": "Injection "},
            {"T": "Total", "R": "Rate"},
        ]
        title_list = [None] * 4
        for i in range(4):
            title_list[i] = dict_list[i].get(keyword[i], keyword[i])
        if keyword.find(":") > 0:
            return "".join(title_list) + " ({})".format(
                keyword[keyword.find(":") + 1 :]
            )
        return "".join(title_list)
    return _title


def create_trace(x, y, curve_name, legend_name, opacity, showlegend, color, meta):
    return go.Scattergl(
        x=x,
        y=y,
        legendgroup=legend_name,
        hovertext=curve_name,
        hoverinfo="y+x+text",
        name=legend_name,
        mode="lines",
        opacity=opacity,
        showlegend=showlegend,
        marker_color=color,
        line_color=color,
        meta=meta,
        line_width=2.0,
    )


def create_trace_dict(
    x, y, curve_name, legend_name, opacity, showlegend, color, meta, xaxis, yaxis
):
    return {
        "x": x,
        "y": y,
        "legendgroup": legend_name,
        "hovertext": curve_name,
        "hoverinfo": "y+x+text",
        "name": legend_name,
        "type": "scattergl",
        "xaxis": xaxis,
        "yaxis": yaxis,
        "mode": "lines",
        "opacity": opacity,
        "showlegend": showlegend,
        "line": {"color": color, "width": 2.0},
        "meta": meta,
    }


def krpc_table_key(table_type):
    dict_table = {
        "SWOF": {"saturation": "Sw", "kr1": "krw", "kr2": "krow", "pc": "pcow"},
        "SGOF": {"saturation": "Sg", "kr1": "krg", "kr2": "krog", "pc": "pcog"},
    }
    keys = dict_table.get(table_type, dict_table["SWOF"])
    return keys["saturation"], keys["kr1"], keys["kr2"], keys["pc"]
