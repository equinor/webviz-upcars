from webviz_config.webviz_store import webvizstore
from webviz_config.common_cache import CACHE
import pandas as pd

try:
    from fmu.ensemble import EnsembleSet, ScratchEnsemble
    from ecl.summary import EclSum
except ImportError:
    pass


@CACHE.memoize(timeout=CACHE.TIMEOUT)
@webvizstore
def get_table_df(csv_table: tuple) -> pd.DataFrame:
    if csv_table is None:
        return pd.DataFrame()
    return pd.read_csv(csv_table, index_col=None)


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
    return ensset.get_smry(column_keys=column_keys)


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
        smry.insert(0, "REAL", 0)
        smry.insert(0, "ENSEMBLE", case_name)

        smrylist.append(smry)
    if smrylist:
        data_frame = pd.concat(smrylist, sort=False)
        data_frame = data_frame.loc[:, ~data_frame.columns.duplicated()]
        return data_frame
    return pd.DataFrame()


@CACHE.memoize(timeout=CACHE.TIMEOUT)
@webvizstore
def load_parameters(ensemble_paths: tuple) -> pd.DataFrame:
    return load_ensemble_set(ensemble_paths).parameters
