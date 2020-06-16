import pathlib
import glob
from pkg_resources import get_distribution, DistributionNotFound

import webviz_config

try:
    __version__ = get_distribution(__name__).version
except DistributionNotFound:
    # package is not installed
    pass


@webviz_config.SHARED_SETTINGS_SUBSCRIPTIONS.subscribe("scratch_ensembles")
def subscribe_scratch_ensemble(scratch_ensembles, config_folder, portable):
    if scratch_ensembles is not None:
        for ensemble_name, ensemble_path in scratch_ensembles.items():
            if not pathlib.Path(ensemble_path).is_absolute():
                scratch_ensembles[ensemble_name] = str(config_folder / ensemble_path)
            if not portable and not glob.glob(scratch_ensembles[ensemble_name]):
                raise ValueError(
                    f"Ensemble {ensemble_name} is said to be located at {ensemble_path},"
                    " but that wildcard path does not give any matches."
                )

    return scratch_ensembles


@webviz_config.SHARED_SETTINGS_SUBSCRIPTIONS.subscribe("realizations")
def subscribe_realizations(realizations, config_folder, portable):
    if realizations is not None:
        for realization_name, realization_path in realizations.items():
            if not pathlib.Path(realization_path).is_absolute():
                realizations[realization_name] = str(config_folder / realization_path)

        for realization_name, realization_path in realizations.items():
            if not portable and not glob.glob(realizations[realization_name]):
                raise ValueError(
                    f"Realization {realization_name} is said to be located at {realization_path},"
                    " but that wildcard path does not give any matches."
                )
    return realizations


@webviz_config.SHARED_SETTINGS_SUBSCRIPTIONS.subscribe("krpc_csv_tables")
def subscribe_krpc_csv_tables(krpc_csv_tables, config_folder, portable):
    if krpc_csv_tables is not None:
        for name, path in krpc_csv_tables.items():
            print(path + "\t" + name)
            if not pathlib.Path(path).is_absolute():
                krpc_csv_tables[name] = str(config_folder / path)
            if not portable and not glob.glob(krpc_csv_tables[name]):
                raise ValueError(
                    f"KrPc CSV {name} is said to be located at {path},"
                    " but that wildcard path does not give any matches."
                )
    return krpc_csv_tables
