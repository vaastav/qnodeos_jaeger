import errno
import json
import os
import pickle
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
import xarray as xr

from lib import adwin
from lib.trace import OrderedTrace, OrderedTraceNew
from lib.traceparser import TraceParser, get_ordered_trace


def _find_file(dir: str, name: str) -> Path:
    p = Path(dir)
    files = list(p.glob("**/" + name))
    if len(files) > 1:
        raise ValueError(f"Multiple files named '{name}' exist in directory '{dir}'")
    if len(files) == 0:
        raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), name)
    return files[0]


def _find_files(dir: str, name: str) -> List[Path]:
    print(f"looking in dir {dir} for files with name {name}")
    p = Path(dir)
    files = list(p.glob("**/" + name))
    sorted_files = sorted(files, key=lambda x: str(x))
    if len(sorted_files) == 0:
        raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), name)
    return sorted_files


def _get_calibration_data(calibration_file, outcomes):
    calibration_dataset = xr.load_dataset(calibration_file)
    F0 = calibration_dataset.F0.sel(x0=outcomes.meas_duration * 1e-6)
    F1 = calibration_dataset.F1.sel(x0=outcomes.meas_duration * 1e-6)
    return (
        np.array([[F0, 1 - F1], [1 - F0, F1]])
        * calibration_dataset.reps_per_sweep_point
    )


def import_qnodeos_outcomes(data_dir: str) -> Dict[str, Any]:
    outcomes_file = _find_file(data_dir, "*outcomes.json")

    with open(outcomes_file, "r") as handle:
        outcomes = json.load(handle)

    return outcomes


def import_results_pickle(
    data_dir: str, allow_multiple: bool = False
) -> Dict[str, Any]:
    if allow_multiple:
        results_files = _find_files(data_dir, "*results.pickle")
        results_list = []
        for file in results_files:
            with open(file, "rb") as handle:
                results_list.append(pickle.load(handle))
        return results_list
    else:
        results_file = _find_file(data_dir, "*results.pickle")
        with open(results_file, "rb") as handle:
            results = pickle.load(handle)
        return results


def import_pickle_by_filename(data_dir: str, filename_end: str) -> Dict[str, Any]:
    results_file = _find_file(data_dir, f"*{filename_end}")
    with open(results_file, "rb") as handle:
        results = pickle.load(handle)
    return results


def import_qnodeos_trace(data_dir: str) -> Dict[str, np.ndarray]:
    trace_file = _find_file(data_dir, "*qnodeos_trace.csv")

    return TraceParser(trace_file).get_trace_by_channel()


def import_trace_by_filename(
    data_dir: str, filename_end: str, allow_multipe: bool = False
) -> Dict[str, np.ndarray]:
    if allow_multipe:
        trace_files = _find_files(data_dir, f"*{filename_end}")
        print(f"files found: {trace_files}")
        return [TraceParser(f).get_trace_by_channel() for f in trace_files]
    else:
        trace_file = _find_file(data_dir, f"*{filename_end}")
        return TraceParser(trace_file).get_trace_by_channel()


def import_ordered_trace_by_filename(
    data_dir: str, filename_end: str, resolution_us: int
) -> OrderedTrace:
    trace_file = _find_file(data_dir, f"*{filename_end}")
    parsed = TraceParser(trace_file).get_trace_by_channel()
    trace_in_ticks = get_ordered_trace(parsed)
    trace_in_ms = [
        ((timestamp * resolution_us) / 1000, event)
        for (timestamp, event) in trace_in_ticks
    ]
    return OrderedTrace.from_raw(trace_in_ms)


def import_ordered_traces(
    data_dir: str, filename_end: str, resolution_us: int
) -> List[OrderedTrace]:
    trace_files = _find_files(data_dir, f"*{filename_end}")
    print(f"files found: {trace_files}")
    traces: List[OrderedTrace] = []
    for file in trace_files:
        parsed = TraceParser(file).get_trace_by_channel()
        trace_in_ticks = get_ordered_trace(parsed)
        trace_in_ms = [
            ((timestamp * resolution_us) / 1000, event)
            for (timestamp, event) in trace_in_ticks
        ]
        traces.append(OrderedTrace.from_raw(trace_in_ms))
    return traces


def import_ordered_traces_new(
    data_dir: str, filename_end: str, resolution_us: int
) -> List[OrderedTraceNew]:
    trace_files = _find_files(data_dir, f"*{filename_end}")
    print(f"files found: {trace_files}")
    traces: List[OrderedTraceNew] = []
    for file in trace_files:
        parsed = TraceParser(file).get_trace_by_channel()
        trace_in_ticks = get_ordered_trace(parsed)
        trace_in_ms = [
            ((timestamp * resolution_us) / 1000, event)
            for (timestamp, event) in trace_in_ticks
        ]
        traces.append(OrderedTraceNew.from_raw(trace_in_ms))
    return traces


def import_qnodeos_timestamps(data_dir: str) -> Dict[str, pd.DataFrame]:
    request_timestamps_file = _find_file(data_dir, "*request_timestamps.csv")
    ent_cmd_timestamps_file = _find_file(data_dir, "*ent_cmd_timestamps.csv")

    request_timestamps = pd.read_csv(request_timestamps_file)
    ent_cmd_timestamps = pd.read_csv(ent_cmd_timestamps_file)
    assert isinstance(request_timestamps, pd.DataFrame)
    assert isinstance(ent_cmd_timestamps, pd.DataFrame)

    return {"request": request_timestamps, "ent_cmd": ent_cmd_timestamps}


def import_adwin_sent_traces(
    data_dir: str, filename_end: str, resolution_us: int
) -> List[xr.DataArray]:
    trace_files = _find_files(data_dir, f"*{filename_end}")
    print(f"files found: {trace_files}")
    traces: List[xr.DataArray] = []
    for file in trace_files:
        dataset = xr.load_dataset(file)
        mask = np.isin(dataset.y0.values, adwin.commands_to_qnodeos_numbers)
        dataset_processed = xr.Dataset()
        dataset_processed["time"] = dataset.x0.values[mask]
        dataset_processed["cmd"] = (
            ["time"],
            np.searchsorted(adwin.commands_to_qnodeos_numbers, dataset.y0.values[mask]),
        )
        data = dataset_processed.cmd
        data["time"] = (data["time"] * resolution_us) / 1000

        traces.append(data)
    return traces


def import_adwin_rcvd_traces(
    data_dir: str, filename_end: str, resolution_us: int
) -> List[xr.DataArray]:
    trace_files = _find_files(data_dir, f"*{filename_end}")
    print(f"files found: {trace_files}")
    traces: List[xr.DataArray] = []
    for file in trace_files:
        dataset = xr.load_dataset(file)
        mask = np.isin(dataset.y0.values, adwin.commands_from_qnodeos_numbers)
        dataset_processed = xr.Dataset()
        dataset_processed["time"] = dataset.x0.values[mask]
        dataset_processed["cmd"] = (
            ["time"],
            np.searchsorted(
                adwin.commands_from_qnodeos_numbers, dataset.y0.values[mask]
            ),
        )
        data = dataset_processed.cmd
        data["time"] = (data["time"] * resolution_us) / 1000

        traces.append(data)
    return traces


def import_phys_layer_data(data_dir: str) -> Dict[str, Any]:
    try:
        outcomes_file = _find_file(data_dir, "single_click_data.hdf5")
        outcomes = xr.load_dataset(outcomes_file)
    except ValueError:
        outcomes_files = _find_files(data_dir, "single_click_data.hdf5")
        outcomes = [xr.load_dataset(f) for f in outcomes_files]
    except FileNotFoundError:
        outcomes = None

    try:
        cr_check_file = _find_file(data_dir, "cr_check.hdf5")
        ionization = xr.load_dataset(cr_check_file)
    except ValueError:
        cr_check_files = _find_files(data_dir, "cr_check.hdf5")
        print(f"found files {cr_check_files}")
        ionization = [xr.load_dataset(f) for f in cr_check_files]
    except FileNotFoundError:
        ionization = None

    try:
        calibration_file = _find_file(data_dir, "dataset_processed.hdf5")
        if isinstance(outcomes, list):
            calibration = [_get_calibration_data(calibration_file, o) for o in outcomes]
        else:
            calibration = _get_calibration_data(calibration_file, outcomes)
    except ValueError:
        calibration_files = _find_files(data_dir, "dataset_processed.hdf5")
        calibration = [
            _get_calibration_data(f, o) for f, o in zip(calibration_files, outcomes)
        ]
    except FileNotFoundError:
        calibration = None

    try:
        commands_trace_file = _find_file(data_dir, "link_layer_rcvd.hdf5")
        commands_trace = xr.load_dataset(commands_trace_file)
    except ValueError:
        commands_trace_files = _find_files(data_dir, "link_layer_rcvd.hdf5")
        commands_trace = [xr.load_dataset(f) for f in commands_trace_files]
    except FileNotFoundError:
        commands_trace = None

    try:
        outcomes_trace_file = _find_file(data_dir, "link_layer_sent.hdf5")
        outcomes_trace = xr.load_dataset(outcomes_trace_file)
    except ValueError:
        outcomes_trace_files = _find_files(data_dir, "link_layer_sent.hdf5")
        outcomes_trace = [xr.load_dataset(f) for f in outcomes_trace_files]
    except:
        outcomes_trace = None

    return {
        "outcomes": outcomes,
        "ionization": ionization,
        "calibration": calibration,
        "commands_trace": commands_trace,
        "outcomes_trace": outcomes_trace,
    }
