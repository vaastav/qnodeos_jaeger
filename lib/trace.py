from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
import xarray as xr

_qnodeos_traces = [
    "SUBROUTINE_SEND_ATTEMPT",
    "SUBROUTINE_SENT",
    "RESULT_RCVD",
    "CLAS_MSG_SENT",
    "CLAS_MSG_RCVD",
    "QDEVICE_PRODUCE_INI_CMD",
    "QDEVICE_PRODUCE_SQG_CMD",
    "QDEVICE_PRODUCE_TQG_CMD",
    "QDEVICE_PRODUCE_ENT_CMD",
    "QDEVICE_PRODUCE_ENM_CMD",
    "QDEVICE_PRODUCE_MSR_CMD",
    "QDEVICE_PRODUCE_CMD",
    "QDEVICE_PRODUCE_OUTCOME",
    "QDEVICE_CONSUME_CMD",
    "QNETWORK_SWAP_PULL",
    "QNETWORK_SWAP_RET",
    "QNETWORK_ENT_PULL",
    "QNETWORK_ENM_PULL",
    "QNP_BQ_DELIVERED",
    "QNPD_COMPLETE_SENT",
    "QNPD_EXPIRE_SENT",
    "QNPD_TRACK_SENT",
    "QNPD_FORWARD_SENT",
    "QNPD_PASS_SENT",
    "QNPD_COMPLETE_RECEIVED",
    "QNPD_EXPIRE_RECEIVED",
    "QNPD_TRACK_RECEIVED",
    "QNPD_FORWARD_RECEIVED",
    "QNPD_PASS_RECEIVED",
    "QNPD_TRACK_CACHED",
    "EGP_ENT_OK",
    "EGP_ENM_OK",
    "EGP_NEI_OK",
    "EGPD_CREATE",
    "PROCESSOR_START_USER_PROCESS",
    "PROCESSOR_WAIT_USER_PROCESS",
    "PROCESSOR_FINISH_USER_PROCESS",
    "PROCESSOR_START_NET_PROCESS",
    "PROCESSOR_FINISH_NET_PROCESS",
    "PROCESSOR_CONSUME_OUTCOME",
    "SCHEDULER_HIGH_PRIO_PROCESS_WAITS_ON_LOWER_PRIO",
    "SCHEDULER_ARRIVE_USER_PROCESS",
    "SCHEDULER_ARRIVE_NET_PROCESS",
    "SCHEDULER_SCHEDULE_USER_PROCESS",
    "SCHEDULER_SCHEDULE_NET_PROCESS",
]

for i in range(10):
    _qnodeos_traces.append(f"PROCMGR_SUBROUTINE_ADDED_P{i}")
    _qnodeos_traces.append(f"PROCMGR_SUBROUTINE_DONE_P{i}")
    _qnodeos_traces.append(f"PROCMGR_SUBROUTINE_ABORTED_P{i}")

_qnodeos_trace_name_to_number = {name: idx for idx, name in enumerate(_qnodeos_traces)}
_qnodeos_trace_number_to_name = {idx: name for idx, name in enumerate(_qnodeos_traces)}


def trace_name_to_number(name: str) -> int:
    return _qnodeos_trace_name_to_number[name]


def trace_number_to_name(number: int) -> str:
    return _qnodeos_trace_number_to_name[number]


@dataclass
class TraceEntry:
    time: float
    event: str


@dataclass
class OrderedTraceNew:
    data: xr.DataArray

    def filter_events(self, events: List[str]) -> OrderedTrace:
        vals = [trace_name_to_number(e) for e in events]
        mask = self.data.isin(vals)
        filtered = self.data.where(mask, drop=True)
        return OrderedTraceNew(filtered)

    def get_times(self, events: List[str]) -> OrderedTraceNew:
        vals = [trace_name_to_number(e) for e in events]
        mask = np.isin(self.data, vals)
        times = self.data.coords["time"].values[mask]
        return np.array(times)

    def get_times_by_num(self, events: List[int]) -> OrderedTraceNew:
        mask = np.isin(self.data, events)
        times = self.data.coords["time"].values[mask]
        return np.array(times)

    def add_offset(self, offset: float) -> OrderedTraceNew:
        return OrderedTraceNew(self.data + offset)

    @classmethod
    def from_raw(cls, raw: List[Tuple[float, str]]) -> OrderedTraceNew:
        coords, values_str = zip(*raw)
        coords = list(coords)
        values = [trace_name_to_number(name) for name in values_str]
        arr = xr.DataArray(data=values, coords={"time": coords}, dims="time")
        return OrderedTraceNew(arr)

    @staticmethod
    def merge(traces: List[OrderedTraceNew]) -> OrderedTraceNew:
        merged = xr.concat([t.data for t in traces], dim="time")
        sorted = merged.sortby("time")
        return OrderedTraceNew(sorted)


@dataclass
class OrderedTrace:
    entries: List[TraceEntry]

    def filter_events(self, events: List[str]) -> OrderedTrace:
        entries = [e for e in self.entries if e.event in events]
        return OrderedTrace(entries)

    def get_timestamps(self, event: str) -> List[float]:
        filtered = self.filter_events([event])
        return [e.time for e in filtered.entries]

    def add_offset(self, offset: float) -> OrderedTrace:
        entries = [TraceEntry(e.time + offset, e.event) for e in self.entries]
        return OrderedTrace(entries)

    def to_time_np_array(self) -> np.ndarray:
        return np.array([e.time for e in self.entries])

    @classmethod
    def from_raw(cls, raw: List[Tuple[float, str]]) -> OrderedTrace:
        entries = [TraceEntry(t, e) for (t, e) in raw]
        return OrderedTrace(entries)

    @staticmethod
    def merge(traces: List[OrderedTrace]) -> OrderedTrace:
        entries: List[TraceEntry] = []
        for trace in traces:
            entries.extend(trace.entries)
        entries.sort(key=lambda entry: entry.time)
        return OrderedTrace(entries)

    def __len__(self) -> int:
        return len(self.entries)


class TraceComparison:
    @staticmethod
    def get_offsets(
        trace1: OrderedTrace, event1: str, trace2: OrderedTrace, event2: str
    ) -> List[float]:
        entries1 = trace1.filter_events([event1]).entries
        entries2 = trace2.filter_events([event2]).entries
        assert len(entries1) == len(entries2)
        return [e2.time - e1.time for (e1, e2) in zip(entries1, entries2)]

    @staticmethod
    def get_offsets_no_filter(
        trace1: OrderedTrace, trace2: OrderedTrace
    ) -> List[float]:
        assert len(trace1) == len(trace2)
        return [e2.time - e1.time for (e1, e2) in zip(trace1.entries, trace2.entries)]


@dataclass
class AdwinTrace:
    events: xr.DataArray

    def get_ent_gen_slices(self) -> List[xr.DataArray]:
        pass
