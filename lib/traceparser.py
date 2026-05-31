import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Set, Tuple

import numpy as np
import pandas as pd

# Regular expressions for extracting metadata and channel information
FILE_GENERATION_DATE_REGEX = r"# File generation date and time: (.*)"
EXPERIMENT_DESCRIPTION_REGEX = r"# Experiment description: (.*)"
CLOCK_CYCLE_DURATION_REGEX = r"# Clock cycle duration: 1 clock cycle of (\d+) (\w+)"
CHANNEL_INFO_REGEX = (
    r"# Channel \d+ \[group_id=(\d+); event_id=(\d+); channel_name=(\w+)\]"
)

# CSV column headers
TIMESTAMPS_STRING = "timestamps"


@dataclass
class TraceMetadata:
    experiment_file_name: str
    experiment_data_time: str
    experiment_description: str
    clock_cycle_duration: int
    clock_cycle_unit: str


class TraceParser:
    """Process QNodeOS data stored in a specifically-formatted CSV file.

    Attributes:
        None
    """

    def __init__(self, csv_trace_file: Path) -> None:
        """TraceParser class initialization.

        Args:
            csv_trace_file: Path
                CSV file containing trace.

        Raises:
            FileNotFoundError: when `csv_trace_file` does not exist.
        """

        # Check if file exists
        if not os.path.isfile(csv_trace_file):
            raise FileNotFoundError(
                "`csv_trace_file` {} does not match any"
                " existing CSV file [aborting]".format(csv_trace_file)
            )

        # Initialize metadata dictionary containing file description
        self._metadata: TraceMetadata = None
        # Initiate trace file name
        self._trace_file_name: Path = csv_trace_file
        # Initialize dictionary of {(group_ID, event_ID): channel_name}
        self._channels_list: Dict[Tuple[int, int], str] = dict()
        # Initialize group_ID set
        self._group_ids: Set[int] = set()
        # Initialize dictionary of structure {channel_name: array[int, int, ...])
        self._trace_by_channel: Dict[str, np.ndarray] = dict()
        # Initialize variable for storing read CSV file's metadata
        self._metadata_lines: List[str] = list()

        # Open CSV file, transform to list, and close file
        with open(csv_trace_file, "r") as trace_file:
            # Read CSV file's comment lines only
            line = trace_file.readline().strip()
            while line.startswith("#"):
                self._metadata_lines.append(line)
                line = trace_file.readline().strip()

        # Read file as a pandas data structure
        self._df: pd.DataFrame = pd.read_csv(csv_trace_file, comment="#")

    def _extract_metadata(self) -> None:
        """Private method to extract metadata from a CSV file."""

        # Check if metadata has already been generated
        if self._metadata:
            return

        # Extract experiment date and time (by removing description text)
        match = re.search(FILE_GENERATION_DATE_REGEX, self._metadata_lines[0])
        assert match is not None
        experiment_date_time = match.group(1)

        # Extract experiment description
        match = re.search(EXPERIMENT_DESCRIPTION_REGEX, self._metadata_lines[1])
        assert match is not None
        experiment_description = match.group(1)

        # Extract clock cycle duration and unit
        match = re.search(CLOCK_CYCLE_DURATION_REGEX, self._metadata_lines[2])

        # Put duration always on 1, with unit "tick".
        # In practice, a "tick" for a Host trace will always be 1 us.
        # A "tick" for a QNodeOS trace will probably be 10 us,
        # but it depends on the ADwin setting!!! (see comment below).
        # It is up to the user of TraceParser to scale time accordingly.
        clock_cycle_duration = 1
        clock_cycle_unit_short = "tick"
        # if match is None:
        #     clock_cycle_duration = 1
        #     clock_cycle_unit_short = "us"
        # else:
        #     # clock_cycle_duration = int(match.group(1))
        #     # clock_cycle_unit = Unit(match.group(2))
        #     # clock_cycle_unit_short = f"{clock_cycle_unit:~}".replace("µ", "u")

        #     # The trace file metadata always say 10 us but this is hardcoded in the
        #     # QNodeOS source code! The actual cycle duration depends on the value of
        #     # SPI_CYCLES_PER_TRANSACTION in qdevice_spi.inc on the ADwin.
        #     clock_cycle_duration = 20
        #     clock_cycle_unit_short = "us"

        # Create dictionary with metadata
        self._metadata = TraceMetadata(
            str(self._trace_file_name),
            experiment_date_time,
            experiment_description,
            clock_cycle_duration,
            clock_cycle_unit_short,
        )

    def _extract_channels_info(self) -> None:
        """Private method to create a dictionary with QNodeOS channel information and a
        set of all *unique* group_IDs present in CSV file containing trace.


        Channel list is a dictionary of the following structure:
            {(group_id, event_id): channel_name}
        """

        # Check if channel information has already been generated
        if self._channels_list:
            return

        # Generate _channels_list and _group_ids
        for line in self._metadata_lines:
            match = re.search(CHANNEL_INFO_REGEX, line)
            if match is not None:
                group_id = int(match.group(1))
                event_id = int(match.group(2))
                channel_name = match.group(3)
                self._channels_list[group_id, event_id] = channel_name
                self._group_ids.add(group_id)

    def _extract_trace_by_channel(self) -> None:
        """Private method to extract dictionary of QNodeOS channel keys with their
        corresponding timestamps.
        """

        # Check if trace has already been generated
        if self._trace_by_channel:
            return

        self._extract_channels_info()
        self._extract_metadata()

        # Get array of all timestamps (multiplied by clock cycle duration)
        # print(f"Multiplying trace timestamps by {self._metadata.clock_cycle_duration}")
        timestamps = self._df[TIMESTAMPS_STRING].to_numpy() * int(
            self._metadata.clock_cycle_duration
        )

        # Get arrays of events, one per group_id
        events = dict()
        for group_id in self._group_ids:
            events[group_id] = self._df["group_id_" + str(group_id)].to_numpy()

        # Compile dictionary of trace channels with the corresponding timestamps
        for (group_id, event_id), channel_name in self._channels_list.items():
            channel_mask = 1 << event_id
            channel_timestamps = timestamps[events[group_id] & channel_mask != 0]
            self._trace_by_channel[channel_name] = channel_timestamps

    def get_trace_by_channel(self) -> Dict[str, np.ndarray]:
        """Extract dictionary of QNodeOS channel keys with their corresponding
        timestamps.

        Output dictionary has the following stucture
            {channel_name: np.ndarray[int, int, ...]}

        Args:
            None

        Returns:
            Dict[str, np.ndarray]

        Raises:
            None
        """

        self._extract_trace_by_channel()
        return self._trace_by_channel

    def get_metadata(self) -> TraceMetadata:
        """Extract metadata information from a CSV file containing trace.

        Args:
            None

        Returns:
            TraceMetadata

        Raises:
            None
        """

        self._extract_metadata()
        return self._metadata


def get_ordered_trace(trace: Dict[str, np.ndarray]) -> List[Tuple[int, str]]:
    time_chan_tuples: List[Tuple[int, str]] = []
    for channel, timestamps in trace.items():
        for time in timestamps:
            time_chan_tuples.append((time, channel))

    time_chan_tuples.sort()
    tuples2 = [(t, c) for (t, c) in time_chan_tuples]

    for i in range(len(time_chan_tuples) - 1):
        (time1, chan1) = time_chan_tuples[i]
        (time2, chan2) = time_chan_tuples[i + 1]
        swap = False
        if time1 == time2:
            if chan2 == "QNETWORK_ENT_PULL" and chan1 == "QDEVICE_PRODUCE_ENT_CMD":
                swap = True
            elif (
                chan1 == "PROCESSOR_CONSUME_OUTCOME"
                and chan2 == "QDEVICE_PRODUCE_OUTCOME"
            ):
                swap = True
            elif (
                chan1 == "PROCESSOR_FINISH_NET_PROCESS"
                and chan2 == "PROCESSOR_START_NET_PROCESS"
            ):
                swap = True

        if swap:
            tuples2[i] = (time2, chan2)
            tuples2[i + 1] = (time1, chan1)

    assert len(tuples2) == len(time_chan_tuples)

    return tuples2


def dump_ordered_trace(trace: Dict[str, np.ndarray], dir_name: str, trace_name: str):
    time_chan_tuples = get_ordered_trace(trace)

    log_file = os.path.join(
        os.getenv("PROCESSED_DATA_PATH"), f"{dir_name}_{trace_name}.log"
    )
    with open(log_file, "w") as file:
        for time, chan in time_chan_tuples:
            file.write(f"{time}: {chan}\n")
