import numpy as np

commands_from_qnodeos = {
    -1: "NOCO",
    0: "NO_OPERATION",
    1: "INIT_QUBIT",
    2: "SQG_X0",
    3: "TQG",
    4: "MEASURE",
    5: "ENTANGLE",
    6: "ENM",
    7: "PMG",
    8: "MOV",
    9: "SWP",
    10: "ESW",
    # SQGates
    4 << 4 | 2: "X45",
    8 << 4 | 2: "X90",
    16 << 4 | 2: "X180",
    24 << 4 | 2: "-X90",
    32 << 4 | 2: "Y0",
    40 << 4 | 2: "Y90",
    48 << 4 | 2: "Y180",
    56 << 4 | 2: "-Y90",
    16 << 4 | 2 | 64 << 4: "Z180",
}

commands_to_qnodeos = {
    0: "UNKNOWN_FAILURE",
    1: "HARDWARE_FAILURE",
    2: "UNSUPPORTED_REQUEST",
    3: "INVALID_QUBIT",
    4: "ENTANGLEMENT_FAILURE",
    5: "ENTANGLEMENT_SYNC_FAILURE",
    8: "SUCCESS_ZERO",
    9: "SUCCESS_PHI_MINUS",
    10: "SUCCESS_PSI_PLUS",
    11: "SUCCESS_PSI_MINUS",
    12: "SUCCESS_ONE",
}

commands_from_qnodeos_numbers = np.array(list(commands_from_qnodeos.keys()))
commands_from_qnodeos_names = list(commands_from_qnodeos.values())

commands_to_qnodeos_numbers = np.array(list(commands_to_qnodeos.keys()))
commands_to_qnodeos_names = list(commands_to_qnodeos.values())


def name_to_xarray_number_to_qnodeos(name: str) -> int:
    # Note: the number returned is the value that you'll find in the actual
    # xarray.DataArray object for the event given by "name".
    # This number is different from the keys in the global dictionaries above!
    # (instead, it is the index in these dictionaries treated as lists)
    lst = list(commands_to_qnodeos.values())
    return lst.index(name)


def name_to_xarray_number_from_qnodeos(name: str) -> int:
    # Note: the number returned is the value that you'll find in the actual
    # xarray.DataArray object for the event given by "name".
    # This number is different from the keys in the global dictionaries above!
    # (instead, it is the index in these dictionaries treated as lists)
    lst = list(commands_from_qnodeos.values())
    return lst.index(name)
