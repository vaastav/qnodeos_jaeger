from dataclasses import dataclass
from typing import Dict, Union

import numpy as np
from qutip.metrics import fidelity
from qutip.operators import identity, sigmax, sigmay, sigmaz
from qutip.states import Qobj, ket2dm

from lib.types import BellState, SingleQubitState, TraceChannel


@dataclass
class EgpTimestamps:
    """EgpTimestamps class.

    Attributes:
        egp_create: np.ndarray
            Timestamps of submission of EGP create requests.
        egp_ok: np.ndarray
            Timestamps of completion of EGP create requests.
        ent_cmd: np.ndarray
            Timestamps of transmission of ENT/ENM command.
        ent_outcome: np.ndarray
            Timestamps of reception of ENT/ENM command outcome.
    """

    egp_create: np.ndarray
    egp_ok: np.ndarray
    ent_cmd: np.ndarray
    ent_outcome: np.ndarray


@dataclass
class SingleAppProcessingMetrics:
    """SingleAppProcessingMetrics class.

    Attributes:
    """


@dataclass
class SingleQubitMetrics:
    """SingleQubitMetrics class.

    Attributes:
        target_state: SingleQubitState
            Target qubit state, to be used for fidelity calculation.
        distr_0_meas_x: float = 0.0
            Distribution of '0' outcomes when qubit was measured in the X basis.
        distr_1_meas_x: float = 0.0
            Distribution of '1' outcomes when qubit was measured in the X basis.
        distr_0_meas_y: float = 0.0
            Distribution of '0' outcomes when qubit was measured in the Y basis.
        distr_1_meas_y: float = 0.0
            Distribution of '1' outcomes when qubit was measured in the Y basis.
        distr_0_meas_z: float
            Distribution of '0' outcomes when qubit was measured in the Z basis.
        distr_1_meas_z: float
            Distribution of '1' outcomes when qubit was measured in the Z basis.
        state_fidelity: float
            Estimated fidelity of the qubit state.
    """

    target_state: SingleQubitState

    distr_0_meas_x: float
    distr_1_meas_x: float

    distr_0_meas_y: float
    distr_1_meas_y: float

    distr_0_meas_z: float
    distr_1_meas_z: float

    state_fidelity: float


@dataclass
class EntQubitsMetrics:
    """EntQubitsMetrics class.

    Attributes:
        target_bell_state: BellState
            Target Bell state.
        distr_00_meas_xx: float
            Distribution of '00 'outcomes when entangled qubits were measured in the X basis.
        distr_01_meas_xx: float
            Distribution of '01 'outcomes when entangled qubits were measured in the X basis.
        distr_10_meas_xx: float
            Distribution of '10 'outcomes when entangled qubits were measured in the X basis.
        distr_11_meas_xx: float
            Distribution of '11 'outcomes when entangled qubits were measured in the X basis.
        distr_00_meas_yy: float
            Distribution of '00 'outcomes when entangled qubits were measured in the Y basis.
        distr_01_meas_yy: float
            Distribution of '01 'outcomes when entangled qubits were measured in the Y basis.
        distr_10_meas_yy: float
            Distribution of '10 'outcomes when entangled qubits were measured in the Y basis.
        distr_11_meas_yy: float
            Distribution of '11 'outcomes when entangled qubits were measured in the Y basis.
        distr_00_meas_zz: float
            Distribution of '00 'outcomes when entangled qubits were measured in the Z basis.
        distr_01_meas_zz: float
            Distribution of '01 'outcomes when entangled qubits were measured in the Z basis.
        distr_10_meas_zz: float
            Distribution of '10 'outcomes when entangled qubits were measured in the Z basis.
        distr_11_meas_zz: float
            Distribution of '11 'outcomes when entangled qubits were measured in the Z basis.
        qber_x: float
            Qubit error rate when entangled qubits were measured in the X basis.
        qber_y: float
            Qubit error rate when entangled qubits were measured in the Y basis.
        qber_z: float
            Qubit error rate when entangled qubits were measured in the Z basis.
        bell_state_fidelity: float
            Estimated fidelity of the entangled state.
    """

    target_bell_state: BellState

    distr_00_meas_xx: float
    distr_01_meas_xx: float
    distr_10_meas_xx: float
    distr_11_meas_xx: float

    distr_00_meas_yy: float
    distr_01_meas_yy: float
    distr_10_meas_yy: float
    distr_11_meas_yy: float

    distr_00_meas_zz: float
    distr_01_meas_zz: float
    distr_10_meas_zz: float
    distr_11_meas_zz: float

    qber_x: float
    qber_y: float
    qber_z: float

    bell_state_fidelity: float


def _str_to_single_qubit_state(state: str) -> SingleQubitState:
    """Convert string to SingleQubitState enum.

    Args:
        state: String to convert.

    Returns:
        SingleQubitState enum constant.

    Raises:
        ValueError: when state string is invalid.
    """

    state_map = {
        "+X": SingleQubitState.PLUS_X,
        "+Y": SingleQubitState.PLUS_Y,
        "+Z": SingleQubitState.PLUS_Z,
        "-X": SingleQubitState.MINUS_X,
        "-Y": SingleQubitState.MINUS_Y,
        "-Z": SingleQubitState.MINUS_Z,
    }

    assert state in state_map.keys()
    return state_map[state]


def _compute_single_qubit_fidelity(
    outcomes_x: np.ndarray,
    outcomes_y: np.ndarray,
    outcomes_z: np.ndarray,
    target: Qobj,
) -> float:
    """Calculate metric state_fidelity (squared fidelity).

    Args:
        outcomes_x: Array of outcomes from X-basis measurements.
        outcomes_y: Array of outcomes from Y-basis measurements.
        outcomes_z: Array of outcomes from Z-basis measurements.
        target: Target state, to compare measured state to.

    Returns:
        state fidelity (squared fidelity).
    """

    def exp_value(outcomes) -> float:
        if outcomes.size == 0:
            return 0
        # Convert measurement outcomes to eigenvalues of basis vectors (a '0' measurement
        # corresponds to the +1 eigenvalue, a '1' measurement corresponds to the -1 eigenvalue), and
        # calculate expected value as mean of eigenvalues.
        eigenvalues = np.where(outcomes == 0, 1, -1)  # 0 -> +1, 1 -> -1
        return np.mean(eigenvalues)

    # exp_x, exp_y and exp_z are the X, Y and Z components of the Bloch vector for which we're
    # calculating the fidelity.
    exp_x = exp_value(outcomes_x)
    exp_y = exp_value(outcomes_y)
    exp_z = exp_value(outcomes_z)
    # Density matrix as linear combination of Pauli matrices.
    # From Nielsen and Chuang, "Quantum Computation and Quantum Information", formula also at
    # https://en.wikipedia.org/wiki/Density_matrix#Pure_and_mixed_states
    density_matrix = 0.5 * (
        identity(2) + exp_x * sigmax() + exp_y * sigmay() + exp_z * sigmaz()
    )
    # Return squared fidelity.
    fid = fidelity(density_matrix.unit(), target)
    return fid**2


def compute_egp_timestamps(trace: Dict[str, np.ndarray]) -> EgpTimestamps:
    """Construct and populate EgpTimestamps object from QNodeOS trace.

    Args:
        trace: Dict[str, np.ndarray]
            QNodeOS trace, as obtained from TraceParser.get_trace_by_channel().

    Returns:
        timestamps: EgpTimestamps
            Populated EgpTimestamps object.
    """
    egp_create = trace[TraceChannel.EGPD_CREATE]
    egp_ok = np.sort(
        np.concatenate((trace[TraceChannel.EGP_ENT_OK], trace[TraceChannel.EGP_ENM_OK]))
    )

    qdevice_produce_ent_cmd = np.sort(
        np.concatenate(
            (
                trace[TraceChannel.QDEVICE_PRODUCE_ENT_CMD],
                trace[TraceChannel.QDEVICE_PRODUCE_ENM_CMD],
            )
        )
    )
    qdevice_consume_cmd = trace[TraceChannel.QDEVICE_CONSUME_CMD]
    qdevice_produce_outcome = trace[TraceChannel.QDEVICE_PRODUCE_OUTCOME]
    consume_ent_cmd_indices = np.searchsorted(
        qdevice_consume_cmd, qdevice_produce_ent_cmd
    )
    produce_ent_outcome_indices = np.searchsorted(
        qdevice_consume_cmd, qdevice_produce_ent_cmd
    )

    ent_cmd = qdevice_consume_cmd[consume_ent_cmd_indices]
    ent_outcome = qdevice_produce_outcome[produce_ent_outcome_indices]

    return EgpTimestamps(egp_create, egp_ok, ent_cmd, ent_outcome)


def compute_single_qubit_metrics(
    outcomes_x: np.ndarray,
    outcomes_y: np.ndarray,
    outcomes_z: np.ndarray,
    target_state: Union[SingleQubitState, Qobj, str],
) -> SingleQubitMetrics:
    """Construct and populate SingleQubitMetrics object from application results.

    Args:
        outcomes_x: np.ndarray
            Measurement outcomes when qubit was measured in the X basis.
        outcomes_y: np.ndarray
            Measurement outcomes when qubit was measured in the Y basis.
        outcomes_z: np.ndarray
            Measurement outcomes when qubit was measured in the Z basis.
        target_state: SingleQubitState | Qobj | str
            Target state.

    Returns:
        metrics: SingleQubitMetrics
            Populated SingleQubitMetrics object.
    """
    distr_1_meas_x = np.mean(outcomes_x)
    distr_0_meas_x = 1 - distr_1_meas_x

    distr_1_meas_y = np.mean(outcomes_y)
    distr_0_meas_y = 1 - distr_1_meas_y

    distr_1_meas_z = np.mean(outcomes_z)
    distr_0_meas_z = 1 - distr_1_meas_z

    if isinstance(target_state, SingleQubitState):
        target_state_qobj = ket2dm(target_state.value)
    elif isinstance(target_state, Qobj):
        target_state_qobj = ket2dm(target_state)
    elif isinstance(target_state, str):
        target_state_qobj = ket2dm(_str_to_single_qubit_state(target_state).value)

    state_fidelity = _compute_single_qubit_fidelity(
        outcomes_x, outcomes_y, outcomes_z, target_state_qobj
    )

    return SingleQubitMetrics(
        target_state=target_state,
        distr_0_meas_x=distr_0_meas_x,
        distr_1_meas_x=distr_1_meas_x,
        distr_0_meas_y=distr_0_meas_y,
        distr_1_meas_y=distr_1_meas_y,
        distr_0_meas_z=distr_0_meas_z,
        distr_1_meas_z=distr_1_meas_z,
        state_fidelity=state_fidelity,
    )


def compute_fidelity_with_uncertainty(rho, sigma_rho, psi):
    """
    Compute the fidelity and its uncertainty between a density matrix rho and a pure state |psi>.

    Parameters:
    rho (numpy.ndarray): 2x2 density matrix.
    sigma_rho (numpy.ndarray): 2x2 matrix of uncertainties corresponding to the elements of rho.
    psi (numpy.ndarray): 2x1 pure state vector.

    Returns:
    F (float): Fidelity.
    sigma_F (float): Uncertainty in the fidelity.
    F_squared (float): Squared fidelity.
    sigma_F_squared (float): Uncertainty in the squared fidelity.
    """

    # Compute the fidelity
    psi_dagger = np.conjugate(psi).T
    F = np.dot(psi_dagger, np.dot(rho, psi))[0, 0].real

    # Compute the partial derivatives of F with respect to each rho_ij
    partial_derivatives = np.zeros((2, 2))
    for i in range(2):
        for j in range(2):
            partial_derivatives[i, j] = (
                np.dot(psi_dagger, np.dot(np.eye(2) * (i == j), psi))
            )[0, 0].real

    # Compute the uncertainty in F
    sigma_F_squared = np.sum((partial_derivatives * sigma_rho) ** 2)
    sigma_F = np.sqrt(sigma_F_squared)

    # Compute the squared fidelity and its uncertainty
    F_squared = F**2
    sigma_F_squared_squared = 2 * F * sigma_F
    sigma_F_squared = sigma_F_squared_squared

    return F, sigma_F, F_squared, sigma_F_squared
