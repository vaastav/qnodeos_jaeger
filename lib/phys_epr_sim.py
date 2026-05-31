import numpy as np
import qutip as qt

I = qt.identity(2)
Z = qt.sigmaz()
II = qt.identity([2, 2])
ZI = qt.tensor(Z, I)
IZ = qt.tensor(I, Z)
ZZ = qt.tensor(Z, Z)


# DiamondOS (physical layer) function to simulate the EPR fidelity
# given physical parameters.


def sce_state(alpha_1, alpha_2, pars):
    assert 0 < alpha_1 < 1, f"alpha_1 = {alpha_1} not in range"
    assert 0 < alpha_2 < 1, f"alpha_2 = {alpha_2} not in range"

    Pdc = pars["dc_rate"] * pars["det_win"]  # Dark count prob in window

    Puu = (
        alpha_1
        * alpha_2
        * (
            (1 - Pdc) ** 2
            * (1 - pars["pdet_psb"])
            * (
                pars["pdet_1"] * (1 - pars["pdet_2"])
                + pars["pdet_2"] * (1 - pars["pdet_1"])
            )
            + 2
            * (1 - Pdc)
            * Pdc
            * (1 - pars["pdet_psb"]) ** 2
            * (1 - pars["pdet_1"])
            * (1 - pars["pdet_2"])
        )
    )
    Pud = (
        alpha_1
        * (1 - alpha_2)
        * ((1 - Pdc) ** 2 * pars["pdet_1"] + 2 * Pdc * (1 - Pdc) * (1 - pars["pdet_1"]))
    )
    Pdu = (
        (1 - alpha_1)
        * alpha_2
        * ((1 - Pdc) ** 2 * pars["pdet_2"] + 2 * Pdc * (1 - Pdc) * (1 - pars["pdet_2"]))
    )
    Pdd = 2 * (1 - alpha_1) * (1 - alpha_2) * Pdc * (1 - Pdc)

    Pe = 0.5 * (
        1 - np.exp(-0.5 * (pars["phase_u"] / 180 * np.pi) ** 2)
    )  # Phase uncertainty

    Ptot = Puu + Pud + Pdu + Pdd
    rate = Ptot / pars["lde_length"]
    fidelity = (1 - pars["depol"]) * (
        Pud
        + Pdu
        + 2 * (1 - pars["p2e"]) ** 2 * (1 - 2 * Pe) * np.sqrt(pars["vis"] * Pud * Pdu)
    ) / (2 * Ptot) + pars["depol"] / 4

    rho = qt.Qobj(
        np.array(
            [
                [Pdd, 0, 0, 0],
                [0, Pud, np.sqrt(pars["vis"] * Pud * Pdu), 0],
                [0, np.sqrt(pars["vis"] * Pud * Pdu), Pdu, 0],
                [0, 0, 0, Puu],
            ]
        )
        / Ptot,
        dims=[[2, 2], [2, 2]],
    )

    # Simulate double excitation
    rho = (
        (1 - pars["p2e"] / 2) ** 2 * rho
        + (1 - pars["pdet_psb"])
        * (1 - pars["p2e"] / 2)
        * (pars["p2e"] / 2)
        * ZI
        * rho
        * ZI
        + (1 - pars["pdet_psb"])
        * (1 - pars["p2e"] / 2)
        * (pars["p2e"] / 2)
        * IZ
        * rho
        * IZ
        + (1 - pars["pdet_psb"]) ** 2 * (pars["p2e"] / 2) ** 2 * ZZ * rho * ZZ
    ).unit()

    # Phase uncertainty
    rho = (1 - Pe) * rho + Pe * IZ * rho * IZ

    # Depolarizing noise
    rho = (1 - pars["depol"]) * rho + pars["depol"] * II / 4

    return rho, rate, fidelity
