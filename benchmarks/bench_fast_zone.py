"""Benchmark: unit_jit fast_zone in Gillespie and SDE vs plain pint.

Systems tested:
  Plain*   -- plain pint throughout (current bcrnnoise behaviour)
  Fast*    -- @unit_jit on reaction_rates + noise, fast_zone loop

For Fast* to work, Quantity params must live as instance attributes (not in a
NamedTuple) so that unit_jit's snapshot converts them to SI floats at fast-zone
entry.
"""

import math
import time
from collections.abc import Sequence

import numpy as np
from pint import Quantity
from unit_jit import fast_zone, unit_jit, ureg

from bcrnnoise import BCRN, Timeseries

# --- shared parameters -------------------------------------------------

INIT_MRNA = 0.0 / ureg.femtoliter
VOLUME = 1.0 * ureg.femtoliter
TIME_HORIZON = 60.0 * ureg.minute
DT = 0.1 * ureg.minute
ALPHA = 1.0 / ureg.minute / ureg.femtoliter
DELTA = 0.1 / ureg.minute
SIGMA = 1.0 / ureg.femtoliter / ureg.minute**0.5

N_REPS = 30
N_WARMUP = 3


# --- 1-species plain pint ----------------------------------------------


class PlainSystem(BCRN):
    def __init__(self) -> None:
        super().__init__(init_state=[INIT_MRNA], time_horizon=TIME_HORIZON, volume=VOLUME, dt=DT)

    @property
    def stoichiometry(self) -> np.ndarray:
        return np.array([[1], [-1]])

    def reaction_rates(self, state: Sequence[Quantity]) -> list[Quantity]:
        return [ALPHA, DELTA * state[0]]

    def noise(self, rng: np.random.Generator, t: Quantity, y: Sequence[Quantity]) -> list[Quantity]:
        dW = np.sqrt(self.dt) * rng.normal()
        return [SIGMA * dW]


# --- 1-species @unit_jit -----------------------------------------------
# Params stored as instance attrs so snapshot() converts them to SI floats.
# self.dt is set by BCRN.__init__ → also snapshotted automatically.


@unit_jit
class FastSystem(BCRN):
    alpha: Quantity
    delta: Quantity
    sigma: Quantity

    def __init__(self) -> None:
        super().__init__(init_state=[INIT_MRNA], time_horizon=TIME_HORIZON, volume=VOLUME, dt=DT)
        self.alpha = ALPHA
        self.delta = DELTA
        self.sigma = SIGMA

    @property
    def stoichiometry(self) -> np.ndarray:
        return np.array([[1], [-1]])

    def reaction_rates(self, state: Sequence[Quantity]) -> list[Quantity]:
        return [self.alpha, self.delta * state[0]]

    def noise(self, rng: np.random.Generator, t: Quantity, y: Sequence[Quantity]) -> list[Quantity]:
        # self.dt and self.sigma are SI floats in the fast zone: no pint overhead
        dW = np.sqrt(self.dt) * rng.normal()
        return [self.sigma * dW]


# --- float-land Gillespie using fast_zone ------------------------------


def gillespie_fast(system: FastSystem, rng: np.random.Generator) -> Timeseries:
    stoich = np.array(system.stoichiometry)
    count_arr = np.array([int(v * system.volume) for v in system.init_state], dtype=float)

    volume_si = system.volume.to_base_units().magnitude
    max_time_si = system.time_horizon.to_base_units().magnitude
    time_unit = system.time_horizon.units
    conc_unit = system.init_state[0].units

    time_si = 0.0
    times_si = [0.0]
    counts = [count_arr.copy()]

    with fast_zone(system) as (fast_system,):
        while time_si < max_time_si:
            conc_si = [c / volume_si for c in count_arr]
            rates_si = fast_system.reaction_rates(conc_si)
            rates_vol_si = [r * volume_si for r in rates_si]
            total_rate_si = sum(rates_vol_si)

            if total_rate_si == 0.0:
                break

            r1, r2 = rng.random(2)
            time_si += math.log(1.0 / r1) / total_rate_si

            rate_arr = np.fromiter(rates_vol_si, dtype=float, count=len(rates_vol_si))
            reaction_idx = int(np.searchsorted(np.cumsum(rate_arr), r2 * total_rate_si))
            count_arr += stoich[reaction_idx]
            times_si.append(time_si)
            counts.append(count_arr.copy())

    times = [t * time_unit for t in times_si]
    states = [[c / volume_si * conc_unit for c in cnt] for cnt in counts]
    return Timeseries(times=times, states=states)


# --- float-land SDE (Euler-Maruyama) using fast_zone ------------------


def sde_fast(system: FastSystem, rng: np.random.Generator) -> Timeseries:
    dt_si = system.dt.to_base_units().magnitude
    time_horizon_si = system.time_horizon.to_base_units().magnitude
    n_steps = int(np.floor(time_horizon_si / dt_si))
    n_species = len(system.init_state)
    stoich_t = system._stoichiometry_t  # plain numpy array

    # initial state: SI float per species
    state_si = [s.to_base_units().magnitude for s in system.init_state]
    times_si = [k * dt_si for k in range(n_steps + 1)]
    all_states_si: list[list[float]] = [state_si[:]]

    with fast_zone(system) as (fast_system,):
        for k in range(n_steps):
            t_si = times_si[k]

            rates_si = fast_system.reaction_rates(state_si)       # raw SI floats
            drift_si = (stoich_t @ np.array(rates_si)).tolist()   # pure numpy
            noise_si = fast_system.noise(rng, t_si, state_si)     # raw SI floats

            state_si = [
                max(0.0, state_si[i] + drift_si[i] * dt_si + noise_si[i])
                for i in range(n_species)
            ]
            all_states_si.append(state_si[:])

    conc_unit = system.init_state[0].units
    time_unit = system.time_horizon.units
    times = [t * time_unit for t in times_si]
    states = [[c * conc_unit for c in s] for s in all_states_si]
    return Timeseries(times=times, states=states)


# --- benchmark helpers -------------------------------------------------


def bench(name: str, fn, n_warmup: int, n_reps: int) -> float:
    for _ in range(n_warmup):
        fn()
    t0 = time.perf_counter()
    for _ in range(n_reps):
        fn()
    elapsed = (time.perf_counter() - t0) / n_reps * 1000
    print(f"  {name:45s}  {elapsed:.2f} ms/call")
    return elapsed


def main() -> None:
    plain = PlainSystem()
    fast = FastSystem()

    print(f"\nGillespie ({N_REPS} reps after {N_WARMUP} warmup):")
    t_plain = bench("plain pint", lambda: plain.simulate_markov_chain(seed=42), N_WARMUP, N_REPS)
    t_fast = bench(
        "@unit_jit + fast_zone",
        lambda: gillespie_fast(fast, np.random.default_rng(42)),
        N_WARMUP,
        N_REPS,
    )
    print(f"  speedup: {t_plain / t_fast:.1f}x")

    print(f"\nSDE Gaussian noise, dt={DT} ({N_REPS} reps after {N_WARMUP} warmup):")
    t_plain_sde = bench(
        "plain pint",
        lambda: plain.simulate_sde(seed=42, noise_fun=plain.noise),
        N_WARMUP,
        N_REPS,
    )
    t_fast_sde = bench(
        "@unit_jit + fast_zone",
        lambda: sde_fast(fast, np.random.default_rng(42)),
        N_WARMUP,
        N_REPS,
    )
    print(f"  speedup: {t_plain_sde / t_fast_sde:.1f}x")


if __name__ == "__main__":
    main()
