# -*- coding: utf-8 -*-
"""
Created on Tue Dec 23 17:39:36 2025

@author: bobva
"""

from typing import Iterable, Iterator, List, Tuple, Dict
import random
import numpy as np
import cantera as ct
import collections
from scipy.interpolate import interp1d


# -------------------------------
# Bolus event functions
# -------------------------------
# A bolus event is (time, species_name, amount)
BolusEvent = Tuple[float, str, float]
def single_event_generator(events: Iterable[BolusEvent]) -> Iterator[BolusEvent]:
    """
    Yield bolus events in ascending time order.

    events: iterable of (time, species, amount)
    """
    events_sorted = sorted(events, key=lambda e: e[0])
    for ev in events_sorted:
        yield ev
        
def generate_random_bolus_events(
    n_bolus: int,
    t_start: float,
    t_end: float,
    amount_range: Tuple[float, float],
    species_names: list,
    rng: np.random.Generator,
) -> List[BolusEvent]:
    """
    Generate a list of random bolus events.

    Parameters
    ----------
    n_bolus : int
        Number of bolus events in this schedule.
    t_start, t_end : float
        Time window for boluses.
    amount_range : (float, float)
        Min and max bolus amount.
    rng : np.random.Generator
        Random number generator (for reproducibility).

    """
    times = rng.uniform(t_start, t_end, size=n_bolus)
    times.sort()  # nice to have them ordered

    species_idx = rng.integers(0, len(species_names), size=n_bolus)
    amounts = rng.uniform(amount_range[0], amount_range[1], size=n_bolus)

    events: List[BolusEvent] = []
    for t, idx, a in zip(times, species_idx, amounts):
        events.append((float(t), species_names[int(idx)], float(a)))
    return events

# -------------------------------
# Simulator with bolus inputs
# -------------------------------
 
"""This is a very simple RK4 solver manually defined
   If you simulate anything more complex u need to use
   a better solver odeint, or any sundials suite"""

def simulate_chain_with_bolus(model,
    k: np.ndarray,
    y0: np.ndarray,
    t_start: float,
    t_end: float,
    dt: float,
    bolus_gen: Iterator[BolusEvent] = None,
    species_names: List[str] = None,
):
    """
    Simulate A -> B -> C -> D -> E with mass-action kinetics and bolus inputs.

    Dynamics:
        dy/dt = f(t, y; k)
    Bolus:
        y_s(t_bolus+) = y_s(t_bolus-) + amount

    Parameters
    ----------
    k : array-like, shape (4,)
        Rate constants [k1, k2, k3, k4].
    y0 : array-like, shape (5,)
        Initial concentrations [A0, B0, C0, D0, E0].
    t_start : float
        Start time.
    t_end : float
        End time.
    dt : float
        Maximum time step for integration (adaptive to hit bolus times exactly).
    bolus_gen : iterator of BolusEvent, optional
        Generator yielding (time, species_name, amount) in (non-strictly) ascending time.
    species_names : list of str, optional
        Names for the 5 species; default is ["A", "B", "C", "D", "E"].

    """
    k = np.asarray(k, dtype=float)
    y = np.asarray(y0, dtype=float)

    if species_names is None:
        species_names = []
    assert len(species_names) == len(y0), "species_names must have length 5"

    species_index: Dict[str, int] = {name: i for i, name in enumerate(species_names)}

    # Prepare bolus generator
    if bolus_gen is not None:
        try:
            next_bolus = next(bolus_gen)
        except StopIteration:
            next_bolus = None
    else:
        next_bolus = None

    t = float(t_start)
    t_end = float(t_end)

    times = [t]
    states = [y.copy()]

    def step_rhs(t_local: float, y_local: np.ndarray, h: float) -> np.ndarray:
        """Single RK4 step."""
        k1 = model(t_local, y_local, k)
        k2 = model(t_local + 0.5 * h, y_local + 0.5 * h * k1, k)
        k3 = model(t_local + 0.5 * h, y_local + 0.5 * h * k2, k)
        k4 = model(t_local + h, y_local + h * k3, k)
        return y_local + (h / 6.0) * (k1 + 2*k2 + 2*k3 + k4)

    # Main loop
    eps = 1e-12
    while t < t_end - eps:
        # Apply any bolus that is exactly at current time
        while next_bolus is not None and abs(next_bolus[0] - t) < eps:
            bolus_time, bolus_species, bolus_amount = next_bolus
            y[species_index[bolus_species]] += bolus_amount

            try:
                next_bolus = next(bolus_gen)
            except StopIteration:
                next_bolus = None

        # Determine step size, ensuring we stop at next bolus or at final time
        h = dt
        if next_bolus is not None and t + h > next_bolus[0]:
            h = next_bolus[0] - t
        if t + h > t_end:
            h = t_end - t

        # Integrate
        if h > 0:
            y = step_rhs(t, y, h)
            t = t + h
            times.append(t)
            states.append(y.copy())
        else:
            # In case h becomes zero due to numerical issues
            t = t_end

    # Apply any bolus exactly at t_end
    while next_bolus is not None and abs(next_bolus[0] - t_end) < eps:
        _, bolus_species, bolus_amount = next_bolus
        y[species_index[bolus_species]] += bolus_amount
        states[-1] = y.copy()
        try:
            next_bolus = next(bolus_gen)
        except StopIteration:
            next_bolus = None
    return np.array(times), np.vstack(states)


def simulate_ivp_with_bolus(
    model,
    k: np.ndarray,
    y0: np.ndarray,
    t_start: float,
    t_end: float,
    bolus_gen: Iterator[BolusEvent] = None,
    species_names: List[str] = None,
    method: str = "BDF",
    rtol: float = 1e-6,
    atol: float = 1e-8,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Simulate a model with bolus inputs using scipy.integrate.solve_ivp.

    Same bolus logic as simulate_chain_with_bolus — random events at random
    times on random control channels — but uses an adaptive stiff solver
    instead of fixed-step RK4. Recommended for the MOF synthesis model where
    fractional-power nucleation terms (SBU^3) create stiff transients.

    Boluses are handled piecewise: solve_ivp is called on each continuous
    segment between bolus events, the bolus is applied instantaneously at the
    segment boundary, then integration resumes. Output format is identical to
    simulate_chain_with_bolus so create_dataset.py interpolation is unchanged.
    """
    from scipy.integrate import solve_ivp

    k = np.asarray(k, dtype=float)
    y = np.asarray(y0, dtype=float).copy()

    if species_names is None:
        species_names = []
    species_index: Dict[str, int] = {name: i for i, name in enumerate(species_names)}

    # Materialise all bolus events so we can build segment breakpoints
    all_events: List[BolusEvent] = sorted(bolus_gen, key=lambda e: e[0]) if bolus_gen is not None else []

    # Group boluses by time
    boluses_at: Dict[float, List[Tuple[str, float]]] = {}
    for bt, bspec, bamt in all_events:
        if bt not in boluses_at:
            boluses_at[bt] = []
        boluses_at[bt].append((bspec, bamt))

    # Segment endpoints: bolus times strictly inside (t_start, t_end), then t_end
    inner_times = sorted({bt for bt in boluses_at if t_start < bt < t_end})
    segment_ends = inner_times + [t_end]

    def rhs(t: float, y_loc: np.ndarray) -> np.ndarray:
        return model(t, y_loc, k)

    times_out: List[float] = [t_start]
    states_out: List[np.ndarray] = [y.copy()]

    # Apply any boluses at t_start before first integration
    for bspec, bamt in boluses_at.get(t_start, []):
        if bspec in species_index:
            y[species_index[bspec]] += bamt

    t_cur = t_start
    for t_next in segment_ends:
        if t_next <= t_cur:
            continue

        sol = solve_ivp(rhs, [t_cur, t_next], y, method=method, rtol=rtol, atol=atol)

        if not sol.success:
            y[:] = np.nan  # caught by NaN check in create_dataset.py
            break

        times_out.extend(sol.t[1:].tolist())
        states_out.extend(sol.y[:, 1:].T.tolist())
        y = np.maximum(0.0, sol.y[:, -1])
        t_cur = t_next

        # Apply boluses at the boundary before the next segment
        if t_next < t_end:
            for bspec, bamt in boluses_at.get(t_next, []):
                if bspec in species_index:
                    y[species_index[bspec]] += bamt

    return np.array(times_out, dtype=float), np.vstack(states_out)

def simulate_cantera_with_bolus(
    gas_object: ct.Solution,
    y0: np.ndarray,
    t_start: float,
    t_end: float,
    bolus_gen,
    species_names: list,
    t_record: np.ndarray = None,
):
    """
    Simulate a stiff chemical mechanism using Cantera's ReactorNet.

    t_record : 1-D array of times at which to record the state (must be sorted,
               within [t_start, t_end]).  If None, records only segment
               endpoints (bolus times + t_end).  Pass the observation grid from
               create_dataset.py so interpolation is exact.
    """
    gas_object.concentrations = y0

    reactor = ct.IdealGasConstPressureReactor(gas_object)
    sim = ct.ReactorNet([reactor])
    sim.rtol = 1.0e-4
    sim.atol = 1.0e-12

    t_history = [t_start]
    y_history = [gas_object.concentrations.copy()]

    current_t = t_start

    events = list(bolus_gen) if bolus_gen else []
    events_by_time = collections.defaultdict(list)
    for t_ev, ch_ev, amt_ev in events:
        events_by_time[t_ev].append((ch_ev, amt_ev))
    sorted_event_times = sorted(events_by_time.keys())

    # All times we need to stop at: bolus times + t_end
    stop_times = [t for t in sorted_event_times if t_start < t <= t_end] + [t_end]

    # Dense record times within each segment (from t_record grid)
    if t_record is not None:
        t_record = np.asarray(t_record, dtype=float)

    def _advance_segment(t_from: float, t_to: float):
        """Advance sim from t_from to t_to, recording at every t_record point inside."""
        if t_record is not None:
            # Find observation times strictly inside this segment
            mask = (t_record > t_from) & (t_record < t_to)
            inner = t_record[mask]
        else:
            inner = np.array([])

        for t_obs_i in inner:
            sim.advance(float(t_obs_i))
            t_history.append(sim.time)
            y_history.append(gas_object.concentrations.copy())

        sim.advance(t_to)
        t_history.append(sim.time)
        y_history.append(gas_object.concentrations.copy())

    for t_stop in stop_times:
        if t_stop <= current_t:
            continue

        _advance_segment(current_t, t_stop)
        current_t = t_stop

        if t_stop < t_end and t_stop in events_by_time:
            T_cur = gas_object.T
            P_cur = gas_object.P
            concs = gas_object.concentrations.copy()
            for ch_ev, amt_ev in events_by_time[t_stop]:
                idx = species_names.index(ch_ev)
                concs[idx] = max(0.0, concs[idx] + amt_ev)
            gas_object.TPY = T_cur, P_cur, concs
            reactor.syncState()
            t_history.append(sim.time)
            y_history.append(gas_object.concentrations.copy())

    return np.array(t_history), np.array(y_history)


###############################################################################
###############################################################################
###############################################################################
from .benchmark_models import FullModel

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    # ------------------------------
    # Example single input schedule
    # ------------------------------
    #Extract model dimensions
    states, parameters, species_names = FullModel(None,None,None, dim = True)

    # Parameters and states y0
    k = np.array([random.random() for i in range(parameters)])
    y0 = np.array([1.0 for i in range(states)])

    # Manually defined input event
    bolus_events = [(80, "A", 2.0),(5.0, "C", 1.0)]

    # Simulation event
    t = 240
    t, y = simulate_chain_with_bolus(
        FullModel,
        k=k,
        y0=y0,
        t_start=0.0,
        t_end= t,
        dt=0.01,
        bolus_gen = single_event_generator(bolus_events),
        species_names = species_names,
    )
    for j, state in enumerate(y.T):
        name = species_names[j]
        is_final_product = name == "M"
        plt.plot(
            t,
            state,
            label=name,
            linestyle="--" if is_final_product else "-",
        )
    plt.xlabel("time")
    plt.ylabel("concentration")
    plt.legend()
    plt.show()


    rng = np.random.default_rng(seed=42)  # fixed seed for reproducibility
    # ------------------------------
    # Create N different schedules
    # ------------------------------
    n_sims = 20

    # The bolus schedule
    bolus_schedules: List[List[BolusEvent]] = []
    for i in range(n_sims):
        # Example: between 2 and 6 boluses per simulation
        n_bolus_i = rng.integers(2, 50)  # 2,3,4,5,6
        events_i = generate_random_bolus_events(
            n_bolus=n_bolus_i,
            t_start=0.0,
            t_end=250.0,
            amount_range=(0.5, 3.0),  # tweak as you like min-max of input given
            species_names = species_names[0],
            rng=rng,
        )
        

        gen = single_event_generator(events_i)
        t, y = simulate_chain_with_bolus(FullModel,
            k=k,
            y0=y0,
            t_start=0.0,
            t_end=250.0,
            dt=0.1,
            bolus_gen=gen,
            species_names = species_names,
        )
        
        for j, state in enumerate(y.T):
            name = species_names[j]
            is_final_product = name == "M"
            plt.plot(
                t,
                state,
                label=name,
                linestyle="--" if is_final_product else "-",
            )
        plt.xlabel("time")
        plt.ylabel("concentration")
        plt.legend()
        plt.show()
        
        # Append schedules to call back    
        bolus_schedules.append(events_i)
        
    # ------------------------------
    # Print what should become the single events in a dataset used to train a model
    # ------------------------------
    # Example: inspect the first few schedules
    for i, sched in enumerate(bolus_schedules[:n_sims]):
        print(f"Simulation {i+1} bolus_events:")
        for ev in sched:
            print("   ", ev)
        print()