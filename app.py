import streamlit as st
import simpy
from collections import defaultdict
import math

# === Step 1: Define Stations ===
st.header("Step 1: Station Groups")
num_groups = st.number_input("Number of Station Groups", min_value=1, step=1, value=2)

if "group_names" not in st.session_state or len(st.session_state.group_names) != num_groups:
    st.session_state.group_names = [""] * num_groups
    st.session_state.station_groups = {}
    st.session_state.conveyor_flags = {}
    st.session_state.board_limits = {}
    st.session_state.inter_arrivals = {}

for i in range(num_groups):
    name = st.text_input(f"Group {i+1} Name", key=f"name_{i}").strip().upper()
    st.session_state.group_names[i] = name

    if name:
        eq_count = st.number_input(f"Number of Equipment in {name}", 1, key=f"eq_count_{i}")
        eq_times = [st.number_input(f"Cycle Time for {name} - EQ{j+1} (sec)", 0.1, key=f"ct_{i}_{j+1}") for j in range(eq_count)]
        
        # Conveyor behavior option
        behave_conveyor = st.checkbox(f"Behave like Conveyor for {name}", key=f"conveyor_{i}")
        st.session_state.conveyor_flags[name] = behave_conveyor
        
        board_limit = None
        inter_arrival = None
        if behave_conveyor:
            limit_input = st.text_input(f"Board Limit for {name} (enter integer or 'infinity')", value="infinity", key=f"limit_{i}")
            if limit_input.strip().lower() == "infinity":
                board_limit = math.inf
            else:
                try:
                    board_limit = max(1, int(limit_input))
                except:
                    board_limit = math.inf
            inter_arrival = st.number_input(f"Inter-arrival Time (sec) for {name}", min_value=0.0, value=0.0, step=0.1, key=f"interarr_{i}")
        else:
            board_limit = math.inf
            inter_arrival = 0.0
        
        st.session_state.board_limits[name] = board_limit
        st.session_state.inter_arrivals[name] = inter_arrival
        
        st.session_state.station_groups[name] = {f"{name} - EQ{j+1}": ct for j, ct in enumerate(eq_times)}

# === Step 2: Connections ===
st.markdown("---")
st.header("Step 2: Connect Stations")

if "from_stations" not in st.session_state:
    st.session_state.from_stations = {}
if "connections" not in st.session_state:
    st.session_state.connections = {}

for i, name in enumerate(st.session_state.group_names):
    if not name:
        continue
    with st.expander(f"{name} Connections"):
        from_options = ['START'] + [g for g in st.session_state.group_names if g and g != name]
        to_options = ['STOP'] + [g for g in st.session_state.group_names if g and g != name]
        from_selected = st.multiselect(f"{name} receives from:", from_options, key=f"from_{i}")
        to_selected = st.multiselect(f"{name} sends to:", to_options, key=f"to_{i}")
        st.session_state.from_stations[name] = [] if "START" in from_selected else from_selected
        st.session_state.connections[name] = [] if "STOP" in to_selected else to_selected

# === Step 3: Duration ===
st.markdown("---")
st.header("Step 3: ⏱️ Enter Simulation Duration")
sim_time = st.number_input("Simulation Time (seconds)", min_value=10, value=100, step=10)

if st.button("▶️ Run Simulation"):
    st.session_state.simulate = True
    st.session_state.sim_time = sim_time

# === Run Simulation ===
if st.session_state.get("simulate"):
    station_groups = st.session_state.station_groups
    from_stations = st.session_state.from_stations
    connections = st.session_state.connections
    sim_time = st.session_state.sim_time
    conveyor_flags = st.session_state.conveyor_flags
    board_limits = st.session_state.board_limits
    inter_arrivals = st.session_state.inter_arrivals

    valid_groups = {g: eqs for g, eqs in station_groups.items() if g}

    class FactorySimulation:
        def __init__(self, env, station_groups, duration, connections, from_stations,
                     conveyor_flags, board_limits, inter_arrivals):
            self.env = env
            self.station_groups = station_groups
            self.connections = connections
            self.from_stations = from_stations
            self.duration = duration
            self.conveyor_flags = conveyor_flags
            self.board_limits = board_limits
            self.inter_arrivals = inter_arrivals
            
            # Buffers per group (non-conveyor) or per equipment (conveyor)
            self.buffers = defaultdict(lambda: simpy.Store(env))
            # Resource per equipment with capacity 1 (processing one board at a time)
            self.resources = {eq: simpy.Resource(env, capacity=1) for group in station_groups.values() for eq in group}
            self.cycle_times = {eq: ct for group in station_groups.values() for eq, ct in group.items()}
            self.equipment_to_group = {eq: group for group, eqs in station_groups.items() for eq in eqs}

            # For conveyor equipment: track boards inside and last entry times
            self.equipment_board_count = defaultdict(int)  # boards currently inside equipment
            self.equipment_last_entry = defaultdict(lambda: -math.inf)  # last board entry time per equipment
            
            self.throughput_in = defaultdict(int)
            self.throughput_out = defaultdict(int)
            self.wip_over_time = defaultdict(list)
            self.time_points = []
            self.equipment_busy_time = defaultdict(float)
            self.board_id = 1
            self.wip_interval = 5
            
            env.process(self.track_wip())

        def equipment_worker(self, eq):
            group = self.equipment_to_group[eq]
            is_conveyor = self.conveyor_flags.get(group, False)

            while True:
                if is_conveyor:
                    # Conveyor equipment has its own buffer queue
                    board = yield self.buffers[eq].get()
                else:
                    # Non conveyor: get board from group buffer
                    board = yield self.buffers[group].get()
                
                self.throughput_in[eq] += 1
                with self.resources[eq].request() as req:
                    yield req
                    start = self.env.now
                    yield self.env.timeout(self.cycle_times[eq])
                    end = self.env.now
                    self.equipment_busy_time[eq] += (end - start)
                
                self.throughput_out[eq] += 1

                # For conveyor, reduce board count when board leaves equipment
                if is_conveyor:
                    self.equipment_board_count[eq] -= 1
                
                # Pass board to next connected groups
                for tgt in self.connections.get(group, []):
                    if self.conveyor_flags.get(tgt, False):
                        # Conveyor tgt => put in each equipment buffer of that group in round robin or equal?
                        # For simplicity, put into group buffer (or we could do load balancing)
                        # Here we put into group buffer to be dispatched to equipment worker later
                        yield self.buffers[tgt].put(board)
                    else:
                        yield self.buffers[tgt].put(board)

        def conveyor_dispatcher(self, eq):
            """Process to dispatch boards into conveyor equipment respecting board limits and inter-arrival times"""
            group = self.equipment_to_group[eq]
            board_limit = self.board_limits.get(group, math.inf)
            inter_arrival = self.inter_arrivals.get(group, 0.0)

            last_entry_time = -math.inf

            while self.env.now < self.duration:
                # Check if we can accept new board into equipment
                can_enter = (self.equipment_board_count[eq] < board_limit)
                time_ok = (self.env.now - self.equipment_last_entry[eq]) >= inter_arrival

                if can_enter and time_ok:
                    # Only feed new boards from group buffer or feeder
                    if self.buffers[group].items:
                        board = yield self.buffers[group].get()
                        # Put board into equipment conveyor buffer
                        yield self.buffers[eq].put(board)
                        self.equipment_board_count[eq] += 1
                        self.equipment_last_entry[eq] = self.env.now
                    else:
                        # No boards waiting, wait some time
                        yield self.env.timeout(0.1)
                else:
                    # Wait until conditions met
                    yield self.env.timeout(0.1)

        def feeder(self):
            start_groups = [g for g in self.station_groups if not self.from_stations.get(g)]
            while self.env.now < self.duration:
                for g in start_groups:
                    board = f"Board-{self.board_id:03d}"
                    self.board_id += 1
                    # Put board in group buffer
                    yield self.buffers[g].put(board)
                yield self.env.timeout(1)

        def track_wip(self):
            while self.env.now < self.duration:
                self.time_points.append(self.env.now)
                for group in self.station_groups:
                    prev_out = sum(
                        sim.throughput_out[eq] for g in self.from_stations.get(group, [])
                        for eq in self.station_groups.get(g, [])
                    )
                    curr_in = sum(sim.throughput_in[eq] for eq in self.station_groups[group])
                    wip = max(0, prev_out - curr_in) if self.from_stations.get(group) else 0
                    self.wip_over_time[group].append(wip)
                yield self.env.timeout(self.wip_interval)

        def run(self):
            # Start equipment worker processes
            for group in self.station_groups:
                is_conveyor = self.conveyor_flags.get(group, False)
                for eq in self.station_groups[group]:
                    self.env.process(self.equipment_worker(eq))
                    if is_conveyor:
                        # Also start dispatcher per equipment
                        self.env.process(self.conveyor_dispatcher(eq))
            # Start feeder for initial boards
            self.env.process(self.feeder())

    env = simpy.Environment()
    sim = FactorySimulation(env, valid_groups, sim_time, connections, from_stations,
                            conveyor_flags, board_limits, inter_arrivals)
    sim.run()
    env.run(until=sim_time)

# === Check for Required Variables ===
if 'valid_groups' not in locals() or 'sim' not in locals() or 'from_stations' not in locals() or 'sim_time' not in locals():
    st.warning("❗ Run the simulation first to generate results.")
    st.stop()

# === Results Summary ===
st.markdown("---")
st.subheader("Simulation Completed")

for group in valid_groups:
    st.write(f"**Group: {group}**")
    st.write(f"Boards processed: {sum(sim.throughput_out[eq] for eq in valid_groups[group])}")
    conveyor = conveyor_flags.get(group, False)
    st.write(f"Behaves like conveyor: {conveyor}")
    if conveyor:
        st.write(f"Board limit: {st.session_state.board_limits[group]}")
        st.write(f"
