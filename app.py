import streamlit as st
import simpy
import pandas as pd
from collections import defaultdict
from io import BytesIO

st.set_page_config(layout="wide")

st.title("🏭 Factory Flow Simulation Tool")

col1, col2 = st.columns([2, 3])

# === Step 1: Define Stations ===
with col1:
    st.header("Step 1: Define Stations")
    num_groups = st.number_input("How many station groups?", min_value=1, value=3, step=1)

    if "group_names" not in st.session_state:
        st.session_state.group_names = ["" for _ in range(num_groups)]
    else:
        if len(st.session_state.group_names) != num_groups:
            st.session_state.group_names = ["" for _ in range(num_groups)]

    if "station_groups" not in st.session_state:
        st.session_state.station_groups = {}

    conveyor_flags = {}
    max_products_dict = {}

    for i in range(num_groups):
        with st.expander(f"Group {i+1} Configuration"):
            group_name = st.text_input(f"Enter group name for Group {i+1}", key=f"group_name_{i}")
            st.session_state.group_names[i] = group_name
            num_eq = st.number_input(f"How many equipment in {group_name}?", min_value=1, value=2, key=f"num_eq_{i}")
            behave_like_conveyor = st.checkbox("Behave like Conveyor", key=f"conveyor_{i}")
            conveyor_flags[group_name] = behave_like_conveyor
            
            max_products = st.number_input(
                "Maximum Products in Station (use a very large number to simulate ∞)",
                min_value=1,
                value=1,                  # default changed to 1
                max_value=1_000_000_000,  # large max to simulate infinity
                step=1,
                key=f"max_products_{i}"
            )
            max_products_dict[group_name] = max_products
            
            equipment = {}
            for j in range(int(num_eq)):
                eq_name = st.text_input(f"Equipment {j+1} name in {group_name}", key=f"eq_{i}_{j}")
                cycle_time = st.number_input(f"Cycle time (sec) for {eq_name}", min_value=1.0, value=5.0, step=1.0, key=f"ct_{i}_{j}")
                if eq_name:
                    equipment[eq_name] = cycle_time
            if group_name:
                st.session_state.station_groups[group_name] = equipment

# === Step 2: Connections ===
with col2:
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
    valid_groups = {g: eqs for g, eqs in station_groups.items() if g}

    class FactorySimulation:
        def __init__(self, env, station_groups, duration, connections, from_stations):
            self.env = env
            self.station_groups = station_groups
            self.connections = connections
            self.from_stations = from_stations
            self.duration = duration
            self.buffers = defaultdict(lambda: simpy.Store(env))
            self.resources = {eq: simpy.Resource(env, capacity=1)
                              for group in station_groups.values() for eq in group}
            self.cycle_times = {eq: ct for group in station_groups.values() for eq, ct in group.items()}
            self.equipment_to_group = {eq: group for group, eqs in station_groups.items() for eq in eqs}
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
            while True:
                board = yield self.buffers[group].get()
                self.throughput_in[eq] += 1
                with self.resources[eq].request() as req:
                    yield req
                    start = self.env.now
                    yield self.env.timeout(self.cycle_times[eq])
                    end = self.env.now
                    self.equipment_busy_time[eq] += (end - start)
                self.throughput_out[eq] += 1
                for tgt in self.connections.get(group, []):
                    yield self.buffers[tgt].put(board)

        def feeder(self):
            start_groups = [g for g in self.station_groups if not self.from_stations.get(g)]
            while self.env.now < self.duration:
                for g in start_groups:
                    board = f"Board-{self.board_id:03d}"
                    self.board_id += 1
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
            for group in self.station_groups:
                for eq in self.station_groups[group]:
                    self.env.process(self.equipment_worker(eq))
            self.env.process(self.feeder())

    env = simpy.Environment()
    sim = FactorySimulation(env, valid_groups, sim_time, connections, from_stations)
    sim.run()
    env.run(until=sim_time)

# === Check for Required Variables ===
if 'valid_groups' not in locals() or 'sim' not in locals() or 'from_stations' not in locals() or 'sim_time' not in locals():
    st.warning("❗ Run the simulation first to generate results.")
    st.stop()

# === Results Summary ===
st.markdown("---")
st.subheader("📊 Simulation Results Summary")
groups = list(valid_groups.keys())
agg = defaultdict(lambda: {'in': 0, 'out': 0, 'busy': 0, 'count': 0, 'cycle_times': [], 'wip': 0})

for group in groups:
    eqs = valid_groups[group]
    for eq in eqs:
        agg[group]['in'] += sim.throughput_in.get(eq, 0)
        agg[group]['out'] += sim.throughput_out.get(eq, 0)
        agg[group]['busy'] += sim.equipment_busy_time.get(eq, 0)
        agg[group]['cycle_times'].append(sim.cycle_times.get(eq, 0))
        agg[group]['count'] += 1
    prev_out = sum(sim.throughput_out.get(eq, 0) for g in from_stations.get(group, []) for eq in valid_groups.get(g, []))
    curr_in = agg[group]['in']
    agg[group]['wip'] = max(0, prev_out - curr_in)

# Prepare DataFrame
df = pd.DataFrame([{
    "Station Group": g,
    "Boards In": agg[g]['in'],
    "Boards Out": agg[g]['out'],
    "WIP": agg[g]['wip'],
    "Number of Equipment": agg[g]['count'],
    "Cycle Times (sec)": ", ".join(str(round(ct, 1)) for ct in agg[g]['cycle_times']),
    "Utilization (%)": round((agg[g]['busy'] / (sim_time * agg[g]['count'])) * 100, 1) if agg[g]['count'] > 0 else 0
} for g in groups])

st.dataframe(df, use_container_width=True)

# Excel download
towrite = BytesIO()
df.to_excel(towrite, index=False, sheet_name="Summary")
towrite.seek(0)
st.download_button("📥 Download Summary Excel", data=towrite, file_name="factory_sim_summary.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
