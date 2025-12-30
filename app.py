import os
import pandas as pd
import numpy as np
import lightgbm as lgb
import networkx as nx
from geopy.distance import geodesic
from flask import Flask, jsonify, request, render_template_string

# ==========================================
# 1. CONFIGURATION & HTML TEMPLATE
# ==========================================
app = Flask(__name__)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_FOLDER = os.path.join(BASE_DIR, 'Data')


# The entire Frontend (HTML+CSS+JS) is here for simplicity
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Delhi Metro AI Navigator</title>
    <style>
        :root { --primary: #2c3e50; --accent: #e74c3c; --bg: #f4f7f6; }
        body { font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; background: var(--bg); margin: 0; padding: 20px; color: #333; }
        .container { max-width: 600px; margin: 0 auto; background: white; padding: 30px; border-radius: 15px; box-shadow: 0 10px 25px rgba(0,0,0,0.1); }
        h1 { text-align: center; color: var(--primary); margin-bottom: 30px; }
        .form-group { margin-bottom: 15px; }
        label { display: block; margin-bottom: 5px; font-weight: 600; }
        input { width: 100%; padding: 12px; border: 2px solid #ddd; border-radius: 8px; font-size: 16px; box-sizing: border-box; }
        input:focus { border-color: var(--primary); outline: none; }
        button { width: 100%; padding: 15px; background: var(--primary); color: white; border: none; border-radius: 8px; font-size: 18px; cursor: pointer; transition: 0.3s; }
        button:hover { background: #34495e; }
        
        /* Results Section */
        #result { margin-top: 30px; display: none; }
        .summary-card { background: #e8f6f3; padding: 15px; border-radius: 10px; border-left: 5px solid #1abc9c; margin-bottom: 20px; }
        .summary-title { font-size: 1.2em; font-weight: bold; }
        .timeline { position: relative; padding-left: 20px; border-left: 2px solid #ddd; }
        .step { margin-bottom: 20px; position: relative; padding-left: 20px; }
        .step::before { content: ''; position: absolute; left: -26px; top: 0; width: 10px; height: 10px; border-radius: 50%; background: #999; border: 2px solid #fff; }
        
        /* Dynamic Line Colors */
        .line-tag { display: inline-block; padding: 2px 8px; border-radius: 4px; color: white; font-size: 0.8em; font-weight: bold; }
        .Yellow { background: #FFC107; color: black; }
        .Blue { background: #2196F3; }
        .Red { background: #F44336; }
        .Violet { background: #9C27B0; }
        .Green { background: #4CAF50; }
        .Pink { background: #E91E63; }
        .Magenta { background: #880E4F; }
        .Orange { background: #FF9800; }
        .Aqua { background: #00BCD4; }
        .Gray { background: #607D8B; }
        .Rapid { background: #3F51B5; }
        .Transfer { background: #333; }

        .segment-info { background: #fff; padding: 10px; border: 1px solid #eee; border-radius: 8px; box-shadow: 0 2px 5px rgba(0,0,0,0.05); }
        .alert-transfer { background: #fff3cd; color: #856404; padding: 10px; border-radius: 5px; margin-top: 5px; font-weight: bold; border-left: 4px solid #ffeeba; }
    </style>
</head>
<body>
    <div class="container">
        <h1>🚇 Delhi Metro ETA </h1>
        
        <div class="form-group">
            <label>Start Station</label>
            <input type="text" id="start" list="station-list" placeholder="e.g. Rajiv Chowk">
        </div>
        
        <div class="form-group">
            <label>Destination</label>
            <input type="text" id="end" list="station-list" placeholder="e.g. Noida City Centre">
        </div>

        <div class="form-group">
            <label>Departure Time</label>
            <input type="time" id="time" value="09:00">
        </div>

        <button onclick="findRoute()">Find Fastest Route</button>

        <div id="result">
            <div class="summary-card">
                <div class="summary-title">⏱ Total Time: <span id="total-time"></span> min</div>
                <div>Estimated Arrival: <span id="arrival-time"></span></div>
            </div>
            <div class="timeline" id="timeline-box"></div>
        </div>
    </div>

    <datalist id="station-list"></datalist>

    <script>
        // Load stations for autocomplete
        fetch('/api/stations').then(r => r.json()).then(data => {
            const list = document.getElementById('station-list');
            data.forEach(stn => {
                const opt = document.createElement('option');
                opt.value = stn;
                list.appendChild(opt);
            });
        });

        async function findRoute() {
            const start = document.getElementById('start').value;
            const end = document.getElementById('end').value;
            const time = document.getElementById('time').value;
            const btn = document.querySelector('button');
            const resultDiv = document.getElementById('result');
            
            if(!start || !end) return alert("Please enter both stations");

            btn.innerText = "Computing AI Route...";
            resultDiv.style.display = 'none';

            try {
                const res = await fetch('/api/predict', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({ start, end, time })
                });
                
                const data = await res.json();
                
                if(data.error) {
                    alert(data.error);
                } else {
                    renderResult(data);
                }
            } catch(e) {
                alert("Server Error");
            } finally {
                btn.innerText = "Find Fastest Route";
            }
        }

        function renderResult(data) {
            document.getElementById('result').style.display = 'block';
            document.getElementById('total-time').innerText = data.total_time.toFixed(0);
            document.getElementById('arrival-time').innerText = data.arrival_time;
            
            const timeline = document.getElementById('timeline-box');
            timeline.innerHTML = '';

            data.segments.forEach(seg => {
                const isTransfer = seg.line === 'Transfer';
                const lineClass = seg.line.split(' ')[0]; // Extract 'Yellow' from 'Yellow Line'
                
                let html = `
                    <div class="step">
                        <div class="segment-info">
                            <div>
                                <span class="line-tag ${lineClass}">${seg.line}</span>
                                <b>${seg.start}</b> ➝ <b>${seg.end}</b>
                            </div>
                            <div style="margin-top:5px; color:#666; font-size:0.9em;">
                                ${isTransfer ? 'Walk' : 'Ride ' + seg.stops + ' stops'} • ${seg.duration.toFixed(1)} min
                            </div>
                        </div>
                `;
                
                if (seg.switch_alert) {
                    html += `<div class="alert-transfer">⚠ CHANGE TRAINS HERE</div>`;
                }
                
                html += `</div>`;
                timeline.innerHTML += html;
            });
        }
    </script>
</body>
</html>
"""

# ==========================================
# 2. GLOBAL VARIABLES (Loaded on Startup)
# ==========================================
G = None
model = None
stop_names_map = {}
stop_names_reverse = {}
features_list = ['stop_id', 'next_stop_id', 'distance_km', 'curr_dep_min']

# ==========================================
# 3. BACKEND LOGIC
# ==========================================
def load_data_and_train():
    global G, model, stop_names_map, stop_names_reverse
    print("🚀 [System] Loading Data Files...")
    
    stops = pd.read_csv(os.path.join(DATA_FOLDER, 'stops.txt'))
    stop_times = pd.read_csv(os.path.join(DATA_FOLDER, 'stop_times.txt'))
    routes = pd.read_csv(os.path.join(DATA_FOLDER, 'routes.txt'))
    trips = pd.read_csv(os.path.join(DATA_FOLDER, 'trips.txt'))

    # Store Maps
    stop_names_map = stops.set_index('stop_id')['stop_name'].to_dict()
    stop_names_reverse = {v.lower(): k for k, v in stop_names_map.items()}

    # Helper: Line Name
    def extract_line_name(long_name):
        if pd.isna(long_name): return "Metro"
        return long_name.split('_')[0].title() + " Line"
    routes['line_name'] = routes['route_long_name'].apply(extract_line_name)
    trip_line_map = trips.merge(routes[['route_id', 'line_name']], on='route_id').set_index('trip_id')['line_name'].to_dict()

    # Process Stop Times
    print("⚙️ [System] Processing Schedules...")
    stop_times = stop_times.sort_values(['trip_id', 'stop_sequence'])
    stop_times['next_arrival'] = stop_times['arrival_time'].shift(-1)
    stop_times['next_stop_id'] = stop_times['stop_id'].shift(-1)
    stop_times['next_trip_id'] = stop_times['trip_id'].shift(-1)
    
    data = stop_times[stop_times['trip_id'] == stop_times['next_trip_id']].copy()
    data['line_name'] = data['trip_id'].map(trip_line_map)
    
    # Vectorized Time Conversion
    def to_min(series):
        t = pd.to_datetime(series, format='%H:%M:%S', errors='coerce')
        return t.dt.hour * 60 + t.dt.minute + t.dt.second/60
    
    data['curr_dep_min'] = to_min(data['departure_time'])
    data['next_arr_min'] = to_min(data['next_arrival'])
    data['duration'] = data['next_arr_min'] - data['curr_dep_min']
    data = data[data['duration'] > 0]

    # Geodesic Distances
    data = data.merge(stops[['stop_id', 'stop_lat', 'stop_lon']], on='stop_id')
    data.rename(columns={'stop_lat': 'lat_start', 'stop_lon': 'lon_start'}, inplace=True)
    data = data.merge(stops[['stop_id', 'stop_lat', 'stop_lon']], left_on='next_stop_id', right_on='stop_id', suffixes=('', '_next'))
    data.rename(columns={'stop_lat': 'lat_end', 'stop_lon': 'lon_end'}, inplace=True)
    
    data['distance_km'] = data.apply(lambda x: geodesic((x['lat_start'], x['lon_start']), (x['lat_end'], x['lon_end'])).km, axis=1)

    # Train Model
    print("🧠 [AI] Training LightGBM Model...")
    X = data[features_list].fillna(0)
    y = data['duration'].fillna(0)
    train_ds = lgb.Dataset(X, label=y, categorical_feature=['stop_id', 'next_stop_id'])
    
    params = {'objective': 'regression', 'metric': 'rmse', 'boosting_type': 'gbdt', 'num_leaves': 31, 'learning_rate': 0.1, 'verbose': -1}
    model = lgb.train(params, train_ds, num_boost_round=300)

    # Build Graph
    print("🕸 [Graph] Building Routing Network...")
    G = nx.DiGraph()
    # Nodes
    for stop_id, name in stop_names_map.items():
        G.add_node(stop_id, name=name)
    
    # Rail Edges
    unique_hops = data[['stop_id', 'next_stop_id', 'line_name', 'distance_km']].drop_duplicates(subset=['stop_id', 'next_stop_id'])
    for _, row in unique_hops.iterrows():
        G.add_edge(row['stop_id'], row['next_stop_id'], line=row['line_name'], dist=row['distance_km'], type='rail')
        
    # Transfer Edges
    stop_names_group = stops.groupby('stop_name')['stop_id'].apply(list)
    for ids in stop_names_group:
        if len(ids) > 1:
            for i in ids:
                for j in ids:
                    if i != j: G.add_edge(i, j, line='Transfer', dist=0.05, type='transfer')

    print("✅ System Ready!")

# ==========================================
# 4. FLASK ROUTES
# ==========================================
@app.route('/')
def index():
    return render_template_string(HTML_TEMPLATE)

@app.route('/api/stations')
def api_stations():
    # Return sorted unique names for autocomplete
    return jsonify(sorted(list(set(stop_names_map.values()))))

@app.route('/api/predict', methods=['POST'])
def api_predict():
    req = request.json
    start_name = req.get('start')
    end_name = req.get('end')
    time_str = req.get('time', '09:00')

    # Parse Time
    h, m = map(int, time_str.split(':'))
    start_time_min = h * 60 + m

    # Resolve IDs
    u = stop_names_reverse.get(start_name.lower())
    v = stop_names_reverse.get(end_name.lower())

    if not u or not v:
        return jsonify({'error': 'Stations not found. Please select from the list.'}), 400

    try:
        path = nx.shortest_path(G, source=u, target=v)
    except nx.NetworkXNoPath:
        return jsonify({'error': 'No route available between these stations.'}), 400

    # --- VECTORIZED PREDICTION LOGIC ---
    path_hops = []
    curr_t = start_time_min

    for i in range(len(path) - 1):
        s1, s2 = path[i], path[i+1]
        edge = G[s1][s2]
        
        path_hops.append({
            'stop_id': s1,
            'next_stop_id': s2,
            'distance_km': edge['dist'],
            'curr_dep_min': curr_t,
            'line': edge['line'],
            'type': edge['type']
        })
    
    df_path = pd.DataFrame(path_hops)
    
    # Predict Rail
    rail_mask = df_path['type'] == 'rail'
    if rail_mask.any():
        preds = model.predict(df_path.loc[rail_mask, features_list])
        df_path.loc[rail_mask, 'pred_time'] = preds
    
    # Predict Transfer
    df_path.loc[~rail_mask, 'pred_time'] = 5.0 # Fixed transfer time
    
    # Add Headway (Wait Time) logic
    df_path['wait_time'] = 0.0
    if len(df_path) > 0: df_path.loc[0, 'wait_time'] = 2.5 # Initial wait
    
    # Add wait after transfers
    transfer_indices = df_path.index[df_path['type'] == 'transfer'].tolist()
    for idx in transfer_indices:
        if idx + 1 in df_path.index:
            df_path.loc[idx + 1, 'wait_time'] = 2.5

    df_path['total_hop_time'] = df_path['pred_time'] + df_path['wait_time']

    # --- FORMAT JSON RESPONSE ---
    segments = []
    current_line = None
    seg_start = stop_names_map[path[0]]
    seg_duration = 0
    seg_stops = 0

    for _, row in df_path.iterrows():
        line = row['line']
        
        # If line changes, push previous segment
        if line != current_line:
            if current_line:
                segments.append({
                    'line': current_line,
                    'start': seg_start,
                    'end': stop_names_map[row['stop_id']],
                    'duration': seg_duration,
                    'stops': seg_stops,
                    'switch_alert': (current_line != 'Transfer' and line != 'Transfer')
                })
            
            current_line = line
            seg_start = stop_names_map[row['stop_id']]
            seg_duration = 0
            seg_stops = 0
        
        seg_duration += row['total_hop_time']
        seg_stops += 1
    
    # Push final segment
    segments.append({
        'line': current_line,
        'start': seg_start,
        'end': stop_names_map[path[-1]],
        'duration': seg_duration,
        'stops': seg_stops,
        'switch_alert': False
    })

    total_time = df_path['total_hop_time'].sum()
    arrival_min = start_time_min + total_time
    arr_h = int((arrival_min // 60) % 24)
    arr_m = int(arrival_min % 60)

    return jsonify({
        'total_time': total_time,
        'arrival_time': f"{arr_h:02d}:{arr_m:02d}",
        'segments': segments
    })

# Main
if __name__ == '__main__':
    load_data_and_train()
    print("🌍 Server starting at http://127.0.0.1:5000")
    app.run(debug=True)
