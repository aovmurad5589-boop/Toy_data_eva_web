import os
import io
import math
import pandas as pd
import numpy as np
import hdbscan
from scipy.stats import norm
from flask import Flask, render_template, request, jsonify
from sklearn.preprocessing import StandardScaler

app = Flask(__name__)

def safe_float(val, default=0.0):
    """Ensures numbers are valid JSON-compliant floats (no NaN or Inf)."""
    if val is None or pd.isna(val) or math.isnan(val) or math.isinf(val):
        return default
    return float(val)

def mann_kendall_test(x):
    n = len(x)
    if n < 3:
        return 0.0, 1.0
    s = 0
    for i in range(n - 1):
        for j in range(i + 1, n):
            s += np.sign(x[j] - x[i])
    
    var_s = (n * (n - 1) * (2 * n + 5)) / 18.0
    if var_s == 0:
        return float(s), 1.0
    
    if s > 0:
        z = (s - 1) / np.sqrt(var_s)
    elif s < 0:
        z = (s + 1) / np.sqrt(var_s)
    else:
        z = 0.0
        
    p_value = 2 * (1 - norm.cdf(abs(z)))
    return safe_float(z, 0.0), safe_float(p_value, 1.0)

def compute_metrics(df_subset):
    if df_subset.empty:
        return {
            "baseline_accuracy": 0.0, "error_rate": 0.0, "error_persistence": 0.0,
            "stability_index": 0.0, "mean_correct_latency": 0.0, "mean_error_latency": 0.0,
            "latency_variance": 0.0, "avg_session_duration": 0.0, "throughput_volume": 0.0,
            "fatigue_slope": 0.0, "trend_status": "Insufficient Data ⚖️", "mann_kendall_z": 0.0,
            "mann_kendall_p": 1.0, "efficiency_velocity": 0.0, "total_sessions": 0
        }

    baseline_accuracy = safe_float(df_subset['Accuracy_Pct'].mean())
    error_rate = safe_float(100 - baseline_accuracy)
    
    # CORRECTED ERROR PERSISTENCE: Ratio of errors to total answers in imperfect sessions
    error_sessions = df_subset[df_subset['Accuracy_Pct'] < 100]
    if not error_sessions.empty:
        total_err_answers = error_sessions['Total_Answers'].sum()
        # Estimating error volume based on accuracy drop
        estimated_errors = (error_sessions['Total_Answers'] * (100 - error_sessions['Accuracy_Pct']) / 100.0).sum()
        error_persistence = safe_float((estimated_errors / total_err_answers) * 100.0) if total_err_answers > 0 else 0.0
    else:
        error_persistence = 0.0

    stability_index = safe_float(df_subset['Accuracy_Pct'].std() if len(df_subset) > 1 else 0.0)
    
    # CORRECTED LATENCY: Volume-weighted averages instead of raw mean of means
    total_answers_vol = df_subset['Total_Answers'].replace(0, 1)
    mean_correct_latency = safe_float(np.average(df_subset['Avg_Correct_Time_Ms'], weights=total_answers_vol))
    
    err_subset = df_subset[df_subset['Avg_Incorrect_Time_Ms'] > 0]
    if not err_subset.empty:
        mean_error_latency = safe_float(np.average(err_subset['Avg_Incorrect_Time_Ms'], weights=err_subset['Total_Answers']))
    else:
        mean_error_latency = 0.0

    latency_variance = safe_float(df_subset['Avg_Correct_Time_Ms'].var() if len(df_subset) > 1 else 0.0)
    avg_session_duration = safe_float((df_subset['Session_Duration_Ms'] / 1000.0).mean())
    throughput_volume = safe_float(df_subset['Total_Answers'].mean())

    if len(df_subset) > 1:
        x = np.arange(len(df_subset))
        y = df_subset['Avg_Correct_Time_Ms'].values
        slope, _ = np.polyfit(x, y, 1)
        fatigue_slope = safe_float(slope)
    else:
        fatigue_slope = 0.0

    mk_z, mk_p = mann_kendall_test(df_subset['Accuracy_Pct'].values)
    if mk_z > 1.96 and mk_p < 0.05:
        trend_status = "Accelerating Upward 🚀"
    elif mk_z < -1.96 and mk_p < 0.05:
        trend_status = "Declining / Fatigued 📉"
    else:
        trend_status = "Stable / Plateau ⚖️"

    efficiencies = df_subset['Accuracy_Pct'] / np.log(df_subset['Avg_Correct_Time_Ms'].replace(0, 1))
    if len(efficiencies) > 1:
        eff_slope, _ = np.polyfit(np.arange(len(efficiencies)), efficiencies.values, 1)
        efficiency_velocity = safe_float(eff_slope)
    else:
        efficiency_velocity = 0.0

    return {
        "baseline_accuracy": round(baseline_accuracy, 2),
        "error_rate": round(error_rate, 2),
        "error_persistence": round(error_persistence, 1), # Now true error density %
        "stability_index": round(stability_index, 2),
        "mean_correct_latency": round(mean_correct_latency, 1),
        "mean_error_latency": round(mean_error_latency, 1),
        "latency_variance": round(latency_variance, 1),
        "avg_session_duration": round(avg_session_duration, 1),
        "throughput_volume": round(throughput_volume, 1),
        "fatigue_slope": round(fatigue_slope, 2),
        "trend_status": trend_status,
        "mann_kendall_z": round(mk_z, 2),
        "mann_kendall_p": round(mk_p, 4),
        "efficiency_velocity": round(efficiency_velocity, 3),
        "total_sessions": int(len(df_subset))
    }

def process_switch_play_data(csv_file):
    df = pd.read_csv(csv_file)

    # 1. Segment raw telemetry into logical sessions
    df['Session_Break'] = (df['Total_Answers'] < df['Total_Answers'].shift(1, fill_value=0)) | (df['Turn_ID'] != df['Turn_ID'].shift(1, fill_value=-1))
    df['Session_Group'] = df['Session_Break'].cumsum()

    sessions = df.groupby('Session_Group').last().reset_index(drop=True)

    # 2. INTELLIGENT DYNAMIC TRIM:
    # Purge front-loaded developer debug runs (Accuracy >= 95% and Correct Latency < 400ms within first 20 records)
    if not sessions.empty:
        is_early_window = sessions.index < 20
        is_developer_signature = (sessions['Accuracy_Pct'] >= 95) & (sessions['Avg_Correct_Time_Ms'] < 400)
        sessions = sessions[~(is_early_window & is_developer_signature)].copy().reset_index(drop=True)

    sessions['Global_Session_Index'] = sessions.index + 1

    # 3. Feature Engineering (Including delta transitions for sequence awareness)
    sessions['Time_Delta'] = sessions['Turn_ID'].diff().fillna(1)
    feature_cols = ['Accuracy_Pct', 'Avg_Correct_Time_Ms', 'Avg_Incorrect_Time_Ms', 'Total_Answers', 'Session_Duration_Ms', 'Time_Delta']
    X = sessions[feature_cols].copy()

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # 4. HDBSCAN Density-Based Clustering (Handles varying densities and noise/erratic sessions natively)
    cluster_model = hdbscan.HDBSCAN(min_cluster_size=2, min_samples=1, metric='euclidean')
    sessions['Kid_Cluster'] = cluster_model.fit_predict(X_scaled)

    # 5. User Mapping (Isolating valid clusters from HDBSCAN noise label '-1')
    unique_clusters = [c for c in sessions['Kid_Cluster'].unique() if c != -1]
    kid_map = {cluster_id: f"Child {idx + 1}" for idx, cluster_id in enumerate(unique_clusters)}
    kid_map[-1] = "Transition State"
    sessions['User_Name'] = sessions['Kid_Cluster'].map(kid_map)

    # Compute overall stats excluding noise/transition anomalies for accuracy
    valid_sessions = sessions[sessions['User_Name'] != "Transition State"]
    overall_stats = compute_metrics(valid_sessions if not valid_sessions.empty else sessions)

    users_data = {}
    for user_name in sorted(sessions['User_Name'].unique()):
        if user_name == "Transition State":
            continue
        user_df = sessions[sessions['User_Name'] == user_name]
        user_stats = compute_metrics(user_df)

        scatter_points = []
        for _, row in user_df.iterrows():
            scatter_points.append({
                "session_idx": int(row['Global_Session_Index']),
                "accuracy": int(row['Accuracy_Pct']),
                "latency": safe_float(row['Avg_Correct_Time_Ms']),
                "answers": int(row['Total_Answers'])
            })

        users_data[user_name] = {
            "stats": user_stats,
            "scatter": scatter_points
        }

    global_scatter = []
    color_palette = ['#3498db', '#e74c3c', '#2ecc71', '#9b59b6', '#f39c12', '#1abc9c', '#d35400', '#34495e']
    valid_users = sorted([u for u in sessions['User_Name'].unique() if u != "Transition State"])
    user_color_map = {name: color_palette[i % len(color_palette)] for i, name in enumerate(valid_users)}
    user_color_map["Transition State"] = "#95a5a6" # Grey fallback for unassigned noise

    for _, row in sessions.iterrows():
        global_scatter.append({
            "session_idx": int(row['Global_Session_Index']),
            "accuracy": int(row['Accuracy_Pct']),
            "user": row['User_Name'],
            "color": user_color_map.get(row['User_Name'], '#95a5a6')
        })

    return {
        "overall": overall_stats,
        "users": users_data,
        "global_scatter": global_scatter
    }

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    try:
        results = process_switch_play_data(file)
        return jsonify(results)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000)