import os
import io
import math
import pandas as pd
import numpy as np
from scipy.stats import norm
from flask import Flask, render_template, request, jsonify
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import AgglomerativeClustering

app = Flask(__name__)

DEV_TEST_SESSIONS_TO_TRIM = 5

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
    
    error_sessions = df_subset[df_subset['Accuracy_Pct'] < 100]
    error_persistence = safe_float(error_sessions['Total_Answers'].mean()) if not error_sessions.empty else 0.0

    stability_index = safe_float(df_subset['Accuracy_Pct'].std() if len(df_subset) > 1 else 0.0)
    mean_correct_latency = safe_float(df_subset['Avg_Correct_Time_Ms'].mean())

    err_latencies = df_subset[df_subset['Avg_Incorrect_Time_Ms'] > 0]['Avg_Incorrect_Time_Ms']
    mean_error_latency = safe_float(err_latencies.mean()) if not err_latencies.empty else 0.0

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
        "error_persistence": round(error_persistence, 1),
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

    df['Session_Break'] = (df['Total_Answers'] < df['Total_Answers'].shift(1, fill_value=0)) | (df['Turn_ID'] != df['Turn_ID'].shift(1, fill_value=-1))
    df['Session_Group'] = df['Session_Break'].cumsum()

    sessions = df.groupby('Session_Group').last().reset_index(drop=True)

    if len(sessions) > DEV_TEST_SESSIONS_TO_TRIM:
        sessions = sessions.iloc[DEV_TEST_SESSIONS_TO_TRIM:].copy().reset_index(drop=True)

    sessions['Global_Session_Index'] = sessions.index + 1

    feature_cols = ['Accuracy_Pct', 'Avg_Correct_Time_Ms', 'Avg_Incorrect_Time_Ms', 'Total_Answers', 'Session_Duration_Ms']
    X = sessions[feature_cols].copy()

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    cluster_model = AgglomerativeClustering(n_clusters=None, distance_threshold=2.5, linkage='ward')
    sessions['Kid_Cluster'] = cluster_model.fit_predict(X_scaled)

    unique_clusters = sessions['Kid_Cluster'].unique()
    kid_map = {cluster_id: f"Child {idx + 1}" for idx, cluster_id in enumerate(unique_clusters)}
    sessions['User_Name'] = sessions['Kid_Cluster'].map(kid_map)

    overall_stats = compute_metrics(sessions)

    users_data = {}
    for user_name in sorted(sessions['User_Name'].unique()):
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
    user_color_map = {name: color_palette[i % len(color_palette)] for i, name in enumerate(sorted(sessions['User_Name'].unique()))}

    for _, row in sessions.iterrows():
        global_scatter.append({
            "session_idx": int(row['Global_Session_Index']),
            "accuracy": int(row['Accuracy_Pct']),
            "user": row['User_Name'],
            "color": user_color_map[row['User_Name']]
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