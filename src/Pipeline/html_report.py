import os
import datetime
import json

def generate_html_report(report_json_path, cm_path, f1_plot_path, output_path):
    with open(report_json_path, 'r') as f:
        report = json.load(f)

    timestamp = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    accuracy = report.get('accuracy', 0)
    weighted_f1 = report.get('weighted avg', {}).get('f1-score', 0)

    # HTML skeleton
    html = f"""
    <html>
    <head>
        <title>Model Test Report</title>
        <style>
            body {{ font-family: Arial, sans-serif; padding: 20px; }}
            table {{ border-collapse: collapse; width: 100%; }}
            th, td {{ border: 1px solid #ddd; padding: 8px; text-align: center; }}
            th {{ background-color: #f2f2f2; }}
            img {{ max-width: 100%; height: auto; }}
        </style>
    </head>
    <body>
        <h1>🧠 Dementia Classification Report</h1>
        <p><b>Date:</b> {timestamp}</p>
        <p><b>Accuracy:</b> {accuracy:.4f}</p>
        <p><b>Weighted F1 Score:</b> {weighted_f1:.4f}</p>

        <h2>📄 Classification Report</h2>
        <table>
            <tr><th>Class</th><th>Precision</th><th>Recall</th><th>F1-Score</th><th>Support</th></tr>
    """

    for label, metrics in report.items():
        if isinstance(metrics, dict):
            html += f"<tr><td>{label}</td>"
            html += f"<td>{metrics.get('precision', 0):.3f}</td>"
            html += f"<td>{metrics.get('recall', 0):.3f}</td>"
            html += f"<td>{metrics.get('f1-score', 0):.3f}</td>"
            html += f"<td>{metrics.get('support', 0)}</td></tr>"

    html += "</table>"

    # Confusion matrix
    if os.path.exists(cm_path):
        html += "<h2>📊 Confusion Matrix</h2>"
        html += f"<img src='{os.path.basename(cm_path)}'>"

    # F1 Plot
    if os.path.exists(f1_plot_path):
        html += "<h2>📉 Per-Class F1 Scores</h2>"
        html += f"<img src='{os.path.basename(f1_plot_path)}'>"

    html += "</body></html>"

    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(html)
