""" Run this script in a dir where p1.log, p2.log, p3_att.log, sdpa.log are available.
"""

import json
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
from bisect import bisect_left
import numpy as np

# Function to read jsonl file
def read_jsonl(file_path):
    data = []
    with open(file_path, 'r') as f:
        for line in f:
            data.append(json.loads(line))
    return pd.DataFrame(data)

# Function to find closest loss value and corresponding FLOPs
def find_flops_for_loss(loss_values, flops_values, target_loss):
    idx = bisect_left(sorted(loss_values), target_loss)
    if idx >= len(loss_values):
        return flops_values[-1]
    if idx == 0:
        return flops_values[0]
    return flops_values[idx]

# Read data from each experiment
experiments = {
    'p1': {'path': 'p1.log/eval.jsonl', 'config': {'attention_kernel': 'power', 'degree': 1}},
    'p2': {'path': 'p2.log/eval.jsonl', 'config': {'attention_kernel': 'power', 'degree': 2}},
    'p3': {'path': 'p3_att.log/eval.jsonl', 'config': {'attention_kernel': 'power', 'degree': 3}},
    'sdpa': {'path': 'sdpa.log/eval.jsonl', 'config': {'attention_kernel': 'sdpa', 'degree': 1}}
}

flops_multiplier = {
    ('sdpa', 1): 3.54*3/2,
    ('power', 1): 1.21*3/2,
    ('power', 2): 1.57*3/2,
    ('power', 3): 3.54*3/2
}

# Load and process data
data = {}
for name, exp in experiments.items():
    df = read_jsonl(exp['path'])
    multiplier = flops_multiplier[(exp['config']['attention_kernel'], exp['config']['degree'])]
    df['theoretical_flops'] = df['iter'] * multiplier
    data[name] = df

# Set fixed target loss value
target_loss = 3.0  # You can adjust this value

# Create the subplot figure with adjusted vertical spacing
fig = make_subplots(
    rows=2, cols=2, 
    subplot_titles=(
        'Training Loss vs Theoretical FLOPs',
        f'Relative FLOPs at Target Loss: {target_loss:.2f}',
        'Training Loss vs Wall-clock Time',
        f'Relative Time at Target Loss: {target_loss:.2f}'
    ),
    column_widths=[0.6, 0.4],
    vertical_spacing=0.2,  # Slightly increased to make room for the text
    row_heights=[0.5, 0.5]
)

# First row: FLOPs plots (existing code)
colors = px.colors.qualitative.Set1
for i, (name, df) in enumerate(data.items()):
    df_sorted = df.sort_values('theoretical_flops')
    df_sorted = df[df['iter'] != 0]
    
    fig.add_trace(
        go.Scatter(
            x=df_sorted['theoretical_flops'],
            y=df_sorted['train.loss.avg'],
            name=name,
            line=dict(color=colors[i]),
            mode='markers+lines',
            connectgaps=False,
            hovertemplate='FLOPs: %{x}<br>Loss: %{y:.4f}<extra></extra>'
        ),
        row=1, col=1
    )

# Add horizontal line for target loss (first row)
fig.add_hline(y=target_loss, line_dash="dash", row=1, col=1)

# Second row: Wall-clock time plots
for i, (name, df) in enumerate(data.items()):
    df_sorted = df.sort_values('train_hours')
    df_sorted = df[df['iter'] != 0]
    
    fig.add_trace(
        go.Scatter(
            x=df_sorted['train_hours'],
            y=df_sorted['train.loss.avg'],
            name=name,
            line=dict(color=colors[i]),
            mode='markers+lines',
            connectgaps=False,
            showlegend=False,  # Don't show legend for second row (redundant)
            hovertemplate='Hours: %{x:.1f}<br>Loss: %{y:.4f}<extra></extra>'
        ),
        row=2, col=1
    )

# Add horizontal line for target loss (second row)
fig.add_hline(y=target_loss, line_dash="dash", row=2, col=1)

# First row: FLOPs ratio bar chart
sdpa_flops = find_flops_for_loss(
    data['sdpa']['train.loss.avg'].values,
    data['sdpa']['theoretical_flops'].values,
    target_loss
)

names = []
flops_ratios = []
time_ratios = []

# Calculate ratios for models
for name in ['p1', 'p2', 'p3']:
    # FLOPs ratio
    flops = find_flops_for_loss(
        data[name]['train.loss.avg'].values,
        data[name]['theoretical_flops'].values,
        target_loss
    )
    flops_ratios.append(flops / sdpa_flops)
    
    # Time ratio
    time = find_flops_for_loss(
        data[name]['train.loss.avg'].values,
        data[name]['train_hours'].values,
        target_loss
    )
    sdpa_time = find_flops_for_loss(
        data['sdpa']['train.loss.avg'].values,
        data['sdpa']['train_hours'].values,
        target_loss
    )
    time_ratios.append(time / sdpa_time)
    names.append(name)

# Add SDPA
names.append('sdpa')
flops_ratios.append(1.0)
time_ratios.append(1.0)

# Add FLOPs ratio bar chart
fig.add_trace(
    go.Bar(
        x=names, 
        y=flops_ratios,
        marker_color=[colors[i] for i in range(len(names))],
        name='Ratio'
    ),
    row=1, col=2
)

# Add Time ratio bar chart
fig.add_trace(
    go.Bar(
        x=names, 
        y=time_ratios,
        marker_color=[colors[i] for i in range(len(names))],
        name='Time Ratio',
        showlegend=False
    ),
    row=2, col=2
)

# Update layout
fig.update_layout(
    title_text=f'Training Loss Analysis (Target Loss: {target_loss:.2f})',
    showlegend=True,
    height=1400,
    width=1200,
    margin=dict(t=150, b=200, l=100, r=100),
    annotations=[
        *[ann for ann in fig.layout.annotations],  # Keep existing subplot titles
        # Text between plots
        dict(
            text="Models were trained on longcrawl64(Buckman, 2024) with batch size 1, context size 32768, 48 layers, 25 heads, hidden dimension 1600, on A100 GPUs.",
            xref="paper",
            yref="paper",
            x=0.5,
            y=0.5,  # Position between the two rows
            showarrow=False,
            font=dict(
                size=14,
                color="gray"
            ),
            align='center',
            yanchor='top'
        ),
        # Text at bottom
        dict(
            text="Models were trained on longcrawl64(Buckman, 2024) with batch size 1, context size 32768, 48 layers, 25 heads, hidden dimension 1600, on A100 GPUs.",
            xref="paper",
            yref="paper",
            x=0.5,
            y=-0.1,
            showarrow=False,
            font=dict(
                size=14,
                color="gray"
            ),
            align='center',
            yanchor='top'
        )
    ]
)

# Set scales and ranges
# First row (FLOPs)
fig.update_xaxes(type="log", row=1, col=1, title="Theoretical FLOPs")
fig.update_yaxes(type="log", row=1, col=1, range=[np.log10(1.5), np.log10(7)], title="Training Loss")
fig.update_xaxes(row=1, col=2, title="Model")
fig.update_yaxes(row=1, col=2, title="FLOPs Ratio vs SDPA (small is better)")

# Second row (Time)
fig.update_xaxes(type="log", row=2, col=1, title="Wall-clock Time (hours)")
fig.update_yaxes(type="log", row=2, col=1, range=[np.log10(1.5), np.log10(7)], title="Training Loss")
fig.update_xaxes(row=2, col=2, title="Model")
fig.update_yaxes(row=2, col=2, title="Time Ratio vs SDPA (small is better)")

# Save to HTML
fig.write_html("loss_analysis.html")