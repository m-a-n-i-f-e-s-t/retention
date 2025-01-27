import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter, FuncFormatter, MaxNLocator
import numpy as np

def parse_csv(csv_string):
    # Split the input into lines and strip whitespace
    lines = csv_string.strip().split("\n")
    
    # Extract the column headers (ignore the first "ctx" column)
    headers = lines[0].split(",")[1:]
    
    # Initialize a dictionary to hold the lists
    data = {header: [] for header in headers}
    
    # Process each data row
    for line in lines[1:]:
        values = line.split(",")
        ctx = float(values[0])  # The first value is "ctx"
        for header, value in zip(headers, values[1:]):
            # Skip empty values
            if value.strip():
                data[header].append((ctx, float(value)))
    
    return data

TIME = parse_csv("""ctx,sdpa,p1_att,p2_att,p1_chunk,p2_chunk
1024,7.385202,17.743979,8.842133,13.195771,
2048,10.626697,31.515773,13.205021,13.079741,
4096,17.699785,58.845238,22.044227,13.017096,42.924817
8192,31.669766,113.638099,39.834294,12.999371,43.637823
16384,60.074586,224.629408,77.119189,13.023138,44.218218
32768,119.395400,449.015556,152.145039,13.052771,44.281001
65536,235.967426,897.338467,298.804286,13.118677,43.726352""")

FLOPS = parse_csv("""ctx,sdpa,p1_att,p2_att,p1_chunk,p2_chunk
1024,2.062e+11,2.094e+11,2.094e+11,1.065e+12,
2048,4.123e+11,4.188e+11,4.188e+11,1.065e+12,
4096,8.246e+11,8.375e+11,8.375e+11,1.065e+12,1.248e+12
8192,1.649e+12,1.675e+12,1.675e+12,1.065e+12,1.248e+12
16384,3.299e+12,3.350e+12,3.350e+12,1.065e+12,1.248e+12
32768,6.597e+12,6.700e+12,6.700e+12,1.065e+12,1.248e+12
65536,1.319e+13,1.340e+13,1.340e+13,1.065e+12,1.248e+12""")

def get_better(data, prefix):
    att_as_dict = {t[0]: t[1] for t in data[f'{prefix}_att']}   
    chunk_as_dict = {t[0]: t[1] for t in data[f'{prefix}_chunk']}
    return [(ctx, min(att_as_dict.get(ctx, float('inf')), chunk_as_dict.get(ctx, float('inf')))) 
            for ctx in att_as_dict]

def relative_improvement(data, prefix):
    better = get_better(data, prefix)
    return [(ctx, sdpa / better) for (ctx, sdpa), (_, better) in zip(data['sdpa'], better)]

if __name__ == "__main__":
    # Set style for a modern, professional look
    plt.style.use('seaborn-v0_8-whitegrid')
    plt.rcParams.update({
        'font.family': 'sans-serif',
        'font.sans-serif': ['Helvetica Neue', 'Arial', 'DejaVu Sans'],
        'font.size': 16,
        'axes.labelsize': 24,
        'axes.titlesize': 24,
        'axes.labelweight': 'bold',
        'grid.alpha': 0.3,
        'grid.color': '#cccccc',
        'xtick.labelsize': 20,
        'ytick.labelsize': 20,
        'axes.formatter.useoffset': False,
    })

    # Create figure
    fig, ax = plt.subplots(figsize=(12, 8), dpi=300)

    # Get data
    p1_speedup = relative_improvement(TIME, 'p1')
    p2_speedup = relative_improvement(TIME, 'p2')
    p1_theory = relative_improvement(FLOPS, 'p1')
    p2_theory = relative_improvement(FLOPS, 'p2')

    # Plot data
    x1, y1 = zip(*p1_speedup)
    x2, y2 = zip(*p2_speedup)
    x1t, y1t = zip(*p1_theory)
    x2t, y2t = zip(*p2_theory)

    # Plot with same colors as xl_performance_plot but dashed for measured
    ax.plot(x1t, y1t, '-', color='#89C2F5', linewidth=5.5, label='Degree=1 (theoretical)')
    ax.plot(x1, y1, '--', color='#89C2F5', linewidth=5.5, label='Degree=1')
    ax.plot(x2t, y2t, '-', color='#2F7AB9', linewidth=5.5, label='Degree=2 (theoretical)')
    ax.plot(x2, y2, '--', color='#2F7AB9', linewidth=5.5, label='Degree=2')

    # Set scales
    ax.set_xscale('log')

    # Customize axis labels with more padding
    ax.set_xlabel('Context length', labelpad=20)
    ax.set_ylabel('Power Attention Speedup (vs Flash)', labelpad=20)

    # Format y-axis labels to show integers with 'x' suffix
    ax.yaxis.set_major_formatter(lambda x, pos: f'{int(x)}x')

    # Format x-axis labels to use regular notation instead of scientific
    ax.xaxis.set_major_formatter(ScalarFormatter(useOffset=False))
    ax.xaxis.set_major_formatter(FuncFormatter(lambda x, p: format(int(x), ',')))

    # Set specific x-axis ticks at 5k, 10k, 50k
    ax.set_xticks([1000, 5000, 10000, 50000])
    ax.xaxis.set_ticklabels(['1k', '5k', '10k', '50k'])


    # Extra large legend
    ax.legend(frameon=True, 
             facecolor='white', 
             edgecolor='none', 
             framealpha=0.9,
             loc='upper left',
             bbox_to_anchor=(0.02, 0.98),
             fontsize=22,
             markerscale=3,
             handlelength=4,
             handletextpad=0.8)

    # Style adjustments
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color('#333333')
    ax.spines['bottom'].set_color('#333333')
    ax.spines['left'].set_linewidth(1.5)
    ax.spines['bottom'].set_linewidth(1.5)
    
    ax.grid(True, which='major', linestyle='-', alpha=0.2)
    ax.grid(True, which='minor', linestyle='-', alpha=0.1)

    plt.tight_layout(pad=1.5)

    # Save plot
    plt.savefig('kernel_speed_plot.png', 
                dpi=300, 
                bbox_inches='tight',
                facecolor='white',
                edgecolor='none')
    plt.close()

