import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage.filters import gaussian_filter1d

def thousands_formatter(x, pos):
    return f'{int(x/1000)}k'

def load_and_process_data(file_pattern, num_runs, sigma=1):
    data = []
    max_length = 0
    
    for i in range(1, num_runs + 1):
        filename = file_pattern.format(i)
        try:
            with open(filename, "r") as f:
                temp = np.genfromtxt(f, delimiter=",")
            data.append(temp)

            max_length = max(max_length, len(temp))
        except Exception as e:
            print(f"Error loading {filename}: {e}")
            continue
    
    if not data:
        raise ValueError("No data was loaded")
    
    padded_data = []
    for arr in data:
        if len(arr) < max_length:
            # Create padded array filled with NaN
            padded = np.full(max_length, np.nan)
            # Copy original data
            padded[:len(arr)] = arr
            padded_data.append(padded)
        else:
            padded_data.append(arr)
    
    
    data = np.array(padded_data)
    
    mean = np.nanmean(data, axis=0)
    std = np.nanstd(data, axis=0)
    smoothed_mean = gaussian_filter1d(mean, sigma=sigma)
    smoothed_std = gaussian_filter1d(std, sigma=sigma)
    
    return data, smoothed_mean, smoothed_std

def plot_training_curves(algorithms, title="Training Results", 
                        xlabel="Episodes", ylabel="Average Return",
                        figsize=(12, 6), x_scale=500, sigma=1):
    
    fig, ax = plt.subplots(figsize=figsize)
    
    for algo in algorithms:
        # Load and process data
        data, smoothed_mean, smoothed_std = load_and_process_data(
            algo['file_pattern'], 
            algo['num_runs'],
            sigma
        )
        
        x = np.arange(len(smoothed_mean)) * x_scale
        ax.plot(x, smoothed_mean, 
                color=algo['color'], 
                label=f"{algo['label']} (n={algo['num_runs']})")
        
        ax.fill_between(x, smoothed_mean - smoothed_std, smoothed_mean + smoothed_std, color=algo['color'], alpha=0.4)
    
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, linestyle='--', alpha=0.7)
    ax.legend()
    ax.xaxis.set_major_formatter(plt.FuncFormatter(thousands_formatter))
    
    plt.tight_layout()
    fig.savefig(f"graphs/Training/{title}.png")
    
    return fig, ax

algorithms_obs = [
    {
        'file_pattern': "ego_obs_avg_return/Egocentric_hObstacle_{}_evaluate_returns.csv",
        'label': 'Egocentric',
        'color': 'blue',
        'num_runs': 5
    },
    {
        'file_pattern': "allo_obs_avg_return/Allocentric_hObstacle_{}_evaluate_returns.csv",
        'label': 'Allocentric',
        'color': 'red',
        'num_runs': 5
    }
]
fig, ax = plot_training_curves(
    algorithms=algorithms_obs,
    title="Egocentric vs Allocentric Obstacle Average Return",
    ylabel="Average Return",
    x_scale=500,
    sigma=1  
)

success_algorithms_obs = [
    {
        'file_pattern': "ego_obs_success_rate/Egocentric_hObstacle_{}_evaluate_successes.csv",
        'label': 'Egocentric',
        'color': 'blue',
        'num_runs': 5
    },
    {
        'file_pattern': "allo_obs_success_rate/Allocentric_hObstacle_{}_evaluate_successes.csv",
        'label': 'Allocentric',
        'color': 'red',
        'num_runs': 5
    }
]
fig2, ax2 = plot_training_curves(
    algorithms=success_algorithms_obs,
    title="Egocentric vs Allocentric Obstacle Success Rate",
    ylabel="Success Rate",
    x_scale=500,
    sigma=1
)

algorithms_nav = [
    {
        'file_pattern': "ego_nav_avg_return/Egocentric_hNavi_{}_evaluate_returns.csv",
        'label': 'Egocentric',
        'color': 'blue',
        'num_runs': 5
    },
    {
        'file_pattern': "allo_nav_avg_return/Allocentric_hNavi_{}_evaluate_returns.csv",
        'label': 'Allocentric',
        'color': 'red',
        'num_runs': 4
    }
]

fig, ax = plot_training_curves(
    algorithms=algorithms_nav,
    title="Egocentric vs Allocentric Navigation Average Return",
    ylabel="Average Return",
    x_scale=500,
    sigma=1 
)

success_algorithms_nav = [
    {
        'file_pattern': "ego_nav_success_rate/Egocentric_hNavi_{}_evaluate_successes.csv",
        'label': 'Egocentric',
        'color': 'blue',
        'num_runs': 5
    },
    {
        'file_pattern': "allo_nav_success_rate/Allocentric_hNavi_{}_evaluate_successes.csv",
        'label': 'Allocentric',
        'color': 'red',
        'num_runs': 4
    }
]

fig2, ax2 = plot_training_curves(
    algorithms=success_algorithms_nav,
    title="Egocentric vs Allocentric Navigation Success Rate",
    ylabel="Success Rate",
    x_scale=500,
    sigma=1
)

algorithms_nav = [
    {
        'file_pattern': "ego_tas_avg_return/Egocentric_hTask_{}_evaluate_returns.csv",
        'label': 'Egocentric',
        'color': 'blue',
        'num_runs': 4
    },
    {
        'file_pattern': "allo_tas_avg_return/Allocentric_hTask_{}_evaluate_returns.csv",
        'label': 'Allocentric',
        'color': 'red',
        'num_runs': 4
    }
]

fig, ax = plot_training_curves(
    algorithms=algorithms_nav,
    title="Egocentric vs Allocentric Task Average Return",
    ylabel="Average Return",
    x_scale=500,
    sigma=1  
)

success_algorithms_nav = [
    {
        'file_pattern': "ego_tas_success_rate/Egocentric_hTask_{}_evaluate_successes.csv",
        'label': 'Egocentric',
        'color': 'blue',
        'num_runs': 4
    },
    {
        'file_pattern': "allo_tas_success_rate/Allocentric_hTask_{}_evaluate_successes.csv",
        'label': 'Allocentric',
        'color': 'red',
        'num_runs': 4
    }
]

fig2, ax2 = plot_training_curves(
    algorithms=success_algorithms_nav,
    title="Egocentric vs Allocentric Task Success Rate",
    ylabel="Success Rate",
    x_scale=500,
    sigma=1
)
plt.show()