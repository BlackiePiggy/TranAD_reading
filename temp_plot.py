import numpy as np
import matplotlib.pyplot as plt
import os


def plot_multivariate_timeseries(file_path, figsize=(15, 10), time_axis=None,
                                 variable_names=None, title="Multivariate Time Series Visualization"):
    """
    Read .npy file and plot multivariate time series on multiple subplots

    Parameters:
    - file_path: path to .npy file
    - figsize: figure size (width, height)
    - time_axis: time axis data (optional, will use index if not provided)
    - variable_names: list of variable names (optional)
    - title: overall figure title
    """
    # Check if file exists
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File {file_path} does not exist")

    # Load .npy file
    try:
        data = np.load(file_path)
        print(f"Data shape: {data.shape}")
        print(f"Data type: {data.dtype}")
    except Exception as e:
        raise Exception(f"Failed to load file: {e}")

    # Ensure data is 2D
    if data.ndim == 1:
        data = data.reshape(-1, 1)
    elif data.ndim > 2:
        raise ValueError(f"Data has too many dimensions ({data.ndim}D), please provide 2D data")

    # Get number of variables
    n_variables = data.shape[1]
    n_timesteps = data.shape[0]

    # Create time axis
    if time_axis is None:
        time_axis = np.arange(n_timesteps)
    else:
        if len(time_axis) != n_timesteps:
            raise ValueError(f"Time axis length ({len(time_axis)}) does not match data length ({n_timesteps})")

    # Create variable names
    if variable_names is None:
        variable_names = [f"Variable {i + 1}" for i in range(n_variables)]
    else:
        if len(variable_names) != n_variables:
            raise ValueError(
                f"Number of variable names ({len(variable_names)}) does not match number of variables ({n_variables})")

    # Calculate subplot layout
    # Try to make the figure roughly square
    n_cols = int(np.ceil(np.sqrt(n_variables)))
    n_rows = int(np.ceil(n_variables / n_cols))

    # Create figure and subplots
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize, squeeze=False)
    fig.suptitle(title, fontsize=16, y=0.98)

    # Adjust subplot spacing
    plt.subplots_adjust(hspace=0.3, wspace=0.3)

    # Plot each variable
    for i in range(n_variables):
        row = i // n_cols
        col = i % n_cols
        ax = axes[row, col]

        # Plot time series
        ax.plot(time_axis, data[:, i], linewidth=1.5, color='blue')
        ax.set_title(variable_names[i], fontsize=11)
        ax.set_xlabel('Time')
        ax.set_ylabel('Value')
        ax.grid(True, alpha=0.3)

        # Rotate x-axis labels
        ax.tick_params(axis='x', rotation=45)

    # Hide extra subplots
    for i in range(n_variables, n_rows * n_cols):
        row = i // n_cols
        col = i % n_cols
        axes[row, col].set_visible(False)

    # Automatically adjust layout
    plt.tight_layout()
    plt.show()

    return fig, axes


def plot_with_statistics(file_path, figsize=(15, 10), show_mean=True, show_std=True):
    """
    Plot time series with statistical information
    """
    data = np.load(file_path)
    n_variables = data.shape[1]
    n_cols = int(np.ceil(np.sqrt(n_variables)))
    n_rows = int(np.ceil(n_variables / n_cols))

    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize, squeeze=False)
    fig.suptitle("Time Series with Statistics", fontsize=16)

    for i in range(n_variables):
        row = i // n_cols
        col = i % n_cols
        ax = axes[row, col]

        # Plot data
        ax.plot(data[:, i], 'b-', linewidth=1)

        # Add statistical lines
        if show_mean:
            mean_val = np.mean(data[:, i])
            ax.axhline(y=mean_val, color='r', linestyle='--', label=f'Mean: {mean_val:.2f}')

        if show_std:
            std_val = np.std(data[:, i])
            mean_val = np.mean(data[:, i])
            ax.axhline(y=mean_val + std_val, color='g', linestyle=':', alpha=0.7)
            ax.axhline(y=mean_val - std_val, color='g', linestyle=':', alpha=0.7)
            ax.fill_between(range(len(data)), mean_val - std_val, mean_val + std_val,
                            alpha=0.1, color='green')

        ax.set_title(f"Variable {i + 1}")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    # Hide extra subplots
    for i in range(n_variables, n_rows * n_cols):
        row = i // n_cols
        col = i % n_cols
        axes[row, col].set_visible(False)

    plt.tight_layout()
    plt.show()
    return fig, axes


if __name__ == "__main__":
    # Specify the file path to read
    file_path = 'processed/m11x11/data_test.npy'

    # Basic usage with default parameters
    plot_multivariate_timeseries(file_path, title="Test Dataset Time Series Analysis")

    # Alternative: use version with statistics
    # plot_with_statistics(file_path, show_mean=True, show_std=True)