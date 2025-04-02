import matplotlib.pyplot as plt
import numpy as np

def create_plot(ax, x_data, y_data, label, color, line_style="None"):
    """
    Create a scatter plot on the given axis.
    
    Args:
        ax (matplotlib.axes.Axes): The axes to plot on
        x_data (list): X values
        y_data (list): Y values
        label (str): Label for the dataset
        color (str): Color for the dataset
        line_style (str): Style of line connecting points ("None", "Solid", "Dashed", "Dotted", "Dash-Dot")
        
    Returns:
        None
    """
    # Plot the scatter points
    ax.plot(x_data, y_data, 'o', label=label, color=color)
    
    # Add connecting lines if requested
    if line_style != "None" and len(x_data) > 1:
        line_style_map = {
            "Solid": '-',
            "Dashed": '--',
            "Dotted": ':',
            "Dash-Dot": '-.'
        }
        ax.plot(x_data, y_data, line_style_map[line_style], color=color, alpha=0.5)

def add_error_bars(ax, x_data, y_data, x_error, y_error, color):
    """
    Add error bars to the plot.
    
    Args:
        ax (matplotlib.axes.Axes): The axes to plot on
        x_data (list): X values
        y_data (list): Y values
        x_error (list): X error values
        y_error (list): Y error values
        color (str): Color for the error bars
        
    Returns:
        None
    """
    # Remove any existing markers
    ax.plot(x_data, y_data, 'none')
    
    # Add error bars
    ax.errorbar(
        x_data, y_data,
        xerr=x_error, yerr=y_error,
        fmt='o', color=color,
        ecolor=color, elinewidth=1, capsize=3
    )
