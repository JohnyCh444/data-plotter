import numpy as np

def generate_code(datasets, curve_fits, plot_params):
    """
    Generate Python code for the current plot configuration.
    
    Args:
        datasets (list): List of dataset dictionaries
        curve_fits (list): List of curve fit dictionaries
        plot_params (dict): Plot parameters
        
    Returns:
        str: Generated Python code
    """
    # Get figure size from plot params or use defaults
    fig_width = plot_params.get('fig_width', 10)
    fig_height = plot_params.get('fig_height', 6)
    
    code = """import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize
import math

# Create figure and axis
fig, ax = plt.subplots(figsize=({0}, {1}))

# Helper function to position annotations by percentage of plot area
def position_by_percent(ax, x_percent, y_percent):
    xmin, xmax = ax.get_xlim()
    ymin, ymax = ax.get_ylim()
    x_range = xmax - xmin
    y_range = ymax - ymin
    return xmin + (x_percent / 100.0) * x_range, ymin + (y_percent / 100.0) * y_range
""".format(fig_width, fig_height)

    # Add data for each dataset
    for i, dataset in enumerate(datasets):
        x_data_str = np.array2string(np.array(dataset['x']), separator=', ')
        y_data_str = np.array2string(np.array(dataset['y']), separator=', ')
        
        code += f"# Dataset {i+1}: {dataset['label']}\n"
        code += f"x{i} = np.array({x_data_str})\n"
        code += f"y{i} = np.array({y_data_str})\n"
        
        # Add error data if present
        if dataset['include_errors'] and dataset['x_error'] is not None:
            x_error_str = np.array2string(np.array(dataset['x_error']), separator=', ')
            code += f"x_error{i} = np.array({x_error_str})\n"
        
        if dataset['include_errors'] and dataset['y_error'] is not None:
            y_error_str = np.array2string(np.array(dataset['y_error']), separator=', ')
            code += f"y_error{i} = np.array({y_error_str})\n"
        
        # Plot the data
        if dataset['include_errors']:
            xerr = f"x_error{i}" if dataset['x_error'] is not None else "None"
            yerr = f"y_error{i}" if dataset['y_error'] is not None else "None"
            code += f"ax.errorbar(x{i}, y{i}, xerr={xerr}, yerr={yerr}, fmt='o', label='{dataset['label']}', color='{dataset['color']}')\n"
        else:
            code += f"ax.plot(x{i}, y{i}, 'o', label='{dataset['label']}', color='{dataset['color']}')\n"
        
        # Add code to highlight max and min points if requested
        if 'show_data_extremes' in plot_params and plot_params.get('show_data_extremes', False):
            code += f"""
# Highlight maximum and minimum points for dataset {i+1}
if len(y{i}) > 0:
    max_idx = np.argmax(y{i})
    min_idx = np.argmin(y{i})
    
    # Highlight max point with a star marker
    ax.plot(x{i}[max_idx], y{i}[max_idx], '*', color='{dataset['color']}', markersize=12, 
           label=f"{{'{dataset['label']} (Max: '}}{{y{i}[max_idx]:.4f}})")
    
    # Highlight min point with a pentagon marker
    ax.plot(x{i}[min_idx], y{i}[min_idx], 'P', color='{dataset['color']}', markersize=12, 
           label=f"{{'{dataset['label']} (Min: '}}{{y{i}[min_idx]:.4f}})")
"""
        
        code += "\n"

    # Add curve fitting code
    for i, curve_fit in enumerate(curve_fits):
        if curve_fit['fit_type'] != "None" and i < len(datasets):
            fit_type = curve_fit['fit_type']
            dataset_idx = curve_fit['dataset_index']
            
            code += f"# Curve fitting for {datasets[dataset_idx]['label']}\n"
            
            if fit_type == "Linear":
                code += generate_linear_fit_code(curve_fit, datasets, dataset_idx)
            
            elif fit_type == "Polynomial":
                code += generate_polynomial_fit_code(curve_fit, datasets, dataset_idx)
            
            elif fit_type == "Logarithmic":
                code += generate_logarithmic_fit_code(curve_fit, datasets, dataset_idx)
            
            elif fit_type == "Exponential":
                code += generate_exponential_fit_code(curve_fit, datasets, dataset_idx)

    # Add graph customization
    code += f"""# Customize the graph
ax.set_title('{plot_params['title']}')
ax.set_xlabel('{plot_params['x_label']}')
ax.set_ylabel('{plot_params['y_label']}')
"""

    # Add raw data derivatives if requested
    if 'show_raw_derivatives' in plot_params and plot_params['show_raw_derivatives']:
        raw_derivative_color = plot_params.get('raw_derivative_color', 'red')
        code += f"""
# Add raw data derivatives
raw_derivative_color = '{raw_derivative_color}'
for i, dataset in enumerate(datasets):
    x_data = dataset['x']
    y_data = dataset['y']
    
    # Need at least 2 points to calculate derivatives
    if len(x_data) >= 2:
        # Calculate finite differences for the raw data
        dx = np.diff(x_data)
        dy = np.diff(y_data)
        derivative_points = dy / dx
        
        # Use original x-coordinates (excluding last point) for plotting derivatives
        # This will shift derivatives to the earlier of the two points used to calculate them
        x_derivative = x_data[:-1]  # Exclude the last point
        
        # Plot the derivatives
        derivative_label = f"{{dataset['label']}} (Raw Derivative)"
        ax.plot(x_derivative, derivative_points, 'x', color=raw_derivative_color, 
               label=derivative_label, markersize=5)
        
        # Connect derivative points if there are more than 1
        if len(x_derivative) > 1:
            ax.plot(x_derivative, derivative_points, '--', color=raw_derivative_color, 
                   alpha=0.7, linewidth=1)
        
        # Highlight max and min derivative points if requested
        if 'show_derivative_extremes' in plot_params and plot_params.get('show_derivative_extremes', False) and len(derivative_points) > 0:
            # Find max and min derivative values
            max_deriv_idx = np.argmax(derivative_points)
            min_deriv_idx = np.argmin(derivative_points)
            
            # Highlight max derivative point
            ax.plot(x_derivative[max_deriv_idx], derivative_points[max_deriv_idx], '*', 
                   color=raw_derivative_color, markersize=12, 
                   label=f"{{dataset['label']}} (Max Deriv: {{derivative_points[max_deriv_idx]:.4f}})")
            
            # Highlight min derivative point
            ax.plot(x_derivative[min_deriv_idx], derivative_points[min_deriv_idx], 'P', 
                   color=raw_derivative_color, markersize=12, 
                   label=f"{{dataset['label']}} (Min Deriv: {{derivative_points[min_deriv_idx]:.4f}})")
"""

    if plot_params['show_grid']:
        code += "ax.grid(True, linestyle='--', alpha=0.7)\n"
    
    if 'show_legend' in plot_params and plot_params['show_legend']:
        code += f"ax.legend(loc='{plot_params['legend_position']}')\n\n"
    else:
        code += "# Hide the legend\nax.legend().set_visible(False)\n\n"
    
    # Add final display commands
    code += """# Make the plot look nice
fig.tight_layout()

# Show the plot
plt.show()

# Uncomment to save the figure
# plt.savefig('data_plot.png', dpi=300, bbox_inches='tight')
"""

    return code

def generate_linear_fit_code(curve_fit, datasets, dataset_idx):
    """Generate code for linear fitting"""
    
    fixed_point = curve_fit.get('fixed_point')
    show_in_legend = curve_fit.get('show_in_legend', False)
    
    if fixed_point:
        x0, y0 = fixed_point
        code = f"""# Linear fit: y = ax + b with fixed point ({x0}, {y0})
def linear_func(x, a, b):
    return a * x + b

# Fixed point constraint
x0, y0 = {x0}, {y0}

# Compute the weighted linear regression through the origin for the translated coordinates
x_centered = x{dataset_idx} - x0
y_centered = y{dataset_idx} - y0

if np.sum(x_centered**2) > 0:
    # Calculate slope using weighted linear regression through origin
    a = np.sum(x_centered * y_centered) / np.sum(x_centered**2)
    b = y0 - a * x0
    
    # Create smooth curve for plotting
    x_fit = np.linspace(min(x{dataset_idx}), max(x{dataset_idx}), 1000)
    y_fit = linear_func(x_fit, a, b)
    
    # Calculate R-squared
    residuals = y{dataset_idx} - linear_func(x{dataset_idx}, a, b)
    ss_res = np.sum(residuals**2)
    ss_tot = np.sum((y{dataset_idx} - np.mean(y{dataset_idx}))**2)
    r_squared = 1 - (ss_res / ss_tot)
    
    # Estimate the covariance matrix (simplified)
    variance = np.sum(residuals**2) / (len(residuals) - 1)
    covariance = np.array([[variance, 0], [0, variance]])
else:
    # Degenerate case
    a, b = 0, y0
    x_fit = np.linspace(min(x{dataset_idx}), max(x{dataset_idx}), 1000)
    y_fit = np.full_like(x_fit, y0)
    r_squared = 0
    covariance = np.zeros((2, 2))
"""
    else:
        code = f"""# Linear fit: y = ax + b
def linear_func(x, a, b):
    return a * x + b

# Perform the fit
params, covariance = optimize.curve_fit(linear_func, x{dataset_idx}, y{dataset_idx})
a, b = params

# Create smooth curve for plotting
x_fit = np.linspace(min(x{dataset_idx}), max(x{dataset_idx}), 1000)
y_fit = linear_func(x_fit, a, b)

# Calculate R-squared
residuals = y{dataset_idx} - linear_func(x{dataset_idx}, a, b)
ss_res = np.sum(residuals**2)
ss_tot = np.sum((y{dataset_idx} - np.mean(y{dataset_idx}))**2)
r_squared = 1 - (ss_res / ss_tot)
"""
    
    # Generate the label with or without equation
    if show_in_legend:
        if curve_fit.get('show_variance', False):
            code += f"""
# Plot the fitted curve with equation in label
a_err = np.sqrt(covariance[0, 0])
b_err = np.sqrt(covariance[1, 1])
"""
            if curve_fit.get('show_r_squared', False):
                code += f"""ax.plot(x_fit, y_fit, '--', color='{datasets[dataset_idx]['color']}', 
       label=f\"{{'{datasets[dataset_idx]['label']}'}} (y = ({{a:.4f}} ± {{a_err:.4f}})x + ({{b:.4f}} ± {{b_err:.4f}}), R²={{r_squared:.4f}})\")
"""
            else:
                code += f"""ax.plot(x_fit, y_fit, '--', color='{datasets[dataset_idx]['color']}', 
       label=f\"{{'{datasets[dataset_idx]['label']}'}} (y = ({{a:.4f}} ± {{a_err:.4f}})x + ({{b:.4f}} ± {{b_err:.4f}}))\")
"""
        else:
            if curve_fit.get('show_r_squared', False):
                code += f"""# Plot the fitted curve with equation in label
ax.plot(x_fit, y_fit, '--', color='{datasets[dataset_idx]['color']}', 
       label=f\"{{'{datasets[dataset_idx]['label']}'}} (y = {{a:.4f}}x + {{b:.4f}}, R²={{r_squared:.4f}})\")
"""
            else:
                code += f"""# Plot the fitted curve with equation in label
ax.plot(x_fit, y_fit, '--', color='{datasets[dataset_idx]['color']}', 
       label=f\"{{'{datasets[dataset_idx]['label']}'}} (y = {{a:.4f}}x + {{b:.4f}})\")
"""
    else:
        code += f"""
# Plot the fitted curve
ax.plot(x_fit, y_fit, '--', color='{datasets[dataset_idx]['color']}', label='{datasets[dataset_idx]['label']} (Linear Fit)')
"""
    if curve_fit.get('show_derivative', False):
        derivative_method = curve_fit.get('derivative_method', 'By Point')
        code += """# Calculate the derivative
derivative_color = 'darkred' if '{0}' == 'blue' else 'darkblue'
""".format(datasets[dataset_idx]['color'])

        if derivative_method == "By Point":
            code += """# Calculate derivative using the finite difference method (y1-y0)/(x1-x0)
# Calculate differences between consecutive points
dx = np.diff(x_fit)
dy = np.diff(y_fit)

# Calculate the derivative at each point
derivative_points = dy / dx

# For the last point, use the same derivative as the second-to-last point
y_derivative = np.zeros_like(x_fit)
y_derivative[:-1] = derivative_points  # Assign derivatives to all points except the last one
y_derivative[-1] = derivative_points[-1]  # The last point gets the same derivative as the second-to-last

# Plot the derivative using a dotted line
ax.plot(x_fit, y_derivative, ':', color=derivative_color, label='{0} (Derivative)')
""".format(datasets[dataset_idx]['label'])
        elif derivative_method == "By Equation":
            code += """# For linear function y = ax + b, the derivative is y' = a
# Calculate derivative using the analytical formula
y_derivative = np.full_like(x_fit, a)

# Plot the derivative using a solid line
ax.plot(x_fit, y_derivative, '-', color=derivative_color, label='{0} (Derivative)')
""".format(datasets[dataset_idx]['label'])
        else:  # Combined
            code += """# Combined method showing both analytical and finite difference approaches
# Analytical derivative: y' = a for y = ax + b
y_derivative_eq = np.full_like(x_fit, a)

# Finite difference method (y1-y0)/(x1-x0)
dx = np.diff(x_fit)
dy = np.diff(y_fit)
derivative_points = dy / dx
y_derivative_pt = np.zeros_like(x_fit)
y_derivative_pt[:-1] = derivative_points
y_derivative_pt[-1] = derivative_points[-1]

# Plot both derivatives
ax.plot(x_fit, y_derivative_pt, '.', color=derivative_color, alpha=0.5, markersize=3, label='{0} (Derivative Points)')
ax.plot(x_fit, y_derivative_eq, '-', color=derivative_color, alpha=0.8, linewidth=1, label='{0} (Derivative Curve)')

# Use the analytical derivative for annotation
y_derivative = y_derivative_eq
""".format(datasets[dataset_idx]['label'])

        code += """
# Add derivative equation annotation
ax.annotate(f"Derivative: y' = {a:.4f}", 
            xy=(x_fit[len(x_fit)//2], y_derivative[len(y_derivative)//2]),
            xytext=(10, -30), 
            textcoords="offset points",
            bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.8),
            fontsize=8,
            color=derivative_color)
"""

    # Add equation and R² annotations
    if curve_fit['show_equation'] or curve_fit['show_r_squared']:
        code += "# Add annotation\n"
        annotation = ""
        if curve_fit['show_equation']:
            annotation += "f'y = {a:.4f}x + {b:.4f}'"
        if curve_fit['show_r_squared']:
            if annotation:
                annotation += " + '\\nR² = {r_squared:.4f}'"
            else:
                annotation += "f'R² = {r_squared:.4f}'"
        
        # Position the annotation according to settings
        position = curve_fit.get('annotation_position', 'Auto')
        if position == "Custom (%)":
            x_percent = curve_fit.get('x_pos_percent', 50)
            y_percent = curve_fit.get('y_pos_percent', 50)
            code += f"# Position annotation at {x_percent}%, {y_percent}% of plot area\n"
            code += f"text_x, text_y = position_by_percent(ax, {x_percent}, {y_percent})\n"
            code += f"ax.annotate({annotation}, xy=(text_x, text_y), xytext=(0, 0),\n"
        else:
            code += f"ax.annotate({annotation}, xy=(x_fit[100], y_fit[100]), xytext=(10, 0),\n"
        code += "           textcoords='offset points', bbox=dict(boxstyle='round,pad=0.3', fc='white', alpha=0.8))\n\n"
    
    return code

def generate_polynomial_fit_code(curve_fit, datasets, dataset_idx):
    """Generate code for polynomial fitting"""
    
    degree = curve_fit['poly_degree']
    fixed_point = curve_fit.get('fixed_point')
    show_in_legend = curve_fit.get('show_in_legend', False)
    
    if fixed_point and degree >= 1:
        x0, y0 = fixed_point
        code = f"""# Polynomial fit of degree {degree} with fixed point ({x0}, {y0})
# First perform an unconstrained fit
coeffs_unconstrained = np.polyfit(x{dataset_idx}, y{dataset_idx}, {degree})

# Create polynomial function from coefficients
p_unconstrained = np.poly1d(coeffs_unconstrained)

# Calculate the value of the unconstrained polynomial at the fixed point
x0, y0 = {x0}, {y0}
y0_unconstrained = p_unconstrained(x0)

# Calculate the adjustment needed to make the curve pass through the fixed point
adjustment = y0 - y0_unconstrained

# Adjust the constant term (lowest coefficient) to make the curve pass through the fixed point
coeffs = coeffs_unconstrained.copy()
coeffs[-1] += adjustment

# Create the constrained polynomial function
p = np.poly1d(coeffs)

# Create smooth curve for plotting
x_fit = np.linspace(min(x{dataset_idx}), max(x{dataset_idx}), 1000)
y_fit = p(x_fit)

# Calculate R-squared
residuals = y{dataset_idx} - p(x{dataset_idx})
ss_res = np.sum(residuals**2)
ss_tot = np.sum((y{dataset_idx} - np.mean(y{dataset_idx}))**2)
r_squared = 1 - (ss_res / ss_tot)

# Estimate covariance matrix (simplified approach)
variance = np.sum(residuals**2) / (len(residuals) - {degree} - 1) if len(residuals) > {degree} + 1 else 1.0
cov = np.eye({degree} + 1) * variance

"""
    else:
        code = f"""# Polynomial fit of degree {degree}
# Perform the fit
coeffs, cov = np.polyfit(x{dataset_idx}, y{dataset_idx}, {degree}, cov=True)
p = np.poly1d(coeffs)

# Create smooth curve for plotting
x_fit = np.linspace(min(x{dataset_idx}), max(x{dataset_idx}), 1000)
y_fit = p(x_fit)

# Calculate R-squared
residuals = y{dataset_idx} - p(x{dataset_idx})
ss_res = np.sum(residuals**2)
ss_tot = np.sum((y{dataset_idx} - np.mean(y{dataset_idx}))**2)
r_squared = 1 - (ss_res / ss_tot)

"""

    # Generate the label with or without equation
    if show_in_legend:
        code += "# Format the polynomial equation for the legend\n"
        code += "eq_string = ''\n"
        code += "for i, coef in enumerate(coeffs):\n"
        code += "    if i == 0:\n"
        code += f"        eq_string += f'{{coef:.4f}}x^{{{degree}-i}}'\n"
        code += f"    elif i == {degree}:\n"
        code += "        if coef >= 0:\n"
        code += "            eq_string += f' + {coef:.4f}'\n"
        code += "        else:\n"
        code += "            eq_string += f' - {abs(coef):.4f}'\n"
        code += "    else:\n"
        code += "        if coef >= 0:\n"
        code += f"            eq_string += f' + {{coef:.4f}}x^{{{degree}-i}}'\n"
        code += "        else:\n"
        code += f"            eq_string += f' - {{abs(coef):.4f}}x^{{{degree}-i}}'\n"
        
        if curve_fit.get('show_r_squared', False):
            code += f"""
# Plot the fitted curve with equation in label
ax.plot(x_fit, y_fit, '--', color='{datasets[dataset_idx]['color']}', 
       label=f\"{{'{datasets[dataset_idx]['label']}'}} (y = {{eq_string}}, R²={{r_squared:.4f}})\")
"""
        else:
            code += f"""
# Plot the fitted curve with equation in label
ax.plot(x_fit, y_fit, '--', color='{datasets[dataset_idx]['color']}', 
       label=f\"{{'{datasets[dataset_idx]['label']}'}} (y = {{eq_string}})\")
"""
    else:
        code += f"""
# Plot the fitted curve
ax.plot(x_fit, y_fit, '--', color='{datasets[dataset_idx]['color']}', label='{datasets[dataset_idx]['label']} (Polynomial Fit)')
"""

    if curve_fit.get('show_derivative', False):
        derivative_method = curve_fit.get('derivative_method', 'By Point')
        code += """# Calculate the derivative
derivative_color = 'darkred' if '{0}' == 'blue' else 'darkblue'
""".format(datasets[dataset_idx]['color'])

        if derivative_method == "By Point":
            code += """# Calculate derivative using the finite difference method (y1-y0)/(x1-x0)
# Calculate differences between consecutive points
dx = np.diff(x_fit)
dy = np.diff(y_fit)

# Calculate the derivative at each point
derivative_points = dy / dx

# For the last point, use the same derivative as the second-to-last point
y_derivative = np.zeros_like(x_fit)
y_derivative[:-1] = derivative_points  # Assign derivatives to all points except the last one
y_derivative[-1] = derivative_points[-1]  # The last point gets the same derivative as the second-to-last

# Plot the derivative using a dotted line
ax.plot(x_fit, y_derivative, ':', color=derivative_color, label='{0} (Derivative)')
""".format(datasets[dataset_idx]['label'])
        elif derivative_method == "By Equation":
            code += """# For polynomial function, the derivative is computed using polyder
p_derivative = np.polyder(p)

# Evaluate the derivative over the x range
y_derivative = p_derivative(x_fit)

# Plot the derivative using a solid line
ax.plot(x_fit, y_derivative, '-', color=derivative_color, label='{0} (Derivative)')
""".format(datasets[dataset_idx]['label'])
        else:  # Combined
            code += """# Combined method showing both analytical and finite difference approaches
# Analytical derivative using polyder
p_derivative = np.polyder(p)
y_derivative_eq = p_derivative(x_fit)

# Finite difference method (y1-y0)/(x1-x0)
dx = np.diff(x_fit)
dy = np.diff(y_fit)
derivative_points = dy / dx
y_derivative_pt = np.zeros_like(x_fit)
y_derivative_pt[:-1] = derivative_points
y_derivative_pt[-1] = derivative_points[-1]

# Plot both derivatives
ax.plot(x_fit, y_derivative_pt, '.', color=derivative_color, alpha=0.5, markersize=3, label='{0} (Derivative Points)')
ax.plot(x_fit, y_derivative_eq, '-', color=derivative_color, alpha=0.8, linewidth=1, label='{0} (Derivative Curve)')

# Use the analytical derivative for annotation
y_derivative = y_derivative_eq
deriv_coeffs = p_derivative.coefficients
""".format(datasets[dataset_idx]['label'])

        code += """
# Format the derivative equation
deriv_eq = "y' = "
deriv_coeffs = p_derivative.coefficients
deriv_degree = len(deriv_coeffs) - 1

for i, coef in enumerate(deriv_coeffs):
    if i == 0:
        deriv_eq += f"{coef:.4f}x^{deriv_degree}"
    elif i == deriv_degree:
        deriv_eq += f" + {coef:.4f}"
    else:
        deriv_eq += f" + {coef:.4f}x^{deriv_degree-i}"

# Add derivative equation annotation
ax.annotate(f"Derivative: {deriv_eq}", 
            xy=(x_fit[len(x_fit)//2], y_derivative[len(y_derivative)//2]),
            xytext=(10, -30), 
            textcoords="offset points",
            bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.8),
            fontsize=8,
            color=derivative_color)
"""

    # Add equation and R² annotations
    if curve_fit['show_equation'] or curve_fit['show_r_squared']:
        code += "# Add annotation\n"
        code += "# Format the polynomial equation\n"
        code += "eq_string = 'y = '\n"
        code += "for i, coef in enumerate(coeffs):\n"
        code += "    if i == 0:\n"
        code += f"        eq_string += f'{{coef:.4f}}x^{{{degree}-i}}'\n"
        code += f"    elif i == {degree}:\n"
        code += "        eq_string += f' + {coef:.4f}'\n"
        code += "    else:\n"
        code += f"        eq_string += f' + {{coef:.4f}}x^{{{degree}-i}}'\n\n"
        
        annotation = ""
        if curve_fit['show_equation']:
            annotation += "eq_string"
        if curve_fit['show_r_squared']:
            if annotation:
                annotation += " + f'\\nR² = {r_squared:.4f}'"
            else:
                annotation += "f'R² = {r_squared:.4f}'"
        
        # Position the annotation according to settings
        position = curve_fit.get('annotation_position', 'Auto')
        if position == "Custom (%)":
            x_percent = curve_fit.get('x_pos_percent', 50)
            y_percent = curve_fit.get('y_pos_percent', 50)
            code += f"# Position annotation at {x_percent}%, {y_percent}% of plot area\n"
            code += f"text_x, text_y = position_by_percent(ax, {x_percent}, {y_percent})\n"
            code += f"ax.annotate({annotation}, xy=(text_x, text_y), xytext=(0, 0),\n"
        else:
            code += f"ax.annotate({annotation}, xy=(x_fit[100], y_fit[100]), xytext=(10, 0),\n"
        code += "           textcoords='offset points', bbox=dict(boxstyle='round,pad=0.3', fc='white', alpha=0.8))\n\n"
    
    return code

def generate_logarithmic_fit_code(curve_fit, datasets, dataset_idx):
    """Generate code for logarithmic fitting"""
    
    fixed_point = curve_fit.get('fixed_point')
    show_in_legend = curve_fit.get('show_in_legend', False)
    
    if fixed_point:
        x0, y0 = fixed_point
        code = f"""# Logarithmic fit: y = a * ln(x) + b with fixed point ({x0}, {y0})
def log_func(x, a, b):
    return a * np.log(x) + b

# Filter out non-positive x values for logarithmic fit
valid_indices = x{{dataset_idx}} > 0
x_valid = x{{dataset_idx}}[valid_indices]
y_valid = y{{dataset_idx}}[valid_indices]

# Fixed point constraint
x0, y0 = {x0}, {y0}

# Perform the fit with fixed point constraint
try:
    # Check if the fixed point has a positive x-value
    if x0 <= 0:
        raise ValueError("Fixed point must have positive x-value for logarithmic fit")
    
    # For logarithmic function with fixed point:
    # y0 = a*ln(x0) + b => b = y0 - a*ln(x0)
    # Substituting: y = a*ln(x) + (y0 - a*ln(x0)) = a*ln(x/x0) + y0
    
    # Compute transformed coordinates
    ln_ratio = np.log(x_valid/x0)
    y_shift = y_valid - y0
    
    if np.sum(ln_ratio**2) > 0:
        # Calculate optimal a using linear regression through origin
        a = np.sum(ln_ratio * y_shift) / np.sum(ln_ratio**2)
        b = y0 - a * np.log(x0)
        
        # Create smooth curve for plotting
        x_fit = np.linspace(min(x_valid), max(x_valid), 1000)
        y_fit = log_func(x_fit, a, b)
        
        # Calculate R-squared
        residuals = y_valid - log_func(x_valid, a, b)
        ss_res = np.sum(residuals**2)
        ss_tot = np.sum((y_valid - np.mean(y_valid))**2)
        r_squared = 1 - (ss_res / ss_tot)
        
        # Estimate covariance matrix (simplified approach)
        variance = np.sum(residuals**2) / (len(residuals) - 1)
        covariance = np.array([[variance, 0], [0, variance]])"""
    else:
        code = f"""# Logarithmic fit: y = a * ln(x) + b
def log_func(x, a, b):
    return a * np.log(x) + b

# Filter out non-positive x values for logarithmic fit
valid_indices = x{{dataset_idx}} > 0
x_valid = x{{dataset_idx}}[valid_indices]
y_valid = y{{dataset_idx}}[valid_indices]

# Perform the fit
try:
    params, covariance = optimize.curve_fit(log_func, x_valid, y_valid)
    a, b = params

    # Create smooth curve for plotting
    x_fit = np.linspace(min(x_valid), max(x_valid), 1000)
    y_fit = log_func(x_fit, a, b)

    # Calculate R-squared
    residuals = y_valid - log_func(x_valid, a, b)
    ss_res = np.sum(residuals**2)
    ss_tot = np.sum((y_valid - np.mean(y_valid))**2)
    r_squared = 1 - (ss_res / ss_tot)"""

    # Generate the label with or without equation
    if show_in_legend:
        if curve_fit.get('show_variance', False):
            code += """

    # Plot the fitted curve with equation in label
    a_err = np.sqrt(covariance[0, 0])
    b_err = np.sqrt(covariance[1, 1])"""
            if curve_fit.get('show_r_squared', False):
                code += f"""
    ax.plot(x_fit, y_fit, '--', color='{datasets[dataset_idx]['color']}', 
           label=f\"{{'{datasets[dataset_idx]['label']}'}} (y = ({{a:.4f}} ± {{a_err:.4f}})ln(x) + ({{b:.4f}} ± {{b_err:.4f}}), R²={{r_squared:.4f}})\")"""
            else:
                code += f"""
    ax.plot(x_fit, y_fit, '--', color='{datasets[dataset_idx]['color']}', 
           label=f\"{{'{datasets[dataset_idx]['label']}'}} (y = ({{a:.4f}} ± {{a_err:.4f}})ln(x) + ({{b:.4f}} ± {{b_err:.4f}}))\")"""
        else:
            if curve_fit.get('show_r_squared', False):
                code += f"""

    # Plot the fitted curve with equation in label
    ax.plot(x_fit, y_fit, '--', color='{datasets[dataset_idx]['color']}', 
           label=f\"{{'{datasets[dataset_idx]['label']}'}} (y = {{a:.4f}}ln(x) + {{b:.4f}}, R²={{r_squared:.4f}})\")"""
            else:
                code += f"""

    # Plot the fitted curve with equation in label
    ax.plot(x_fit, y_fit, '--', color='{datasets[dataset_idx]['color']}', 
           label=f\"{{'{datasets[dataset_idx]['label']}'}} (y = {{a:.4f}}ln(x) + {{b:.4f}})\")"""
    else:
        code += f"""

    # Plot the fitted curve
    ax.plot(x_fit, y_fit, '--', color='{datasets[dataset_idx]['color']}', label='{datasets[dataset_idx]['label']} (Log Fit)')
"""

    if curve_fit.get('show_derivative', False):
        derivative_method = curve_fit.get('derivative_method', 'By Point')
        code += """
    # Calculate the derivative
    derivative_color = 'darkred' if '{0}' == 'blue' else 'darkblue'
""".format(datasets[dataset_idx]['color'])

        if derivative_method == "By Point":
            code += """    # Calculate derivative using the finite difference method (y1-y0)/(x1-x0)
    # Calculate differences between consecutive points
    dx = np.diff(x_fit)
    dy = np.diff(y_fit)
    
    # Calculate the derivative at each point
    derivative_points = dy / dx
    
    # For the last point, use the same derivative as the second-to-last point
    y_derivative = np.zeros_like(x_fit)
    y_derivative[:-1] = derivative_points  # Assign derivatives to all points except the last one
    y_derivative[-1] = derivative_points[-1]  # The last point gets the same derivative as the second-to-last
    
    # Plot the derivative using a dotted line
    ax.plot(x_fit, y_derivative, ':', color=derivative_color, label='{0} (Derivative)')
""".format(datasets[dataset_idx]['label'])
        elif derivative_method == "By Equation":
            code += """    # For logarithmic function y = a*ln(x) + b, the derivative is y' = a/x
    # Calculate derivative using the analytical formula
    y_derivative = a / x_fit
    
    # Plot the derivative using a solid line
    ax.plot(x_fit, y_derivative, '-', color=derivative_color, label='{0} (Derivative)')
""".format(datasets[dataset_idx]['label'])
        else:  # Combined
            code += """    # Combined method showing both analytical and finite difference approaches
    # Analytical derivative: y' = a/x for y = a*ln(x) + b
    y_derivative_eq = a / x_fit
    
    # Finite difference method (y1-y0)/(x1-x0)
    dx = np.diff(x_fit)
    dy = np.diff(y_fit)
    derivative_points = dy / dx
    y_derivative_pt = np.zeros_like(x_fit)
    y_derivative_pt[:-1] = derivative_points
    y_derivative_pt[-1] = derivative_points[-1]
    
    # Plot both derivatives
    ax.plot(x_fit, y_derivative_pt, '.', color=derivative_color, alpha=0.5, markersize=3, label='{0} (Derivative Points)')
    ax.plot(x_fit, y_derivative_eq, '-', color=derivative_color, alpha=0.8, linewidth=1, label='{0} (Derivative Curve)')
    
    # Use the analytical derivative for annotation
    y_derivative = y_derivative_eq
""".format(datasets[dataset_idx]['label'])

        code += """
    # Add derivative equation annotation
    ax.annotate(f"Derivative: y' = {a:.4f}/x", 
                xy=(x_fit[len(x_fit)//2], y_derivative[len(y_derivative)//2]),
                xytext=(10, -30), 
                textcoords="offset points",
                bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.8),
                fontsize=8,
                color=derivative_color)
"""

    # Add equation and R² annotations
    if curve_fit['show_equation'] or curve_fit['show_r_squared']:
        code += "    # Add annotation\n"
        annotation = ""
        if curve_fit['show_equation']:
            annotation += "f'y = {a:.4f}ln(x) + {b:.4f}'"
        if curve_fit['show_r_squared']:
            if annotation:
                annotation += " + '\\nR² = {r_squared:.4f}'"
            else:
                annotation += "f'R² = {r_squared:.4f}'"
        
        # Position the annotation according to settings
        position = curve_fit.get('annotation_position', 'Auto')
        if position == "Custom (%)":
            x_percent = curve_fit.get('x_pos_percent', 50)
            y_percent = curve_fit.get('y_pos_percent', 50)
            code += f"    # Position annotation at {x_percent}%, {y_percent}% of plot area\n"
            code += f"    text_x, text_y = position_by_percent(ax, {x_percent}, {y_percent})\n"
            code += f"    ax.annotate({annotation}, xy=(text_x, text_y), xytext=(0, 0),\n"
        else:
            code += f"    ax.annotate({annotation}, xy=(x_fit[100], y_fit[100]), xytext=(10, 0),\n"
        code += "               textcoords='offset points', bbox=dict(boxstyle='round,pad=0.3', fc='white', alpha=0.8))\n"
    
    code += "except:\n"
    code += f"    print(f'Could not perform logarithmic fit for {{\"" + datasets[dataset_idx]['label'] + "\"}}.')\n\n"
    
    return code

def generate_exponential_fit_code(curve_fit, datasets, dataset_idx):
    """Generate code for exponential fitting"""
    
    fixed_point = curve_fit.get('fixed_point')
    show_in_legend = curve_fit.get('show_in_legend', False)
    
    if fixed_point:
        x0, y0 = fixed_point
        code = f"""# Exponential fit: y = a * exp(b * x) with fixed point ({x0}, {y0})
def exp_func(x, a, b):
    return a * np.exp(b * x)

# Fixed point constraint
x0, y0 = {x0}, {y0}

# Perform the fit with fixed point constraint
try:
    # Check if fixed point has positive y value (required for exponential)
    if y0 <= 0:
        raise ValueError("Fixed point must have positive y-value for exponential fit")
    
    # For exponential function with fixed point:
    # y0 = a*exp(b*x0) => a = y0/exp(b*x0)
    
    # We need to find optimal b, then calculate a from it
    def objective_fn(b_value):
        # Calculate 'a' based on fixed point constraint
        a_value = y0 / np.exp(b_value * x0)
        
        # Calculate predicted y values
        y_pred = a_value * np.exp(b_value * x{{dataset_idx}})
        
        # Return sum of squared errors
        residuals = y_pred - y{{dataset_idx}}
        return np.sum(residuals**2)
    
    # Use scipy's optimization to find best b value
    from scipy import optimize as scipy_optimize
    result = scipy_optimize.minimize_scalar(objective_fn)
    
    if result.success:
        b = result.x
        a = y0 / np.exp(b * x0)
        
        # Create smooth curve for plotting
        x_fit = np.linspace(min(x{{dataset_idx}}), max(x{{dataset_idx}}), 1000)
        y_fit = exp_func(x_fit, a, b)
        
        # Calculate R-squared
        residuals = y{{dataset_idx}} - exp_func(x{{dataset_idx}}, a, b)
        ss_res = np.sum(residuals**2)
        ss_tot = np.sum((y{{dataset_idx}} - np.mean(y{{dataset_idx}}))**2)
        r_squared = 1 - (ss_res / ss_tot)
        
        # Estimate covariance matrix (simplified approach)
        variance = np.sum(residuals**2) / (len(residuals) - 1)
        covariance = np.array([[variance, 0], [0, variance]])
    else:
        raise ValueError("Failed to optimize the exponential fit with fixed point")
"""
    else:
        code = f"""# Exponential fit: y = a * exp(b * x)
def exp_func(x, a, b):
    return a * np.exp(b * x)

# Perform the fit
try:
    # Use non-linear least squares to fit the exponential function
    # Need positive y values for initial guess
    params, covariance = optimize.curve_fit(exp_func, x{{dataset_idx}}, y{{dataset_idx}}, p0=(1.0, 0.1))
    a, b = params

    # Create smooth curve for plotting
    x_fit = np.linspace(min(x{{dataset_idx}}), max(x{{dataset_idx}}), 1000)
    y_fit = exp_func(x_fit, a, b)

    # Calculate R-squared
    residuals = y{{dataset_idx}} - exp_func(x{{dataset_idx}}, a, b)
    ss_res = np.sum(residuals**2)
    ss_tot = np.sum((y{{dataset_idx}} - np.mean(y{{dataset_idx}}))**2)
    r_squared = 1 - (ss_res / ss_tot)
"""

    # Generate the label with or without equation
    if show_in_legend:
        if curve_fit.get('show_variance', False):
            code += """
    # Plot the fitted curve with equation in label
    a_err = np.sqrt(covariance[0, 0])
    b_err = np.sqrt(covariance[1, 1])
"""
            if curve_fit.get('show_r_squared', False):
                code += f"""    ax.plot(x_fit, y_fit, '--', color='{datasets[dataset_idx]['color']}', 
           label=f\"{{'{datasets[dataset_idx]['label']}'}} (y = ({{a:.4f}} ± {{a_err:.4f}})e^(({{b:.4f}} ± {{b_err:.4f}})x), R²={{r_squared:.4f}})\")
"""
            else:
                code += f"""    ax.plot(x_fit, y_fit, '--', color='{datasets[dataset_idx]['color']}', 
           label=f\"{{'{datasets[dataset_idx]['label']}'}} (y = ({{a:.4f}} ± {{a_err:.4f}})e^(({{b:.4f}} ± {{b_err:.4f}})x))\")
"""
        else:
            if curve_fit.get('show_r_squared', False):
                code += f"""
    # Plot the fitted curve with equation in label
    ax.plot(x_fit, y_fit, '--', color='{datasets[dataset_idx]['color']}', 
           label=f\"{{'{datasets[dataset_idx]['label']}'}} (y = {{a:.4f}}e^({{b:.4f}}x), R²={{r_squared:.4f}})\")
"""
            else:
                code += f"""
    # Plot the fitted curve with equation in label
    ax.plot(x_fit, y_fit, '--', color='{datasets[dataset_idx]['color']}', 
           label=f\"{{'{datasets[dataset_idx]['label']}'}} (y = {{a:.4f}}e^({{b:.4f}}x))\")
"""
    else:
        code += f"""
    # Plot the fitted curve
    ax.plot(x_fit, y_fit, '--', color='{datasets[dataset_idx]['color']}', label='{datasets[dataset_idx]['label']} (Exp Fit)')
"""

    if curve_fit.get('show_derivative', False):
        derivative_method = curve_fit.get('derivative_method', 'By Point')
        code += """
    # Calculate the derivative
    derivative_color = 'darkred' if '{0}' == 'blue' else 'darkblue'
""".format(datasets[dataset_idx]['color'])

        if derivative_method == "By Point":
            code += """    # Calculate derivative using the finite difference method (y1-y0)/(x1-x0)
    # Calculate differences between consecutive points
    dx = np.diff(x_fit)
    dy = np.diff(y_fit)
    
    # Calculate the derivative at each point
    derivative_points = dy / dx
    
    # For the last point, use the same derivative as the second-to-last point
    y_derivative = np.zeros_like(x_fit)
    y_derivative[:-1] = derivative_points  # Assign derivatives to all points except the last one
    y_derivative[-1] = derivative_points[-1]  # The last point gets the same derivative as the second-to-last
    
    # Plot the derivative using a dotted line
    ax.plot(x_fit, y_derivative, ':', color=derivative_color, label='{0} (Derivative)')
""".format(datasets[dataset_idx]['label'])
        elif derivative_method == "By Equation":
            code += """    # For exponential function y = a*exp(b*x), the derivative is y' = a*b*exp(b*x)
    # Calculate derivative using the analytical formula
    y_derivative = a * b * np.exp(b * x_fit)
    
    # Plot the derivative using a solid line
    ax.plot(x_fit, y_derivative, '-', color=derivative_color, label='{0} (Derivative)')
""".format(datasets[dataset_idx]['label'])
        else:  # Combined
            code += """    # Combined method showing both analytical and finite difference approaches
    # Analytical derivative: y' = a*b*exp(b*x) for y = a*exp(b*x)
    y_derivative_eq = a * b * np.exp(b * x_fit)
    
    # Finite difference method (y1-y0)/(x1-x0)
    dx = np.diff(x_fit)
    dy = np.diff(y_fit)
    derivative_points = dy / dx
    y_derivative_pt = np.zeros_like(x_fit)
    y_derivative_pt[:-1] = derivative_points
    y_derivative_pt[-1] = derivative_points[-1]
    
    # Plot both derivatives
    ax.plot(x_fit, y_derivative_pt, '.', color=derivative_color, alpha=0.5, markersize=3, label='{0} (Derivative Points)')
    ax.plot(x_fit, y_derivative_eq, '-', color=derivative_color, alpha=0.8, linewidth=1, label='{0} (Derivative Curve)')
    
    # Use the analytical derivative for annotation
    y_derivative = y_derivative_eq
""".format(datasets[dataset_idx]['label'])

        code += """
    # Add derivative equation annotation
    ax.annotate(f"Derivative: y' = {a*b:.4f}e^({b:.4f}x)", 
                xy=(x_fit[len(x_fit)//2], y_derivative[len(y_derivative)//2]),
                xytext=(10, -30), 
                textcoords="offset points",
                bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.8),
                fontsize=8,
                color=derivative_color)
"""

    # Add equation and R² annotations
    if curve_fit['show_equation'] or curve_fit['show_r_squared']:
        code += "    # Add annotation\n"
        annotation = ""
        if curve_fit['show_equation']:
            annotation += "f'y = {a:.4f}e^({b:.4f}x)'"
        if curve_fit['show_r_squared']:
            if annotation:
                annotation += " + '\\nR² = {r_squared:.4f}'"
            else:
                annotation += "f'R² = {r_squared:.4f}'"
        
        # Position the annotation according to settings
        position = curve_fit.get('annotation_position', 'Auto')
        if position == "Custom (%)":
            x_percent = curve_fit.get('x_pos_percent', 50)
            y_percent = curve_fit.get('y_pos_percent', 50)
            code += f"    # Position annotation at {x_percent}%, {y_percent}% of plot area\n"
            code += f"    text_x, text_y = position_by_percent(ax, {x_percent}, {y_percent})\n"
            code += f"    ax.annotate({annotation}, xy=(text_x, text_y), xytext=(0, 0),\n"
        else:
            code += f"    ax.annotate({annotation}, xy=(x_fit[100], y_fit[100]), xytext=(10, 0),\n"
        code += "               textcoords='offset points', bbox=dict(boxstyle='round,pad=0.3', fc='white', alpha=0.8))\n"
    
    code += "except:\n"
    code += f"    print(f'Could not perform exponential fit for {{\"" + datasets[dataset_idx]['label'] + "\"}}.')\n\n"
    
    return code