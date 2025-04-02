import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import io
from utils.data_parser import parse_data, parse_error_data
from utils.curve_fitting import perform_curve_fitting, get_equation_text, calculate_derivative, get_derivative_equation_text
from utils.plotting import create_plot, add_error_bars

# Set page configuration
st.set_page_config(
    page_title="Data Plotting & Curve Fitting Tool",
    page_icon="📊",
    layout="wide"
)

# Title and introduction
st.title("Data Plotting & Curve Fitting Tool")
st.markdown(
    """
    This tool allows you to plot data points, add error bars, perform curve fitting, 
    and generate ready-to-use Python code for your visualizations. 
    """
)

# Sidebar for controlling datasets
with st.sidebar:
    st.header("Data Series")
    
    if 'dataset_count' not in st.session_state:
        st.session_state.dataset_count = 1
    
    # Button to add new dataset
    if st.button("Add Dataset"):
        st.session_state.dataset_count += 1
        st.rerun()
    
    # Button to remove dataset
    if st.session_state.dataset_count > 1:
        if st.button("Remove Last Dataset"):
            st.session_state.dataset_count -= 1
            st.rerun()

# Initialize containers for datasets
datasets = []
labels = []
colors = ['blue', 'red', 'green', 'purple', 'orange', 'brown', 'pink', 'gray', 'olive', 'cyan']

# Main content area with tabs
tab1, tab2 = st.tabs(["Data Entry & Plotting", "Generated Code"])

with tab1:
    col1, col2 = st.columns([2, 3])
    
    with col1:
        st.header("Data Input")
        
        # Generate input fields for each dataset
        for i in range(st.session_state.dataset_count):
            with st.expander(f"Dataset {i+1}", expanded=(i == 0)):
                st.markdown("### X/Y Data")
                st.write("""Enter your data with one number per line:
                ```
                1
                2
                3
                4
                5
                ```
                
                Other supported formats:
                - Space-separated: `1 2 3 4 5`
                - Comma-separated: `1.5, 2.0, 3.5, 4.0`
                - One pair per line: `1.5, 2.0\n3.5, 4.0`
                - Decimal formats with dots or commas: `1.5` or `1,5`
                """)
                
                x_data = st.text_area(f"X values for dataset {i+1}", 
                                     placeholder="e.g.,\n1\n2\n3\n4\n5", key=f"x_data_{i}")
                y_data = st.text_area(f"Y values for dataset {i+1}", 
                                     placeholder="e.g.,\n2.1\n4.2\n6.3\n8.4\n10.5", key=f"y_data_{i}")
                
                # Error bars
                include_errors = st.checkbox("Include error bars", key=f"include_errors_{i}")
                
                x_error = None
                y_error = None
                
                if include_errors:
                    x_error = st.text_area(f"X error values for dataset {i+1} (optional)", 
                                          placeholder="e.g.,\n0.1\n0.1\n0.1\n0.1\n0.1", key=f"x_error_{i}")
                    y_error = st.text_area(f"Y error values for dataset {i+1} (optional)", 
                                          placeholder="e.g.,\n0.2\n0.2\n0.2\n0.2\n0.2", key=f"y_error_{i}")
                
                # Dataset metadata
                label = st.text_input("Dataset Label", value=f"Dataset {i+1}", key=f"label_{i}")
                
                # Visualization options
                col_meta1, col_meta2 = st.columns(2)
                with col_meta1:
                    color = st.selectbox("Color", options=colors, index=i % len(colors), key=f"color_{i}")
                with col_meta2:
                    line_style = st.selectbox("Line Style", 
                                             options=["None", "Solid", "Dashed", "Dotted", "Dash-Dot"], 
                                             index=0, key=f"line_style_{i}")
                
                # Parse and store data
                try:
                    if x_data and y_data:
                        x_parsed = parse_data(x_data)
                        y_parsed = parse_data(y_data)
                        
                        # Ensure x and y have the same length
                        if len(x_parsed) != len(y_parsed):
                            st.error(f"Dataset {i+1}: X and Y must have the same number of values")
                            continue
                        
                        # Parse error data if provided
                        x_error_parsed = parse_error_data(x_error, len(x_parsed)) if x_error else None
                        y_error_parsed = parse_error_data(y_error, len(y_parsed)) if y_error else None
                        
                        datasets.append({
                            'x': x_parsed,
                            'y': y_parsed,
                            'x_error': x_error_parsed,
                            'y_error': y_error_parsed,
                            'label': label,
                            'color': color,
                            'line_style': line_style,
                            'include_errors': include_errors
                        })
                        labels.append(label)
                except Exception as e:
                    st.error(f"Error in dataset {i+1}: {str(e)}")
        
        # Graph customization
        st.header("Graph Customization")
        title = st.text_input("Graph Title", "Data Plot")
        x_label = st.text_input("X-Axis Label", "X")
        y_label = st.text_input("Y-Axis Label", "Y")
        
        # Figure size settings
        st.subheader("Figure Size")
        col_width, col_height = st.columns(2)
        with col_width:
            fig_width = st.slider("Width", min_value=4, max_value=20, value=10, step=1)
        with col_height:
            fig_height = st.slider("Height", min_value=3, max_value=15, value=6, step=1)
        
        # Raw data display options
        st.subheader("Raw Data Display Options")
        
        # Show max/min for raw data
        show_data_extremes = st.checkbox("Highlight Max & Min Data Points", False,
                                        help="Mark and display the maximum and minimum values in the dataset")
        
        # Raw derivative options
        show_raw_derivatives = st.checkbox("Show Derivatives of Raw Data", False, 
                                          help="Calculate and display derivatives directly from the input data points without curve fitting")
        if show_raw_derivatives:
            raw_derivative_color = st.selectbox("Raw Derivative Color", 
                                              ["red", "blue", "green", "purple", "orange", "black"], 
                                              index=0)
            show_derivative_extremes = st.checkbox("Highlight Max & Min Derivative Points", False,
                                                help="Mark and display the maximum and minimum derivative values")
        
        show_grid = st.checkbox("Show Grid", True)
        show_legend = st.checkbox("Show Legend", True)
        if show_legend:
            legend_position = st.selectbox("Legend Position", 
                                          ["best", "upper right", "upper left", "lower left", "lower right", 
                                           "right", "center left", "center right", "lower center", "upper center", "center"])
        else:
            legend_position = "best"  # Default value, won't be used if legend is hidden

        # Curve fitting options
        st.header("Curve Fitting")
        
        # List to store curve fit data for code generation
        curve_fits = []
        
        for i, label in enumerate(labels):
            with st.expander(f"Fit for {label}", expanded=(i == 0)):
                fit_type = st.selectbox("Regression Type", 
                                       ["None", "Linear", "Polynomial", "Logarithmic", "Exponential"], 
                                       key=f"fit_type_{i}")
                
                poly_degree = 2
                if fit_type == "Polynomial":
                    poly_degree = st.slider("Polynomial Degree", 2, 10, 2, key=f"poly_degree_{i}")
                
                # Fixed point option (for Linear and Polynomial)
                fixed_point = None
                if fit_type in ["Linear", "Polynomial"]:
                    use_fixed_point = st.checkbox("Force curve through specific point", False, key=f"use_fixed_point_{i}")
                    if use_fixed_point:
                        fixed_point_col1, fixed_point_col2 = st.columns(2)
                        with fixed_point_col1:
                            fixed_point_x = st.number_input("X coordinate", value=0.0, key=f"fixed_point_x_{i}")
                        with fixed_point_col2:
                            fixed_point_y = st.number_input("Y coordinate", value=0.0, key=f"fixed_point_y_{i}")
                        fixed_point = (fixed_point_x, fixed_point_y)
                
                # Annotation options
                col_anno1, col_anno2 = st.columns(2)
                with col_anno1:
                    show_equation = st.checkbox("Show Equation", True, key=f"show_equation_{i}")
                    show_variance = st.checkbox("Show Parameter Variance", False, key=f"show_variance_{i}")
                    show_derivative = st.checkbox("Show Derivative", False, key=f"show_derivative_{i}")
                    
                    # Only show derivative options if derivative is selected
                    if show_derivative:
                        derivative_method = st.radio(
                            "Derivative Calculation Method",
                            ["By Point", "By Equation", "Combined"],
                            key=f"deriv_method_{i}",
                            horizontal=True
                        )
                        derivative_position_options = ["Auto", "Upper Right", "Upper Left", "Lower Left", "Lower Right", 
                                               "Center", "Top Center", "Bottom Center", "Left Center", "Right Center"]
                        derivative_position = st.selectbox("Derivative Annotation Position", 
                                                    derivative_position_options,
                                                    key=f"derivative_pos_{i}")
                with col_anno2:
                    show_r_squared = st.checkbox("Show R² Value", True, key=f"show_r_squared_{i}")
                    show_in_legend = st.checkbox("Show Equation in Legend", False, key=f"show_in_legend_{i}", 
                                              help="Add equation to legend label instead of as annotation")
                    position_options = ["Auto", "Upper Right", "Upper Left", "Lower Left", "Lower Right", "Center", "Top Center", "Bottom Center", "Left Center", "Right Center"]
                    annotation_position = st.selectbox("Annotation Position", 
                                                     position_options,
                                                     key=f"annotation_pos_{i}")
                    
                    # No custom percentages anymore
                
                # Build the curve fits dictionary
                curve_fit_entry = {
                    'dataset_index': i,
                    'fit_type': fit_type,
                    'poly_degree': poly_degree,
                    'fixed_point': fixed_point,
                    'show_equation': show_equation,
                    'show_r_squared': show_r_squared,
                    'show_in_legend': show_in_legend,
                    'show_variance': show_variance,
                    'show_derivative': show_derivative,
                    'annotation_position': annotation_position
                }
                
                # Add derivative settings if enabled
                if show_derivative:
                    curve_fit_entry['derivative_method'] = derivative_method
                    curve_fit_entry['derivative_position'] = derivative_position
                
                curve_fits.append(curve_fit_entry)

    with col2:
        # Only proceed if we have valid datasets
        if datasets:
            st.header("Plot")
            
            # Create the plot with user-defined figure size
            fig, ax = plt.subplots(figsize=(fig_width, fig_height))
            
            # Plot each dataset
            for i, dataset in enumerate(datasets):
                create_plot(ax, dataset['x'], dataset['y'], dataset['label'], dataset['color'], dataset['line_style'])
                
                # Add error bars if included
                if dataset['include_errors']:
                    add_error_bars(ax, dataset['x'], dataset['y'], 
                                  dataset['x_error'], dataset['y_error'], 
                                  dataset['color'])
                
                # Highlight max and min points if requested
                if 'show_data_extremes' in locals() and show_data_extremes:
                    x_data = dataset['x']
                    y_data = dataset['y']
                    
                    if len(y_data) > 0:
                        # Find max and min values
                        max_idx = np.argmax(y_data)
                        min_idx = np.argmin(y_data)
                        
                        # Highlight max point with a star marker
                        ax.plot(x_data[max_idx], y_data[max_idx], '*', 
                               color=dataset['color'], markersize=12, 
                               label=f"{dataset['label']} (Max: {y_data[max_idx]:.4f})")
                        
                        # Highlight min point with a star marker
                        ax.plot(x_data[min_idx], y_data[min_idx], 'P', 
                               color=dataset['color'], markersize=12, 
                               label=f"{dataset['label']} (Min: {y_data[min_idx]:.4f})")
            
            # Apply curve fitting
            for i, curve_fit in enumerate(curve_fits):
                if curve_fit['fit_type'] != "None" and i < len(datasets):
                    dataset = datasets[i]
                    x_data = dataset['x']
                    y_data = dataset['y']
                    
                    # Perform the curve fitting
                    x_fit, y_fit, equation, r_squared, covariance = perform_curve_fitting(
                        x_data, y_data, 
                        curve_fit['fit_type'], 
                        curve_fit['poly_degree'],
                        curve_fit.get('fixed_point')
                    )
                    
                    # Create the label for the fitted curve
                    if curve_fit.get('show_in_legend', False) and curve_fit['fit_type'] != "None":
                        # Create a label with the equation
                        equation_text = get_equation_text(
                            curve_fit['fit_type'], 
                            equation, 
                            covariance, 
                            curve_fit['show_variance']
                        )
                        r_squared_text = f" (R²={r_squared:.4f})" if curve_fit.get('show_r_squared', False) else ""
                        label = f"{dataset['label']} {equation_text}{r_squared_text}"
                    else:
                        # Standard label
                        label = f"{dataset['label']} ({curve_fit['fit_type']} Fit)"
                    
                    # Plot the fitted curve
                    ax.plot(x_fit, y_fit, '--', color=dataset['color'], label=label)
                    
                    # Plot the derivative if requested
                    if curve_fit.get('show_derivative', False):
                        # Get the derivative calculation method
                        derivative_method = curve_fit.get('derivative_method', 'By Point')
                        
                        # Calculate the derivative
                        if derivative_method == "By Point":
                            # Use original data points for "By Point" method
                            y_derivative, derivative_params = calculate_derivative(
                                x_fit, equation, curve_fit['fit_type'], curve_fit['poly_degree'],
                                method=derivative_method, x_data=x_data, y_data=y_data
                            )
                        else:
                            # Use analytical method for "By Equation" and "Combined"
                            y_derivative, derivative_params = calculate_derivative(
                                x_fit, equation, curve_fit['fit_type'], curve_fit['poly_degree'],
                                method=derivative_method
                            )
                        
                        derivative_color = 'darkred' if dataset['color'] == 'blue' else 'darkblue'
                        
                        # Different methods of derivative visualization
                        if derivative_method == "By Point":
                            # Just plot the calculated derivative points
                            ax.plot(x_fit, y_derivative, ':', color=derivative_color, 
                                   label=f"{dataset['label']} (Derivative)")
                        elif derivative_method == "By Equation":
                            # Plot a smooth curve using the derivative equation
                            ax.plot(x_fit, y_derivative, '-', color=derivative_color, 
                                   label=f"{dataset['label']} (Derivative)")
                        else:  # Combined method
                            # For combined method, calculate both derivatives and plot both
                            
                            # First, get the derivative by original data points
                            y_derivative_points, _ = calculate_derivative(
                                x_fit, equation, curve_fit['fit_type'], curve_fit['poly_degree'],
                                method="By Point", x_data=x_data, y_data=y_data
                            )
                            
                            # Then get the analytical derivative
                            y_derivative_equation, _ = calculate_derivative(
                                x_fit, equation, curve_fit['fit_type'], curve_fit['poly_degree'],
                                method="By Equation"
                            )
                            
                            # Plot both derivatives
                            ax.plot(x_fit, y_derivative_points, '.', color=derivative_color, alpha=0.5,
                                   markersize=3, label=f"{dataset['label']} (Derivative Points)")
                            ax.plot(x_fit, y_derivative_equation, '-', color=derivative_color, alpha=0.8,
                                   linewidth=1, label=f"{dataset['label']} (Derivative Curve)")
                        
                        # Add derivative equation
                        derivative_equation = get_derivative_equation_text(
                            curve_fit['fit_type'], equation
                        )
                        
                        # Add annotation for derivative based on user preference
                        if curve_fit.get('show_equation', False):
                            # Determine annotation position
                            deriv_position = curve_fit.get('derivative_position', 'Auto')
                            if deriv_position == "Auto":
                                # Position in the middle of the derivative curve
                                deriv_y_index = len(y_derivative) // 2
                                text_x = x_fit[deriv_y_index] 
                                text_y = y_derivative[deriv_y_index]
                                xytext = (10, -30)
                            else:
                                # Fixed positions based on selection (same as for main equation)
                                xmin, xmax = ax.get_xlim()
                                ymin, ymax = ax.get_ylim()
                                x_range = xmax - xmin
                                y_range = ymax - ymin
                                
                                if deriv_position == "Upper Right":
                                    text_x = xmin + 0.75 * x_range
                                    text_y = ymin + 0.65 * y_range  # Offset from main equation
                                elif deriv_position == "Upper Left":
                                    text_x = xmin + 0.25 * x_range
                                    text_y = ymin + 0.65 * y_range
                                elif deriv_position == "Lower Left":
                                    text_x = xmin + 0.25 * x_range
                                    text_y = ymin + 0.15 * y_range
                                elif deriv_position == "Lower Right":
                                    text_x = xmin + 0.75 * x_range
                                    text_y = ymin + 0.15 * y_range
                                elif deriv_position == "Center":
                                    text_x = xmin + 0.5 * x_range
                                    text_y = ymin + 0.4 * y_range
                                elif deriv_position == "Top Center":
                                    text_x = xmin + 0.5 * x_range
                                    text_y = ymin + 0.8 * y_range
                                elif deriv_position == "Bottom Center":
                                    text_x = xmin + 0.5 * x_range
                                    text_y = ymin + 0.2 * y_range
                                elif deriv_position == "Left Center":
                                    text_x = xmin + 0.2 * x_range
                                    text_y = ymin + 0.5 * y_range
                                elif deriv_position == "Right Center":
                                    text_x = xmin + 0.8 * x_range
                                    text_y = ymin + 0.5 * y_range
                                
                                xytext = (0, 0)
                            
                            ax.annotate(
                                f"Derivative: {derivative_equation}", 
                                xy=(text_x, text_y),
                                xytext=xytext, 
                                textcoords="offset points",
                                bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.8),
                                fontsize=8,
                                color=derivative_color
                            )
                    
                    # Add equation and R² to the graph as annotations (if not shown in legend)
                    if (curve_fit['show_equation'] or curve_fit['show_r_squared']) and not curve_fit.get('show_in_legend', False):
                        annotation_text = ""
                        
                        if curve_fit['show_equation']:
                            annotation_text += get_equation_text(
                                curve_fit['fit_type'], 
                                equation, 
                                covariance, 
                                curve_fit['show_variance']
                            )
                        
                        if curve_fit['show_r_squared']:
                            if annotation_text:
                                annotation_text += f"\nR² = {r_squared:.4f}"
                            else:
                                annotation_text += f"R² = {r_squared:.4f}"
                        
                        # Determine annotation position
                        position = curve_fit['annotation_position']
                        if position == "Auto":
                            # Position near the start of the curve
                            text_x = x_fit[min(5, len(x_fit)-1)]
                            text_y = y_fit[min(5, len(y_fit)-1)]
                            xytext = (10, 0)
                        else:
                            # Fixed positions based on selection
                            xmin, xmax = ax.get_xlim()
                            ymin, ymax = ax.get_ylim()
                            x_range = xmax - xmin
                            y_range = ymax - ymin
                            
                            if position == "Upper Right":
                                text_x = xmin + 0.75 * x_range
                                text_y = ymin + 0.75 * y_range
                            elif position == "Upper Left":
                                text_x = xmin + 0.25 * x_range
                                text_y = ymin + 0.75 * y_range
                            elif position == "Lower Left":
                                text_x = xmin + 0.25 * x_range
                                text_y = ymin + 0.25 * y_range
                            elif position == "Lower Right":
                                text_x = xmin + 0.75 * x_range
                                text_y = ymin + 0.25 * y_range
                            elif position == "Center":
                                text_x = xmin + 0.5 * x_range
                                text_y = ymin + 0.5 * y_range
                            elif position == "Top Center":
                                text_x = xmin + 0.5 * x_range
                                text_y = ymin + 0.9 * y_range
                            elif position == "Bottom Center":
                                text_x = xmin + 0.5 * x_range
                                text_y = ymin + 0.1 * y_range
                            elif position == "Left Center":
                                text_x = xmin + 0.1 * x_range
                                text_y = ymin + 0.5 * y_range
                            elif position == "Right Center":
                                text_x = xmin + 0.9 * x_range
                                text_y = ymin + 0.5 * y_range
                            
                            xytext = (0, 0)
                        
                        ax.annotate(annotation_text, xy=(text_x, text_y), 
                                   xytext=xytext, textcoords="offset points",
                                   bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.8),
                                   fontsize=8)
            
            # Add raw data derivatives if requested
            if 'show_raw_derivatives' in locals() and show_raw_derivatives:
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
                        derivative_label = f"{dataset['label']} (Raw Derivative)"
                        ax.plot(x_derivative, derivative_points, 'x', color=raw_derivative_color, 
                               label=derivative_label, markersize=5)
                        
                        # Connect derivative points if there are more than 1
                        if len(x_derivative) > 1:
                            ax.plot(x_derivative, derivative_points, '--', color=raw_derivative_color, 
                                   alpha=0.7, linewidth=1)
                        
                        # Highlight max and min derivative points if requested
                        if 'show_derivative_extremes' in locals() and show_derivative_extremes and len(derivative_points) > 0:
                            # Find max and min derivative values
                            max_deriv_idx = np.argmax(derivative_points)
                            min_deriv_idx = np.argmin(derivative_points)
                            
                            # Highlight max derivative point
                            ax.plot(x_derivative[max_deriv_idx], derivative_points[max_deriv_idx], '*', 
                                   color=raw_derivative_color, markersize=12, 
                                   label=f"{dataset['label']} (Max Deriv: {derivative_points[max_deriv_idx]:.4f})")
                            
                            # Highlight min derivative point
                            ax.plot(x_derivative[min_deriv_idx], derivative_points[min_deriv_idx], 'P', 
                                   color=raw_derivative_color, markersize=12, 
                                   label=f"{dataset['label']} (Min Deriv: {derivative_points[min_deriv_idx]:.4f})")
            
            # Customize the graph
            ax.set_title(title)
            ax.set_xlabel(x_label)
            ax.set_ylabel(y_label)
            
            if show_grid:
                ax.grid(True, linestyle='--', alpha=0.7)
            
            if show_legend:
                ax.legend(loc=legend_position)
            else:
                ax.legend().set_visible(False)
            
            # Make the plot look nice
            fig.tight_layout()
            
            # Display the plot
            st.pyplot(fig)
            
            # Display data statistics
            st.subheader("Data Statistics")
            
            # Create expandable sections for each dataset
            for i, dataset in enumerate(datasets):
                with st.expander(f"Statistics for {dataset['label']}"):
                    x_data = dataset['x']
                    y_data = dataset['y']
                    
                    if len(y_data) > 0:
                        # Find max and min values
                        max_idx = np.argmax(y_data)
                        min_idx = np.argmin(y_data)
                        
                        # Display data point statistics
                        st.markdown("#### Data Point Statistics")
                        st.markdown(f"**Maximum value:** {y_data[max_idx]:.4f} (at x-position: {x_data[max_idx]:.4f})")
                        st.markdown(f"**Minimum value:** {y_data[min_idx]:.4f} (at x-position: {x_data[min_idx]:.4f})")
                        
                        # Calculate and display derivative statistics if applicable
                        if 'show_raw_derivatives' in locals() and show_raw_derivatives and len(x_data) >= 2:
                            # Calculate derivatives
                            dx = np.diff(x_data)
                            dy = np.diff(y_data)
                            derivative_points = dy / dx
                            x_derivative = x_data[:-1]  # Use original x-coordinates (excluding last point)
                            
                            if len(derivative_points) > 0:
                                # Find max and min derivative values
                                max_deriv_idx = np.argmax(derivative_points)
                                min_deriv_idx = np.argmin(derivative_points)
                                
                                # Display derivative statistics
                                st.markdown("#### Derivative Statistics")
                                st.markdown(f"**Maximum derivative:** {derivative_points[max_deriv_idx]:.4f} (at x-position: {x_derivative[max_deriv_idx]:.4f})")
                                st.markdown(f"**Minimum derivative:** {derivative_points[min_deriv_idx]:.4f} (at x-position: {x_derivative[min_deriv_idx]:.4f})")
            
            # Save the figure to a bytes buffer for download
            buf = io.BytesIO()
            fig.savefig(buf, format='png', dpi=300)
            buf.seek(0)
            
            # Download button for the plot
            st.download_button(
                label="Download Plot as PNG",
                data=buf,
                file_name="data_plot.png",
                mime="image/png"
            )
            
            # Store the figure and data for code generation
            st.session_state.figure = fig
            st.session_state.datasets = datasets
            st.session_state.curve_fits = curve_fits
            st.session_state.plot_params = {
                'title': title,
                'x_label': x_label,
                'y_label': y_label,
                'fig_width': fig_width,
                'fig_height': fig_height,
                'show_grid': show_grid,
                'show_legend': show_legend,
                'legend_position': legend_position,
                'show_raw_derivatives': show_raw_derivatives if 'show_raw_derivatives' in locals() else False,
                'show_data_extremes': show_data_extremes if 'show_data_extremes' in locals() else False,
                'show_derivative_extremes': show_derivative_extremes if 'show_derivative_extremes' in locals() else False
            }
            
            # Add raw derivative color if applicable
            if 'show_raw_derivatives' in locals() and show_raw_derivatives and 'raw_derivative_color' in locals():
                st.session_state.plot_params['raw_derivative_color'] = raw_derivative_color
        else:
            st.info("Enter your data in the fields on the left to generate a plot.")

with tab2:
    if datasets:
        st.header("Generated Python Code")
        st.write("Below is the Python code to reproduce this plot using matplotlib and scipy:")
        
        # Import the code generator
        from utils.code_generator import generate_code
        
        # Generate the code
        code = generate_code(datasets, curve_fits, st.session_state.plot_params)
        
        # Display the code with syntax highlighting
        st.code(code, language="python")
        
        # Provide download button for the code
        st.download_button(
            label="Download Python Code",
            data=code,
            file_name="plot_code.py",
            mime="text/plain"
        )
    else:
        st.info("Enter your data in the Data Entry tab to generate code.")
