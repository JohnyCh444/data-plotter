import numpy as np
from scipy import optimize, interpolate

def calculate_derivative(x_fit, params, fit_type, poly_degree=2, method="By Equation", x_data=None, y_data=None):
    """
    Calculate the derivative of the fitted curve.
    
    Args:
        x_fit (array): X values for the fitted curve
        params: Equation parameters
        fit_type (str): Type of curve fitting
        poly_degree (int): Degree of polynomial for polynomial fitting
        method (str): Method for calculating derivatives - "By Equation" (analytical) or "By Point" (finite difference)
        x_data (array, optional): Original x data points for "By Point" method
        y_data (array, optional): Original y data points for "By Point" method
        
    Returns:
        tuple: (y_derivative, derivative_params)
    """
    # First calculate the y values for the fitted curve
    y_fit = None
    
    if fit_type == "Linear":
        # Linear: y = ax + b
        a, b = params
        y_fit = a * x_fit + b
    elif fit_type == "Polynomial":
        # Polynomial: y = coeffs[0]x^n + ... + coeffs[n]
        coeffs = params
        p = np.poly1d(coeffs)
        y_fit = p(x_fit)
    elif fit_type == "Logarithmic":
        # Logarithmic: y = a*ln(x) + b
        a, b = params
        y_fit = a * np.log(x_fit) + b
    elif fit_type == "Exponential":
        # Exponential: y = a*exp(b*x)
        a, b = params
        y_fit = a * np.exp(b * x_fit)
    
    # Calculate derivative based on the specified method
    if method == "By Point" and x_data is not None and y_data is not None:
        # Use the original data points for finite difference calculation
        # Sort data by x values to ensure correct order for derivative calculation
        if len(x_data) > 1:
            sort_indices = np.argsort(x_data)
            x_sorted = np.array(x_data)[sort_indices]
            y_sorted = np.array(y_data)[sort_indices]
        
            # Calculate the derivatives at each original data point using finite differences
            dx = np.diff(x_sorted)
            dy = np.diff(y_sorted)
            
            # Calculate derivatives: (y[i+1] - y[i]) / (x[i+1] - x[i])
            data_derivatives = dy / dx
            
            # For the last point, use the same derivative as the previous point
            full_derivatives = np.zeros(len(x_sorted))
            full_derivatives[:-1] = data_derivatives
            full_derivatives[-1] = data_derivatives[-1]
            
            # Now we have derivative values at original data points
            # We need to interpolate to get derivatives at the fitted curve points
            from scipy.interpolate import interp1d
            
            # Create an interpolation function (use linear interpolation)
            if len(x_sorted) > 2:  # Need at least 3 points for cubic interpolation
                try:
                    interp_func = interp1d(x_sorted, full_derivatives, kind='cubic', bounds_error=False, fill_value='extrapolate')
                except:
                    interp_func = interp1d(x_sorted, full_derivatives, kind='linear', bounds_error=False, fill_value='extrapolate')
            else:
                interp_func = interp1d(x_sorted, full_derivatives, kind='linear', bounds_error=False, fill_value='extrapolate')
            
            # Interpolate to get derivatives at the fitted curve points
            y_derivative = interp_func(x_fit)
        else:
            # If only one data point, can't calculate derivative
            y_derivative = np.zeros_like(x_fit)
        
        # Pass the first coefficient as derivative params (for equation representation)
        if fit_type == "Linear":
            derivative_params = [a]
        elif fit_type == "Polynomial":
            coeffs = params
            degree = len(coeffs) - 1  # Get the degree from coefficient length
            derivative_coeffs = [(degree - i) * coef for i, coef in enumerate(coeffs) if i < len(coeffs) - 1]
            derivative_params = derivative_coeffs
        elif fit_type == "Logarithmic":
            derivative_params = [a]
        elif fit_type == "Exponential":
            derivative_params = [a, b]
        else:
            derivative_params = []
    
    elif method == "By Point" and y_fit is not None:
        # Fallback to using fitted curve points if original data not provided
        # Finite difference method: (y[i+1] - y[i]) / (x[i+1] - x[i])
        dx = np.diff(x_fit)
        dy = np.diff(y_fit)
        
        # Calculate the derivative at each point
        derivative_points = dy / dx
        
        # For the last point, use the same derivative as the second-to-last point
        y_derivative = np.zeros_like(x_fit)
        y_derivative[:-1] = derivative_points
        y_derivative[-1] = derivative_points[-1]
        
        # Pass the first coefficient as derivative params (for equation representation)
        if fit_type == "Linear":
            derivative_params = [a]
        elif fit_type == "Polynomial":
            coeffs = params
            degree = len(coeffs) - 1  # Get the degree from coefficient length
            derivative_coeffs = [(degree - i) * coef for i, coef in enumerate(coeffs) if i < len(coeffs) - 1]
            derivative_params = derivative_coeffs
        elif fit_type == "Logarithmic":
            derivative_params = [a]
        elif fit_type == "Exponential":
            derivative_params = [a, b]
        else:
            derivative_params = []
    else:
        # Analytical derivatives (By Equation method)
        if fit_type == "Linear":
            # Derivative of y = ax + b is y' = a
            a, b = params
            y_derivative = np.full_like(x_fit, a)
            derivative_params = [a]
            
        elif fit_type == "Polynomial":
            # Derivative of polynomial
            coeffs = params
            degree = len(coeffs) - 1  # Get the degree from coefficient length
            derivative_coeffs = [(degree - i) * coef for i, coef in enumerate(coeffs) if i < len(coeffs) - 1]
            p_derivative = np.poly1d(derivative_coeffs)
            y_derivative = p_derivative(x_fit)
            derivative_params = derivative_coeffs
            
        elif fit_type == "Logarithmic":
            # Derivative of y = a*ln(x) + b is y' = a/x
            a, b = params
            y_derivative = a / x_fit
            derivative_params = [a]
            
        elif fit_type == "Exponential":
            # Derivative of y = a*exp(b*x) is y' = a*b*exp(b*x)
            a, b = params
            y_derivative = a * b * np.exp(b * x_fit)
            derivative_params = [a, b]
        
        else:
            # Default for unknown types
            y_derivative = np.zeros_like(x_fit)
            derivative_params = []
    
    return y_derivative, derivative_params

def get_derivative_equation_text(fit_type, params):
    """
    Get the text representation of the derivative equation.
    
    Args:
        fit_type (str): Type of curve fitting
        params: Original equation parameters
        
    Returns:
        str: Text representation of the derivative equation
    """
    if fit_type == "Linear":
        a, b = params
        return f"y' = {a:.4f}"
        
    elif fit_type == "Polynomial":
        degree = len(params) - 1
        derivative_eq = "y' = "
        
        for i, coef in enumerate(params):
            if i == len(params) - 1:  # Constant term
                break
                
            power = degree - i
            deriv_coef = power * coef
            
            if i == 0:
                derivative_eq += f"{deriv_coef:.4f}x^{power-1}" if power > 1 else f"{deriv_coef:.4f}"
            else:
                sign = "+" if deriv_coef >= 0 else "-"
                abs_coef = abs(deriv_coef)
                
                if power == 1:  # x^0 = 1
                    derivative_eq += f" {sign} {abs_coef:.4f}"
                else:
                    derivative_eq += f" {sign} {abs_coef:.4f}x^{power-1}"
        
        return derivative_eq
        
    elif fit_type == "Logarithmic":
        a, b = params
        return f"y' = {a:.4f}/x"
        
    elif fit_type == "Exponential":
        a, b = params
        return f"y' = {a*b:.4f}e^({b:.4f}x)"
        
    return ""

def perform_curve_fitting(x_data, y_data, fit_type, poly_degree=2, fixed_point=None):
    """
    Perform curve fitting on the provided data.
    
    Args:
        x_data (list): X values
        y_data (list): Y values
        fit_type (str): Type of curve fitting ('Linear', 'Polynomial', 'Logarithmic', 'Exponential')
        poly_degree (int): Degree of polynomial for polynomial fitting
        fixed_point (tuple, optional): An (x,y) point the curve must pass through
        
    Returns:
        tuple: (x_fit, y_fit, equation_params, r_squared, covariance)
    """
    x_array = np.array(x_data)
    y_array = np.array(y_data)
    
    # Create a smooth x array for the fitted curve
    x_fit = np.linspace(min(x_array), max(x_array), 1000)
    
    if fit_type == "Linear":
        # Linear fit: y = ax + b
        def linear_func(x, a, b):
            return a * x + b
        
        if fixed_point is not None:
            # If we have a fixed point (x0, y0), then we have:
            # y0 = a*x0 + b => b = y0 - a*x0
            # Substitute this into the original equation:
            # y = a*x + (y0 - a*x0) = a*(x - x0) + y0
            # So we're fitting: y - y0 = a*(x - x0)
            # We only need to find 'a' now
            x0, y0 = fixed_point
            
            # Compute the weighted linear regression through the origin
            # for the translated coordinates
            x_centered = x_array - x0
            y_centered = y_array - y0
            
            if np.sum(x_centered**2) > 0:
                # Calculate slope using weighted linear regression through origin
                a = np.sum(x_centered * y_centered) / np.sum(x_centered**2)
                b = y0 - a * x0
                
                # Calculate y values for the fitted curve
                y_fit = linear_func(x_fit, a, b)
                
                # Estimate the covariance matrix
                # This is a simplified approach since we forced a point
                residuals = y_array - linear_func(x_array, a, b)
                variance = np.sum(residuals**2) / (len(residuals) - 1)
                covariance = np.array([[variance, 0], [0, variance]])
            else:
                # Can't fit a line with fixed point constraint - degenerate case
                a, b = 0, y0
                y_fit = np.full_like(x_fit, y0)
                covariance = np.zeros((2, 2))
                
            params = np.array([a, b])
        else:
            # Standard fit without fixed point
            from scipy import optimize as scipy_optimize
            params, covariance = scipy_optimize.curve_fit(linear_func, x_array, y_array)
            a, b = params
            
            # Calculate y values for the fitted curve
            y_fit = linear_func(x_fit, a, b)
        
        # Calculate R-squared
        residuals = y_array - linear_func(x_array, a, b)
        ss_res = np.sum(residuals**2)
        ss_tot = np.sum((y_array - np.mean(y_array))**2)
        r_squared = 1 - (ss_res / ss_tot)
        
        return x_fit, y_fit, params, r_squared, covariance
    
    elif fit_type == "Polynomial":
        if fixed_point is not None and poly_degree >= 1:
            # If we have a fixed point (x0, y0), we can constrain one coefficient
            # We'll use Lagrange interpolation to ensure the curve passes through this point
            x0, y0 = fixed_point
            
            # First we fit a polynomial without constraints
            coeffs_unconstrained = np.polyfit(x_array, y_array, poly_degree)
            
            # Calculate the value at our fixed point
            p_unconstrained = np.poly1d(coeffs_unconstrained)
            y0_unconstrained = p_unconstrained(x0)
            
            # Calculate the adjustment needed to pass through fixed point
            adjustment = y0 - y0_unconstrained
            
            # Add a polynomial that's zero everywhere except at our fixed point
            # The simplest way is to add a constant to the lowest coefficient (constant term)
            coeffs = coeffs_unconstrained.copy()
            coeffs[-1] += adjustment
            
            # Create the final polynomial
            p = np.poly1d(coeffs)
            
            # Estimate covariance from the residuals
            # This is an approximation as we're not really computing proper covariance
            # after applying the constraint
            residuals = y_array - p(x_array)
            variance = np.sum(residuals**2) / (len(residuals) - poly_degree - 1) if len(residuals) > poly_degree + 1 else 1.0
            cov = np.eye(poly_degree + 1) * variance
        else:
            # Standard polynomial fit without constraint
            coeffs, cov = np.polyfit(x_array, y_array, poly_degree, cov=True)
            p = np.poly1d(coeffs)
        
        # Calculate y values for the fitted curve
        y_fit = p(x_fit)
        
        # Calculate R-squared
        residuals = y_array - p(x_array)
        ss_res = np.sum(residuals**2)
        ss_tot = np.sum((y_array - np.mean(y_array))**2)
        r_squared = 1 - (ss_res / ss_tot)
        
        return x_fit, y_fit, coeffs, r_squared, cov
    
    elif fit_type == "Logarithmic":
        # Logarithmic fit: y = a * ln(x) + b
        def log_func(x, a, b):
            return a * np.log(x) + b
        
        # Filter out non-positive x values for logarithmic fit
        valid_indices = x_array > 0
        x_valid = x_array[valid_indices]
        y_valid = y_array[valid_indices]
        
        if len(x_valid) < 2:  # Need at least 2 points for a fit
            return [], [], [], 0, None
        
        try:
            if fixed_point is not None:
                x0, y0 = fixed_point
                
                # Check if the fixed point has a positive x-value (required for logarithmic fit)
                if x0 <= 0:
                    # Can't use a non-positive x-value for a logarithmic fit
                    return [], [], [], 0, None
                
                # For a logarithmic function y = a*ln(x) + b with a fixed point (x0,y0):
                # y0 = a*ln(x0) + b => b = y0 - a*ln(x0)
                # To find 'a', we substitute this into the original equation:
                # y = a*ln(x) + (y0 - a*ln(x0)) = a*ln(x/x0) + y0
                
                # Transform the data
                ln_ratio = np.log(x_valid/x0)
                y_shift = y_valid - y0
                
                # Find the optimal 'a' using linear regression through origin
                if np.sum(ln_ratio**2) > 0:
                    a = np.sum(ln_ratio * y_shift) / np.sum(ln_ratio**2)
                    b = y0 - a * np.log(x0)
                    
                    # Calculate y values for the fitted curve
                    x_fit_valid = np.linspace(min(x_valid), max(x_valid), 1000)
                    y_fit = log_func(x_fit_valid, a, b)
                    
                    # Calculate R-squared
                    residuals = y_valid - log_func(x_valid, a, b)
                    ss_res = np.sum(residuals**2)
                    ss_tot = np.sum((y_valid - np.mean(y_valid))**2)
                    r_squared = 1 - (ss_res / ss_tot)
                    
                    # Estimate covariance matrix (simplified)
                    variance = np.sum(residuals**2) / (len(residuals) - 1)
                    covariance = np.array([[variance, 0], [0, variance]])
                else:
                    # Degenerate case
                    return [], [], [], 0, None
            else:
                # Regular fit without fixed point constraint
                from scipy import optimize as scipy_optimize
                params, covariance = scipy_optimize.curve_fit(log_func, x_valid, y_valid)
                a, b = params
                
                # Calculate y values for the fitted curve
                x_fit_valid = np.linspace(min(x_valid), max(x_valid), 1000)
                y_fit = log_func(x_fit_valid, a, b)
                
                # Calculate R-squared
                residuals = y_valid - log_func(x_valid, a, b)
                ss_res = np.sum(residuals**2)
                ss_tot = np.sum((y_valid - np.mean(y_valid))**2)
                r_squared = 1 - (ss_res / ss_tot)
            
            return x_fit_valid, y_fit, [a, b], r_squared, covariance
        except:
            # Return empty arrays if fit fails
            return [], [], [], 0, None
    
    elif fit_type == "Exponential":
        # Exponential fit: y = a * exp(b * x)
        def exp_func(x, a, b):
            return a * np.exp(b * x)
        
        try:
            if fixed_point is not None:
                x0, y0 = fixed_point
                
                # For exponential function with fixed point, we need y0 > 0
                if y0 <= 0:
                    # Can't use non-positive y-value for exponential fit with fixed point
                    return [], [], [], 0, None
                
                # Filter out non-positive y values for the fit
                valid_indices = y_array > 0
                if not np.any(valid_indices):
                    # If no positive y values, can't perform the fit
                    return [], [], [], 0, None
                
                x_valid = x_array[valid_indices]
                y_valid = y_array[valid_indices]
                
                if len(x_valid) < 2:  # Need at least 2 points for a fit
                    return [], [], [], 0, None
                
                # For exponential function y = a*exp(b*x) with a fixed point (x0,y0):
                # y0 = a*exp(b*x0) => a = y0/exp(b*x0)
                
                # We still need to find the optimal b. We can solve this using optimization.
                def objective_fn(b_value):
                    # Calculate 'a' based on fixed point constraint
                    a_value = y0 / np.exp(b_value * x0)
                    
                    # Calculate residuals for this set of parameters
                    y_pred = a_value * np.exp(b_value * x_valid)
                    residuals = y_pred - y_valid
                    return np.sum(residuals**2)  # Return sum of squared residuals
                
                # Find the optimal 'b' using scipy optimization
                from scipy import optimize
                result = optimize.minimize_scalar(objective_fn)
                
                if result.success:
                    b = result.x
                    a = y0 / np.exp(b * x0)
                    
                    # Calculate y values for the fitted curve
                    y_fit = exp_func(x_fit, a, b)
                    
                    # Calculate R-squared
                    residuals = y_valid - exp_func(x_valid, a, b)
                    ss_res = np.sum(residuals**2)
                    ss_tot = np.sum((y_valid - np.mean(y_valid))**2)
                    r_squared = 1 - (ss_res / ss_tot)
                    
                    # Estimate covariance matrix (simplified)
                    variance = np.sum(residuals**2) / (len(residuals) - 1)
                    covariance = np.array([[variance, 0], [0, variance]])
                else:
                    # Optimization failed
                    return [], [], [], 0, None
            else:
                # Regular fit without fixed point constraint
                # Filter out non-positive y values for initial guess
                valid_indices = y_array > 0
                if not np.any(valid_indices):
                    # If no positive y values, use absolute values
                    y_abs = np.abs(y_array)
                    from scipy import optimize as scipy_optimize
                    params, covariance = scipy_optimize.curve_fit(exp_func, x_array, y_abs, p0=(1.0, 0.1))
                else:
                    x_valid = x_array[valid_indices]
                    y_valid = y_array[valid_indices]
                    from scipy import optimize as scipy_optimize
                    if len(x_valid) < 2:  # Need at least 2 points for a fit
                        params, covariance = scipy_optimize.curve_fit(exp_func, x_array, np.abs(y_array), p0=(1.0, 0.1))
                    else:
                        params, covariance = scipy_optimize.curve_fit(exp_func, x_valid, y_valid, p0=(1.0, 0.1))
                
                a, b = params
                
                # Calculate y values for the fitted curve
                y_fit = exp_func(x_fit, a, b)
                
                # Calculate R-squared
                residuals = y_array - exp_func(x_array, a, b)
                ss_res = np.sum(residuals**2)
                ss_tot = np.sum((y_array - np.mean(y_array))**2)
                r_squared = 1 - (ss_res / ss_tot)
            
            return x_fit, y_fit, [a, b], r_squared, covariance
        except:
            # Return empty arrays if fit fails
            return [], [], [], 0, None
    
    # Default: return empty arrays
    return [], [], [], 0, None

def get_equation_text(fit_type, params, covariance=None, show_variance=False):
    """
    Get the text representation of the equation for the fitted curve.
    
    Args:
        fit_type (str): Type of curve fitting
        params: Equation parameters
        covariance: Optional covariance matrix for parameter uncertainties
        show_variance: Whether to display parameter uncertainties
        
    Returns:
        str: Text representation of the equation
    """
    if fit_type == "Linear":
        a, b = params
        
        if show_variance and covariance is not None:
            # Calculate standard errors (square root of diagonal elements of covariance matrix)
            a_err = np.sqrt(covariance[0, 0])
            b_err = np.sqrt(covariance[1, 1])
            return f"y = ({a:.4f} ± {a_err:.4f})x + ({b:.4f} ± {b_err:.4f})"
        else:
            return f"y = {a:.4f}x + {b:.4f}"
    
    elif fit_type == "Polynomial":
        degree = len(params) - 1
        equation = "y = "
        
        if show_variance and covariance is not None:
            # For polynomial with uncertainties
            for i, coef in enumerate(params):
                power = degree - i
                std_err = np.sqrt(covariance[i, i])
                
                if i == 0:
                    equation += f"({coef:.4f} ± {std_err:.4f})x^{power}"
                else:
                    sign = "+" if coef >= 0 else "-"
                    abs_coef = abs(coef)
                    
                    if power == 0:
                        equation += f" {sign} ({abs_coef:.4f} ± {std_err:.4f})"
                    elif power == 1:
                        equation += f" {sign} ({abs_coef:.4f} ± {std_err:.4f})x"
                    else:
                        equation += f" {sign} ({abs_coef:.4f} ± {std_err:.4f})x^{power}"
        else:
            # For polynomial without uncertainties
            for i, coef in enumerate(params):
                power = degree - i
                
                if i == 0:
                    equation += f"{coef:.4f}x^{power}"
                else:
                    sign = "+" if coef >= 0 else "-"
                    abs_coef = abs(coef)
                    
                    if power == 0:
                        equation += f" {sign} {abs_coef:.4f}"
                    elif power == 1:
                        equation += f" {sign} {abs_coef:.4f}x"
                    else:
                        equation += f" {sign} {abs_coef:.4f}x^{power}"
        
        return equation
    
    elif fit_type == "Logarithmic":
        a, b = params
        
        if show_variance and covariance is not None:
            # Calculate standard errors
            a_err = np.sqrt(covariance[0, 0])
            b_err = np.sqrt(covariance[1, 1])
            return f"y = ({a:.4f} ± {a_err:.4f})ln(x) + ({b:.4f} ± {b_err:.4f})"
        else:
            return f"y = {a:.4f}ln(x) + {b:.4f}"
    
    elif fit_type == "Exponential":
        a, b = params
        
        if show_variance and covariance is not None:
            # Calculate standard errors
            a_err = np.sqrt(covariance[0, 0])
            b_err = np.sqrt(covariance[1, 1])
            return f"y = ({a:.4f} ± {a_err:.4f})e^(({b:.4f} ± {b_err:.4f})x)"
        else:
            return f"y = {a:.4f}e^({b:.4f}x)"
    
    return ""
