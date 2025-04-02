# Data Plotting & Curve Fitting Tool

A versatile data visualization tool that handles multiple input formats, supports error bars, and generates ready-to-use Python code.

## Features

### Data Input Flexibility
- Accepts X/Y data in multiple formats:
  - Space-separated: `1 2 3 4 5`
  - Comma/line-separated: `1.5, 2.0\n3.5, 4.0`
  - One number per line: `1\n2\n3\n4\n5`
  - Mixed decimal formats: `1.5` or `1,5`

### Error Bars Support
- Optional X/Y error values
- Visualizes vertical/horizontal error bars

### Multiple Data Series
- Support for unlimited datasets on one graph
- Customizable labels and colors

### Curve Fitting
- Linear regression: `y = ax + b`
- Polynomial regression (customizable degree)
- Logarithmic regression: `y = a * ln(x) + b`
- Exponential regression: `y = a * exp(b * x)`
- Annotations for equations and R² values
- Parameter uncertainty/variance display
- Custom annotation positioning (percentage-based)

### Graph Customization
- Titles and axis labels
- Grid and legend positioning
- Auto-scaling for clean visuals

### Output
- Ready-to-run Python code (using matplotlib/scipy)
- Download graph as PNG image

## Usage

1. Enter your X/Y data in the text areas
2. Add optional error bars if needed
3. Customize your graph settings
4. Choose curve fitting options
5. View the plot and download as needed
6. Copy the generated Python code to reproduce the graph

### Annotation Positioning

You can precisely position equations and R² values on your plot using the percentage-based positioning system:

1. Select "Custom (%)" from the "Annotation Position" dropdown
2. Use the X Position (%) slider to set horizontal position (0% = left edge, 100% = right edge)
3. Use the Y Position (%) slider to set vertical position (0% = bottom edge, 100% = top edge)
4. The annotation will be placed at the exact position specified by these percentages

Other position options include:
- Auto (near the start of the curve)
- Upper Right
- Upper Left
- Lower Left
- Lower Right
- Center

## Requirements

- Python 3.6+
- Streamlit
- NumPy
- SciPy
- Matplotlib
- Pandas

## Running the Application

```bash
streamlit run app.py
