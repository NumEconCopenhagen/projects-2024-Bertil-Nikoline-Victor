# Importing packages
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit


############################
#     Time series plot     #
############################


def line_plot_with_labels(data, title, xlabel, ylabel, xName, yName, color="orange"):
    """
    Creates a line plot with specified labels and attributes.

    Parameters:
    data (DataFrame): A pandas DataFrame containing the data.
    title (str): The title of the plot.
    xlabel (str): The label for the x-axis.
    ylabel (str): The label for the y-axis.
    xName (str): The name of the column representing the x-axis data.
    yName (str): The name of the column representing the y-axis data.
    color (str, optional): The color of the line plot. Defaults to "orange".

    Returns:
    matplotlib.pyplot.figure: A matplotlib figure object containing the plot.
    """    
    # Creating a seaborn plot
    plt.figure(figsize=(6, 4))
    sns.lineplot(data=data, x=xName, y=yName, linewidth=1.5, color=color, marker='o', markersize=5)

    # Adding title and labels
    plt.title(title, fontsize=14)
    plt.xlabel(xlabel, fontsize=11)
    plt.ylabel(ylabel, fontsize=11)

    # Adding grid and annotations
    plt.grid(True, linestyle='--', alpha=0.7)

    # Adding a custom color palette
    custom_palette = sns.color_palette("husl", 5)
    sns.set_palette(custom_palette)

    return plt


####################
#     Box Plot     #
####################

def plot_boxplots(data, column_name1, column_name2):
    """
    Creates a box plot with two time series.

    Parameters:
    data (DataFrame): A pandas DataFrame containing the data.
    column_name1 (str): The name of the column representing the x-axis data.
    column_name2 (str): The name of the column representing the y-axis data.

    Returns:
    An interactive plot with two box plots over two different time-series.
    """

    # Creating subplots and setting figsize
    fig, axes = plt.subplots(1, 2, figsize=(6, 4))

    # Boxplot design commands
    boxplot_params = dict(
        boxprops=dict(color='red', linewidth=3),
        whiskerprops=dict(color='black', linewidth=2),
        capprops=dict(color='black', linewidth=2),
        medianprops=dict(color='black', linewidth=2,linestyle='--'),
        flierprops=dict(marker='o', markersize=4, markerfacecolor='red', markeredgecolor='black')
    )
    # First boxplot w. specified design and y-/x-axis
    axes[0].boxplot(data[column_name1].dropna(),**boxplot_params)
    axes[0].set_title(f'{column_name1}')
    axes[0].set_ylabel('%')
    axes[0].set_ylim(-5,20)
    axes[0].set_xticklabels([])

    # Second boxplot w. specified design and y-/x-axis
    axes[1].boxplot(data[column_name2].dropna(), **boxplot_params)
    axes[1].set_title(f'{column_name2}')
    axes[1].set_ylabel('%')
    axes[1].set_ylim(-5,20)
    axes[1].yaxis.set_ticks_position('right')
    axes[1].yaxis.set_label_position('right')
    axes[1].set_xticklabels([])

    plt.show()


##########################
#     Phillips Curve     #
##########################

def plot_philips_curve_static(data, column_names, title, country_names, width = 5, height = 5):
    """
    Plots Phillips curve with static attributes.

    Parameters:
    data (DataFrame): A pandas DataFrame containing the data.
    column_names a list of tuples: A list of tuples that contains the x and y column names, so it's possible to plot multiple series.
    title (str): The title of the plot.
    country_names list of (str): A list of country names corresponding to the column_names.
    width (number): Height of the plot.
    height (number): Width of the plot.

    Returns:
    matplotlib.pyplot.figure: A matplotlib figure object containing the plot.

    Notes:
    The function internally calls calculate_exponential_fit to determine the x and y values
    for the exponential regression line.

    calculate_exponential_fit returns two lists, xValues and yValues, which represent the x and y
    values respectively for the exponential regression line. These values are used for plotting
    the tendency line.
    """
    # Create scatter plot with tendency line
    plt.figure(figsize=(width, height))  # Set figure size

    # Set Seaborn style and context
    sns.set_style("whitegrid")
    sns.set_context("notebook", font_scale=1.2)

    for index, x_and_y_column in enumerate(column_names):

        # Column names
        (xName, yName) = x_and_y_column

        # Filter out NaN rows - or else the plot wont work
        data_without_NaN = data.dropna(subset=[xName, yName], how='any')

        # Colors for the graph
        graph_colors = [
            'blue',
            'green',
            'red',
            'cyan',
            'magenta',
            'yellow',
            'black'
        ]

        # Scatter plot
        sns.scatterplot(data=data_without_NaN, x=xName, y=yName, color=graph_colors[index], alpha=0.4, label=f"Data {country_names[index]}")

        # Tendency line - Exponential
        (xValues, yValues) = calculate_exponential_fit(data_without_NaN[xName], data_without_NaN[yName])

        xName.replace('_', ' ')

        yName.replace('_', ' ')

        # Plot exponential curve
        plt.plot(xValues, yValues, color=graph_colors[index], linewidth=2, linestyle='dashed', label=f"Exponential {country_names[index]}")


    # Adjust tick font size
    plt.xticks(fontsize=10)  # Increase tick label font size
    plt.yticks(fontsize=10)  # Increase tick label font size

    # Some more styling
    plt.grid(True, linestyle='--', alpha=0.7)  # Add gridlines with dashed style and reduced opacity
    plt.legend(loc='upper right', fontsize=10)  # Add a legend with increased font size
    plt.tight_layout()  # Adjust plot layout for better presentation

    # Add title and labels
    plt.title(title, fontsize=14)
    plt.xlabel('Unemployment rate (pct.)', fontsize=11)
    plt.ylabel('Inflation rate (pct.)', fontsize=11)

    # Change legend font size
    plt.legend(fontsize=12)

    return plt




###########################
#     Exponential fit     #
###########################

def calculate_exponential_fit(x: list, y: list):
    """
    Calculates an expontial fit from a series of x and y values.

    Parameters:
    x (list): A list of x values
    y (list): A list of y values

    Returns:
    type: Returns two lists of the calculated exponential fit based on the parameters with x and y values.
    """
    # Define an exponential function
    def exponential_func(x, a, b):
        return a * np.exp(b * x)

    # Fit the exponential curve to the data
    popt, pcov = curve_fit(exponential_func, x, y)

    # Modify parameters for steeper slope
    popt[1] *= 1.0  # Adjust the slope parameter b to increase the slope

    # Generate points for the exponential curve
    x_values = np.linspace(min(x) - 1, max(x) + 1, 100)
    y_values = exponential_func(x_values, *popt)

    return (x_values, y_values)