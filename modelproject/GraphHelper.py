import seaborn as sns
import matplotlib.pyplot as plt

import ModelClass


###########################
#     Transition plot     #
###########################


def transition_to_ss_plot():
    """
    Generates a steady-state plot for a given model using seaborn and matplotlib. It is designed to work with a model
    that has a 'T' attribute representing the time period.

    Parameters:
    None

    Returns:
    matplotlib.pyplot: A matplotlib pyplot object with the plot configuration set.
                       
    """
    
    # Set figure
    plt.figure(figsize=(7, 3))
    sns.set_style("whitegrid")
    
    # Set labels
    plt.xlabel('t - period', fontsize=11)
    plt.ylabel('Value', fontsize=11)

    # Adding a custom color palette
    custom_palette = sns.color_palette("husl", 5)
    sns.set_palette(custom_palette)

    # Set limits on the x axis
    model = ModelClass.MalthusModel()
    plt.xlim(1, model.val.T)

    return plt