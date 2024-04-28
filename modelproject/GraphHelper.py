import seaborn as sns
import matplotlib.pyplot as plt

import ModelClass


###########################
#     Transition plot     #
###########################


def transition_to_ss_plot():
    
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