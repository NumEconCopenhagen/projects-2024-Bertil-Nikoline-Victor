import matplotlib.pyplot as plt
plt.rcParams.update({"axes.grid":True,"grid.color":"black","grid.alpha":"0.25","grid.linestyle":"--"})
plt.rcParams.update({'font.size': 14})


def create_edgeworth_plot(w1bar = 1, w2bar = 1):
    """
    This function creates an Edgeworth Box plot for a two-good economy with two consumers (A and B).
    
    Parameters:
    w1bar (float): The total endowment of good 1 in the economy. Default is 1.
    w2bar (float): The total endowment of good 2 in the economy. Default is 1.
    
    Returns:
    ax_A (matplotlib.axes.Axes): The Axes object for consumer A's plot.
    
    The function first creates a square plot area representing the total endowment of goods. 
    It then sets up two coordinate systems: one for consumer A (bottom left origin) and one for consumer B (top right origin).
    The x-axis represents good 1 and the y-axis represents good 2 for both consumers.
    """

    fig = plt.figure(frameon=False,figsize=(7,7), dpi=100)

    ax_A = fig.add_subplot(1, 1, 1)
    ax_A.set_xlabel("$x_1^A$")
    ax_A.set_ylabel("$x_2^A$")

    temp = ax_A.twinx()
    temp.set_ylabel("$x_2^B$")

    ax_B = temp.twiny()
    ax_B.set_xlabel("$x_1^B$")
    ax_B.invert_xaxis()
    ax_B.invert_yaxis()
 
    ax_A.plot([0,w1bar],[0,0],lw=2,color='black')
    ax_A.plot([0,w1bar],[w2bar,w2bar],lw=2,color='black')
    ax_A.plot([0,0],[0,w2bar],lw=2,color='black')
    ax_A.plot([w1bar,w1bar],[0,w2bar],lw=2,color='black')

    ax_A.set_xlim([-0.1, w1bar + 0.1])
    ax_A.set_ylim([-0.1, w2bar + 0.1])
    ax_B.set_xlim([w1bar + 0.1, -0.1])
    ax_B.set_ylim([w2bar + 0.1, -0.1])

    return ax_A