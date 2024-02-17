import matplotlib.pyplot as plt
import seaborn as sns

custom_params = {"axes.spines.right": False, 'grid.color': 'lightgray', 'axes.grid': True, "axes.spines.top": False}
sns.set_theme(style="ticks", rc=custom_params)


def plot_surival_function(sf, bins, title=""):
    plt.step(bins, sf)
    plt.xlabel('Time')
    plt.ylabel('Survival probability')
    plt.title(title)
    plt.show()
