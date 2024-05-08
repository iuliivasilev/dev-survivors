import matplotlib.pyplot as plt
import seaborn as sns

custom_params = {"axes.spines.right": False, 'grid.color': 'lightgray', 'axes.grid': True, "axes.spines.top": False}
sns.set_theme(style="ticks", rc=custom_params)


def plot_survival_function(sf, bins, title=""):
    """
    Classic survival function plotting

    Parameters
    ----------
    sf: iterable (with len(sf) == len(bins))
        Survival probabilities
    bins: iterable
        Times of probabilities
    title: str

    """
    plt.step(bins, sf)
    plt.xlabel('Time')
    plt.ylabel('Survival probability')
    plt.title(title)
    plt.show()
