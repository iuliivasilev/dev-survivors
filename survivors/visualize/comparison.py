import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

custom_params = {"axes.spines.right": False, 'grid.color': 'lightgray', 'axes.grid': True, "axes.spines.top": False}
sns.set_theme(style="ticks", rc=custom_params)


def plot_func_comparison(y_true, y_preds, labels):
    """
    Plot multiple functions by method

    Parameters
    ----------
    y_true: structured np.ndarray with shape=(n, 2)
    y_preds: list of np.ndarray with shape=(n, n_bins)
    labels: list of str

    """
    linestyle = ('solid' if y_true["cens"] else 'dashed')
    fig, ax = plt.subplots(figsize=(8, 5))

    for y_pred, label in zip(y_preds, labels):
        ax.plot(y_pred, label=label)
    ax.hlines(1.0, 0, y_true["time"], color='r', linestyle=linestyle)
    ax.hlines(0.0, y_true["time"], len(y_preds[0]), color='r', linestyle=linestyle)
    ax.vlines(y_true["time"], 0, 1, color='r',
              linestyle=linestyle, linewidth=2, label="Target S(t)")
    ax.legend()
    #     plt.title(f'Прогноз для терминального события с временем T={y_true["time"]}')
    #     plt.xlabel('Время')
    #     plt.ylabel('Вероятность выживания')
    plt.title(f'Prediction for {"terminal" if y_true["cens"] else "censured"} event with time={y_true["time"]}')
    plt.xlabel('Time')
    plt.ylabel('Survival probability')
    plt.show()


def plot_metric_comparison(y_true, y_preds, labels, bins, metric):
    """
    Comparison quality metric in time

    Parameters
    ----------
    y_true: structured np.ndarray with shape=(n, 2)
    y_preds: list of np.ndarray with shape=(n, n_bins)
    labels: list of str
    bins: np.ndarray with shape=(n_bins)
    metric: function of internal metric

    """
    fig, ax = plt.subplots()
    m_name = metric.__name__

    for y_pred, label in zip(y_preds, labels):
        m_by_time = metric(y_true[np.newaxis], y_true[np.newaxis], y_pred[np.newaxis], bins, axis=1)
        m_val = metric(y_true[np.newaxis], y_true[np.newaxis], y_pred[np.newaxis], bins, axis=-1)
        ax.plot(m_by_time, label=f"{label} ({m_name}={m_val:.3f})")
    ax.legend()
    plt.title(f'{m_name}(t) for {"terminal" if y_true["cens"] else "censured"} event with time={y_true["time"]}')
    #     plt.title(f'{m_name}(t) для терминального события с временем T={y_true["time"]}')
    if m_name.find("auprc") == -1:
        plt.xlabel('Time')
    else:
        plt.xlabel(r'$\varphi$')
        plt.xticks(np.linspace(0, 100, 5), labels=np.linspace(0, 1, 5))
    plt.ylabel(f'{m_name}(t) value')
    #     plt.ylabel(f'Значение {m_name}(t)')
    plt.show()
