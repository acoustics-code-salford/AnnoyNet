import matplotlib.pyplot as plt

def plot_circumplex(dataframe):

    _, axes = plt.subplots(nrows=1, ncols=1, figsize=(6, 6))

    axes.axhline(0, color='black', linewidth=0.9)
    axes.axvline(0, color='black', linewidth=0.9)

    axes.set_xlim((-1.1, 1.1))
    axes.set_ylim((-1.1, 1.1))

    circ = plt.Circle((0, 0), radius=1, edgecolor='black', facecolor='None', linewidth=0.9)
    axes.add_patch(circ)

    axes.text(-1.18, -0.16, 'Unpleasant', rotation=90)
    axes.text(1.1, -0.13, 'Pleasant', rotation=270)
    axes.text(-0.13, 1.13, 'Aroused')
    axes.text(-0.09, -1.16, 'Calm')

    dataframe.plot.scatter('Valence', 'Arousal', ax=axes, s=10)

    plt.axis('off')
