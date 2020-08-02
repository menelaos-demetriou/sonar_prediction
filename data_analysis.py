import pandas as pd
import seaborn as sns
from itertools import combinations
from dython.nominal import associations
import matplotlib.pyplot as plt
plt.style.use('ggplot')


def plot_against_target(data):
    # Get a list of all features
    features = list(data.columns)

    # Remove the last one since it's the target variable
    features.pop()

    # Split list into 6 lists in order to plot them easier
    f = lambda features, n=10: [features[i:i + n] for i in range(0, len(features), n)]
    new_list = f(features)

    for val, sub_list in enumerate(new_list):

        fig, axs = plt.subplots(2, 5, figsize=(15, 6), facecolor='w', edgecolor='k')
        fig.subplots_adjust(hspace=.5, wspace=.001)

        axs = axs.ravel()

        # Plot each feature against the target
        for index, feature in enumerate(sub_list):
            sns.boxplot(y=feature, x=60, data=data, ax = axs[index])
            axs[index].set_xlabel("Type")
            axs[index].set_title("Feature %s" % feature)

        plt.tight_layout()
        plt.savefig("plots/plot_against_target_%s.png" % val)
        plt.show()

# TODO: Need to optimise pairplot (not all pairs are implemented)
def plot_against_each_other(data):
    # Get a list of all features
    features = list(data.columns)

    # Remove the last one since it's the target variable
    features.pop()

    # Split list into lists in order to plot them easier
    f = lambda features, n=5: [features[i:i + n] for i in range(0, len(features), n)]
    new_list = f(features)

    for val, sub_list in enumerate(new_list):

        sns.pairplot(data, vars=sub_list, hue=60)
        plt.tight_layout()
        plt.savefig("plots/pair_plot_%s.png" % val)
        plt.show()


def correlations(data):
    associations(data, figsize=(15, 15), cmap="viridis")
    plt.savefig("plots/correlations.png")
    plt.show()



def main():
    # Read file with dataset
    data = pd.read_csv("data/sonar.all-data", header= None)

    # Data type of each feature
    print(data.info())

    # Check Null entries
    print(round(100 * (data.isnull().sum() / len(data.index)), 2))

    # Check if imbalanced test set
    print(data[60].value_counts(normalize=True))

    # Plot all features against the target
    # plot_against_target(data)

    # Plot all features against each other
    # plot_against_each_other(data)

    # Create heatmap of correlations
    correlations(data)

if __name__ == "__main__":
    main()