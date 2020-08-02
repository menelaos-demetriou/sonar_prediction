import pandas as pd
from sklearn.feature_selection import RFECV
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
plt.style.use('ggplot')


def feature_selection(data):
    clf = LogisticRegression(max_iter=1000)
    y = data[60].copy()
    X = data.loc[:, ~data.columns.isin([60])].copy()

    enc = LabelEncoder()
    y = enc.fit_transform(y)

    rfecv = RFECV(estimator=clf, step=1, cv=StratifiedKFold(2),
                  scoring='accuracy')
    rfecv.fit(X, y)

    # Plot number of features VS. cross-validation scores
    plt.figure()
    plt.xlabel("Number of features selected")
    plt.ylabel("Cross validation score (nb of correct classifications)")
    plt.plot(range(1, len(rfecv.grid_scores_) + 1), rfecv.grid_scores_)
    plt.savefig("plots/RFE.png")
    plt.show()

    print("Optimal number of features : %d" % rfecv.n_features_)
    print("Hello")
    X = X.loc[:, list(rfecv.support_)]
    # Return dataframe with best features and target
    return pd.concat([X, pd.Series(y, name="target")], axis = 1)

def main():
    # Read file with dataset
    data = pd.read_csv("data/sonar.all-data", header=None)
    data = data.dropna()
    # Perform feature selection
    data_selected = feature_selection(data)


if __name__ == "__main__":
    main()

