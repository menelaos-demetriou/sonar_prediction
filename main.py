import pandas as pd

from keras.layers import Dense
from keras.models import Sequential
from sklearn.decomposition import PCA
from sklearn.feature_selection import RFECV
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split
from keras.wrappers.scikit_learn import KerasClassifier

import matplotlib.pyplot as plt
plt.style.use('ggplot')


def feature_selection(data, type):
    y = data[60].copy()
    X = data.loc[:, ~data.columns.isin([60])].copy()

    enc = LabelEncoder()
    y = enc.fit_transform(y)

    if type == "rfe":
        clf = LogisticRegression(max_iter=1000)

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

        X = X.loc[:, list(rfecv.support_)]
        # Return dataframe with best features and target
        return pd.concat([X, pd.Series(y, name="target")], axis = 1)
    elif type == "pca":
        pca = PCA(n_components=0.95)
        x_reduced = pca.fit_transform(X)
        return pd.concat([pd.DataFrame(x_reduced), pd.Series(y, name="target")], axis = 1)


def create_model(num):
    # create model
    model = Sequential()
    model.add(Dense(num, input_dim=num, activation='relu'))
    model.add(Dense(int(num/2), activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

def main():
    # Read file with dataset
    data = pd.read_csv("data/sonar.all-data", header=None)
    data = data.dropna()

    # Perform feature selection
    # data_selected = feature_selection(data, "rfe")
    final_data = feature_selection(data, "pca")

    # Create model
    model = KerasClassifier(build_fn=create_model, num=len(final_data.columns)-1, epochs=150, verbose=1)

    # Split train and test set
    X = final_data.loc[:, ~final_data.columns.isin(["target"])]
    y = final_data["target"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

    kfold = StratifiedKFold(n_splits=5, random_state=1)
    results = cross_val_score(model, X_train, y_train, cv=kfold)

    print(results.mean())


if __name__ == "__main__":
    main()

