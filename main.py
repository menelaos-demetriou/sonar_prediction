import pandas as pd
import pickle

from sklearn import metrics
import keras
from keras.layers import Dense
from keras import backend as K
from keras.optimizers import SGD
from keras.layers import Dropout
from keras.models import Sequential
from sklearn.decomposition import PCA
from sklearn.feature_selection import RFECV
from ann_visualizer.visualize import ann_viz
from keras.utils.vis_utils import plot_model
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split
from keras.wrappers.scikit_learn import KerasClassifier

import matplotlib.pyplot as plt
plt.style.use('ggplot')

GRID = True


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
        return pd.concat([X, pd.Series(y, name="target")], axis=1)
    elif type == "pca":
        pca = PCA(n_components=0.95)
        x_reduced = pca.fit_transform(X)
        return pd.concat([pd.DataFrame(x_reduced), pd.Series(y, name="target")], axis=1)


def create_model(num, dropout_rate=0.1, neurons1=10, neurons2=5, init_mode='uniform', learn_rate=0.01, momentum=0,
                 optimizer='adam'):
    # create model
    K.clear_session()
    model = Sequential()
    model.add(Dense(neurons1, input_dim=num, kernel_initializer=init_mode, activation='relu'))
    model.add(Dropout(dropout_rate))
    model.add(Dense(neurons2, kernel_initializer=init_mode, activation='relu'))
    model.add(Dense(1, kernel_initializer=init_mode, activation='sigmoid'))
    optimizer = SGD(lr=learn_rate, momentum=momentum)
    model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    return model


def main():
    # Read file with dataset
    data = pd.read_csv("data/sonar.all-data", header=None)
    data = data.dropna()

    # Perform feature selection
    # data_selected = feature_selection(data, "rfe")
    final_data = feature_selection(data, "pca")

    # Create model
    ann_model = create_model(num=len(final_data.columns)-1)
    early_stopping = keras.callbacks.EarlyStopping(
        verbose=1,
        patience=10,
        mode='max',
        restore_best_weights=True)

    model = KerasClassifier(build_fn=create_model, num=len(final_data.columns)-1, verbose=1)

    ann_model.summary()
    plot_model(ann_model, to_file='plots/model_plot.png', show_shapes=True, show_layer_names=True)
    ann_viz(ann_model, filename="plots/neural_network.png", title="NN Architecture")

    # Split train and test set
    X = final_data.loc[:, ~final_data.columns.isin(["target"])]
    y = final_data["target"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

    # Setting a dictionary with hyperparameters
    epochs = [200]
    optimizer = ['RMSprop', 'Adam', 'Nadam']
    learn_rate = [0.0001, 0.001]
    momentum = [0.2, 0.4, 0.6, 0.8, 0.9]
    init_mode = ['lecun_uniform', 'glorot_normal', 'glorot_uniform', 'he_normal', 'he_uniform']
    dropout_rate = [0.3, 0.4, 0.5]
    neurons1 = [15, 20]
    neurons2 = [15, 20]

    param_grid = dict(epochs=epochs, optimizer=optimizer, learn_rate=learn_rate, momentum=momentum,
                      init_mode=init_mode, dropout_rate=dropout_rate, neurons1=neurons1, neurons2=neurons2)

    if GRID:
        grid = GridSearchCV(estimator=model, param_grid=param_grid, cv=5)
        grid_result = grid.fit(X_train, y_train, **{'callbacks': [early_stopping]})

        # summarize results
        print("Best score: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
        means = grid_result.cv_results_['mean_test_score']
        stds = grid_result.cv_results_['std_test_score']
        params = grid_result.cv_results_['params']
        for mean, stdev, param in zip(means, stds, params):
            print("%f (%f) with: %r" % (mean, stdev, param))

        print("Saving best estimator for evaluations on test dataset")
        filename = "model/best_estimator.sav"
        pickle.dump(grid_result.best_estimator_, open(filename, 'wb'))

        y_pred_class = grid_result.best_estimator_.predict(X_test)
        print("Accuracy on the test set ", metrics.accuracy_score(y_test, y_pred_class))
    else:
        kfold = StratifiedKFold(n_splits=5, random_state=1)
        results = cross_val_score(model, X_train, y_train, cv=kfold)

        print(results.mean())


if __name__ == "__main__":
    main()

