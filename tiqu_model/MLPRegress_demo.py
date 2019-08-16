
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from sklearn.inspection import partial_dependence
from sklearn.inspection import plot_partial_dependence
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.datasets.california_housing import fetch_california_housing


def main():
    cal_housing = fetch_california_housing()

    X, y = cal_housing.data, cal_housing.target
    names = cal_housing.feature_names

    # Center target to avoid gradient boosting init bias: gradient boosting
    # with the 'recursion' method does not account for the initial estimator
    # (here the average target, by default)
    y -= y.mean()

    print("Training MLPRegressor...")
    est = MLPRegressor(activation='logistic')
    est.fit(X, y)
    print('Computing partial dependence plots...')
    # We don't compute the 2-way PDP (5, 1) here, because it is a lot slower
    # with the brute method.
    features = [0, 5, 1, 2]
    plot_partial_dependence(est, X, features, feature_names=names,
                            n_jobs=3, grid_resolution=50)
    fig = plt.gcf()
    fig.suptitle('Partial dependence of house value on non-location features\n'
                 'for the California housing dataset, with MLPRegressor')
    plt.subplots_adjust(top=0.9)  # tight_layout causes overlap with suptitle

    print("Training GradientBoostingRegressor...")
    est = GradientBoostingRegressor(n_estimators=100, max_depth=4,
                                    learning_rate=0.1, loss='huber',
                                    random_state=1)
    est.fit(X, y)
    print('Computing partial dependence plots...')
    features = [0, 5, 1, 2, (5, 1)]
    plot_partial_dependence(est, X, features, feature_names=names,
                            n_jobs=3, grid_resolution=50)
    fig = plt.gcf()
    fig.suptitle('Partial dependence of house value on non-location features\n'
                 'for the California housing dataset, with Gradient Boosting')
    plt.subplots_adjust(top=0.9)

    print('Custom 3d plot via ``partial_dependence``')
    fig = plt.figure()

    target_feature = (1, 5)
    pdp, axes = partial_dependence(est, X, target_feature,
                                   grid_resolution=50)
    XX, YY = np.meshgrid(axes[0], axes[1])
    Z = pdp[0].T
    ax = Axes3D(fig)
    surf = ax.plot_surface(XX, YY, Z, rstride=1, cstride=1,
                           cmap=plt.cm.BuPu, edgecolor='k')
    ax.set_xlabel(names[target_feature[0]])
    ax.set_ylabel(names[target_feature[1]])
    ax.set_zlabel('Partial dependence')
    #  pretty init view
    ax.view_init(elev=22, azim=122)
    plt.colorbar(surf)
    plt.suptitle('Partial dependence of house value on median\n'
                 'age and average occupancy, with Gradient Boosting')
    plt.subplots_adjust(top=0.9)

    plt.show()


# Needed on Windows because plot_partial_dependence uses multiprocessing
if __name__ == '__main__':
    main()