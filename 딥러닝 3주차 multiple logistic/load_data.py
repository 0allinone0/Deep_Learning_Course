import gzip
import numpy as np
import matplotlib.pyplot as plt


def load_data(path = '딥러닝 3주차 multiple logistic/dataset/MNIST/{}.gz', n_label=10, is_vis=False):
    mnist_dataset = {}

    # Read Images
    for key in ('training_images', 'test_images'):
        with gzip.open(path.format(key), 'rb') as f:
            mnist_dataset[key] = np.frombuffer(f.read(), np.uint8, offset=16).reshape(-1, 28*28)

    # Read Labels
    for key in ('training_labels', 'test_labels'):
        with gzip.open(path.format(key), 'rb') as f:
            mnist_dataset[key] = np.frombuffer(f.read(), np.uint8, offset=8)

    x_train, y_train, x_test, y_test = (mnist_dataset["training_images"],
                                        mnist_dataset["training_labels"],
                                        mnist_dataset["test_images"],
                                        mnist_dataset["test_labels"])

    # check the shape of data
    print("Total number of data for train {}/{} and for test {}/{}".format(x_train.shape, y_train.shape, x_test.shape, y_test.shape))

    X_train, Y_train, X_test, Y_test = [], [], [], []
    for i in range(n_label):
        # training set
        temp_x = x_train[(y_train == (i))] # get samples corresponding to i class.
        temp_y = y_train[(y_train == (i))] # get samples corresponding to i class.
        X_train.append(temp_x)
        Y_train.append(temp_y)
        # test set
        temp_x = x_test[(y_test == (i))] # get samples corresponding to i class.
        temp_y = y_test[(y_test == (i))] # get samples corresponding to i class.
        X_test.append(temp_x)
        Y_test.append(temp_y)


    # list to numpy array
    X_train = np.concatenate(X_train, 0)
    Y_train = np.concatenate(Y_train, 0).reshape(-1, 1)
    X_test = np.concatenate(X_test, 0)
    Y_test = np.concatenate(Y_test, 0).reshape(-1, 1)
    print("Number of selected data for train {}/{} and for test {}/{}".format(X_train.shape, Y_train.shape, X_test.shape, Y_test.shape))
    print("Sorting classes for trian {} and for test {}".format(np.unique(Y_train), np.unique(Y_test)))
    # visualize the first sample and check its label
    if is_vis:
        plt.imshow(X_train[0].reshape(28, 28))
        plt.xlabel("Number: {}".format(Y_train[0]))
        plt.show()

    return X_train, Y_train, X_test, Y_test


if __name__ == "__main__":
    load_data()