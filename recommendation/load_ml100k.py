# Charpter 8 recommendation from Building Machine Learning Systems With Python
# building a recommendation systems based on customer movie ratings using Movielens ML-100k data.
# Eric Liao @2017

def load_data():
    """
    Load ML-100k data
    :return: review matrix as a numpy array
    """

    import numpy as np
    from scipy import sparse
    from os import path

    if not path.exists('data/ml-100k/u.data'):
        raise IOError("Data has not been downloaded.\nTry the following:\n\n\tcd data\n\t./download.sh")

    # The input is in the form of a CSC sparse matrix, so it's a natural fit to
    # load the data, but we then convert to a more traditional array before
    # returning
    data = np.loadtxt('data/ml-100k/u.data')
    # print("Print data:")
    # print(str(data))
    ij = data[:, :2]
    # print("Print ij:")
    # print(str(ij[:11, :]))

    ij -= 1 # original data is in 1-based system, change it to 0-based


    values = data[:, 2]
    # print("Print values:")
    # print(str(values))

    reviews = sparse.csc_matrix((values, ij.T)).astype(float)

    return reviews.toarray()



def get_train_test(reviews=None, random_state=None):
    """
    Prepare data into training and testing
    :param reviews: ndarray, optional
    :param random_state:
    :return: train : ndarray, training data; test : ndarray
        testing data
    """
    import numpy as np
    import random
    r = random.Random(random_state)

    if reviews is None:
        reviews = load_data()

    print("Print 'reviews':")
    print(reviews)
    print('Shape of reviews data is {}'.format(reviews.shape))

    U, M = np.where(reviews)
    print('U is {}, \nM is {}'.format(U, M))
    print('length U is {}, \nlength M is {}'.format(len(U), len(M)))
    test_idx = np.array(r.sample(range(len(U)), len(U) // 10)) # choose 10% of data as testing

    # build the train matrix, which is like reviews, but with the testing entries set to zero:
    train = reviews.copy()
    train[U[test_idx], M[test_idx]] = 0

    # the test matrix contains just the testing values
    test = np.zeros_like(reviews)
    test[U[test_idx], M[test_idx]] = reviews[U[test_idx], M[test_idx]]

    return train, test

if __name__ == '__main__':

    reviews = load_data()
    train, test = get_train_test(reviews)

    # print("Print reviews:")
    # print(reviews[:10, :])

    print('Train data is \n{}, size is {}, \nTest data is \n{}, size is {}'.format(train, train.shape, test, test.shape))





