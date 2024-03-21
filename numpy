задача 1 
def matrix_vector_product_sum(X, V):
    return np.sum([np.dot(x, v) for x, v in zip(X, V)], axis=0)


def test_sum_prod():
    X = [np.array([[1, 2], [3, 4]]), np.array([[5, 6], [7, 8]])]
    V = [np.array([1, 2]), np.array([3, 4])]
    expected_result = np.array([44, 56])
    assert np.array_equal(sum_prod(X, V), expected_result)

    X = [np.array([[1, 0], [0, 1]]), np.array([[2, 0], [0, 2]])]
    V = [np.array([1, 1]), np.array([2, 2])]
    expected_result = np.array([6, 12])
    assert np.array_equal(sum_prod(X, V), expected_result)

    X = []
    V = []
    expected_result = np.array([])
    assert np.array_equal(sum_prod(X, V), expected_result)

    print("All tests successful")

test_sum_prod()



задача 2

import numpy as np

def binarize_matrix(M, threshold=0.5):
    return (M > threshold).astype(int)

def test_binarize_matrix():
    M = np.array([[0.1, 0.6, 0.7],
                  [0.4, 0.8, 0.2]])
    threshold = 0.5
    expected_result = np.array([[0, 1, 1],
                                [0, 1, 0]])
    assert np.array_equal(binarize_matrix(M), expected_result)

    M = np.array([[0.1, 0.6, 0.7],
                  [0.4, 0.8, 0.2]])
    threshold = 0.7
    expected_result = np.array([[0, 0, 1],
                                [0, 1, 0]])
    assert np.array_equal(binarize_matrix(M, threshold), expected_result)

    M = np.array([])
    threshold = 0.5
    expected_result = np.array([])
    assert np.array_equal(binarize_matrix(M), expected_result)

    print("All tests successful")

test_binarize_matrix()


задача 3 -


задача 4


def show_axis_info(axis_count, matrix):
    for axis in range(axis_count):
        mean = np.mean(matrix[axis])
        std_dev = np.std(matrix[axis])
        plt.hist(matrix[axis])
        plt.xlabel('Value')
        plt.ylabel('Frequency')
        plt.title(f'Histogram for axis {axis+1}, mean: {mean:.2f}, std: {std_dev:.2f}')
        plt.show()

def show_matrix_random_normal_info(m, n):
    random = np.random.default_rng()
    matrix = random.normal(size=(m, n))
    show_axis_info(m, matrix)
    show_axis_info(n, matrix.T)


show_matrix_random_normal_info(3, 4)



задача 5


def chess_matrix(m, n, a, b):
    matrix = np.zeros((m, n))

    for i in range(m):
        for j in range(n):
            if (i + j) % 2 == 0:
                matrix[i][j] = a
            else:
                matrix[i][j] = b

    return matrix

def test_chessboard_matrix():
  
    assert np.array_equal(chess_matrix(3, 3, 0, 1), np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]]))
 
    assert np.array_equal(chess_matrix(2, 4, 3, 8), np.array([[3, 8, 3, 8], [8, 3, 8, 3]]))
   
    assert np.array_equal(chess_matrix(4, 2, -1, 2), np.array([[-1, 2], [2, -1], [-1, 2], [2, -1]]))

    print("All tests successful")

test_chessboard_matrix()

задача 7 




import numpy as np

def compute_statistics(time_series):
    mean = np.mean(time_series)
    variance = np.var(time_series)
    std_deviation = np.std(time_series)
    return mean, variance, std_deviation

time_series = [1, 2, 3, 4, 5]
mean, variance, std_deviation = compute_statistics(time_series)
print("Mean:", mean)
print("Variance:", variance)
print("Standard Deviation:", std_deviation)

def test_compute_statistics():
    time_series_1 = [1, 2, 3, 4, 5]
    mean_1, variance_1, std_deviation_1 = compute_statistics(time_series_1)
    assert mean_1 == 3.0
    assert variance_1 == 2.0
    assert std_deviation_1 == np.sqrt(2.0)

    time_series_2 = [10, 20, 30, 40, 50]
    mean_2, variance_2, std_deviation_2 = compute_statistics(time_series_2)
    assert mean_2 == 30.0
    assert variance_2 == 200.0
    assert std_deviation_2 == np.sqrt(200.0)

    time_series_3 = [-1, -2, -3, -4, -5]
    mean_3, variance_3, std_deviation_3 = compute_statistics(time_series_3)
    assert mean_3 == -3.0
    assert variance_3 == 2.0
    assert std_deviation_3 == np.sqrt(2.0)

    print("All tests successful")

test_compute_statistics()



задача 8



def one_hot_encoding(labels, num_classes):
    num_samples = len(labels)
    encoded_labels = np.zeros((num_samples, num_classes), dtype=int)
    for i in range(num_samples):
        label = labels[i]
        encoded_labels[i, label] = 1
    return encoded_labels

def test_one_hot_encoding():
    labels = [0, 2, 3, 0]
    num_classes = 4
    expected_result = np.array([[1, 0, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1], [1, 0, 0, 0]])
    assert np.array_equal(one_hot_encoding(labels, num_classes), expected_result)

    labels = [2, 1, 3]
    num_classes = 4
    expected_result = np.array([[0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1]])
    assert np.array_equal(one_hot_encoding(labels, num_classes), expected_result)

    labels = [0]
    num_classes = 1
    expected_result = np.array([[1]])
    assert np.array_equal(one_hot_encoding(labels, num_classes), expected_result)

    print("All tests successful")

test_one_hot_encoding()
