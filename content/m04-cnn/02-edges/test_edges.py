import numpy as np

from m04.conv import conv2d, edge_magnitude, max_pool2d, sobel_x, sobel_y


def test_max_pool():
    x = np.arange(16, dtype=float).reshape(1, 1, 4, 4)
    assert max_pool2d(x).tolist() == [[[[5.0, 7.0], [13.0, 15.0]]]]
    y = np.arange(25, dtype=float).reshape(1, 1, 5, 5)
    assert max_pool2d(y).shape == (1, 1, 2, 2), "the leftover row and column are dropped"
    assert max_pool2d(y, k=3).tolist() == [[[[12.0]]]]


def test_sobel_kernels():
    assert np.array_equal(sobel_x(), [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    assert np.array_equal(sobel_y(), np.array(sobel_x()).T)


def test_vertical_edge():
    img = np.zeros((8, 8))
    img[:, 4:] = 1.0
    gx = conv2d(img[None, None], np.array(sobel_x())[None, None])[0, 0]
    gy = conv2d(img[None, None], np.array(sobel_y())[None, None])[0, 0]
    assert gx.max() == 4.0 and np.allclose(gy, 0), "a vertical edge excites sobel_x only"


def test_stripes_inside_are_invisible_to_the_wrong_kernel():
    img = np.zeros((10, 10))
    img[::3] = 1.0  # horizontal stripes
    gx = conv2d(img[None, None], np.array(sobel_x())[None, None])[0, 0]
    assert np.allclose(gx, 0), "horizontal stripes don't change from left to right"


def test_edge_magnitude():
    img = np.zeros((8, 8))
    img[2:6, 2:6] = 1.0
    m = edge_magnitude(img)
    assert m.shape == (8, 8), "pad by 1 so the output has the image's size"
    assert m[4, 4] == 0 and m[0, 7] == 0, "flat regions have no edges"
    assert m[4, 2] > 0 and m[2, 4] > 0, "both vertical and horizontal edges show up"
    assert np.isclose(m[4, 1], m[1, 4]), "the magnitude doesn't care about the edge's direction"
