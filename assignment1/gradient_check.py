import numpy as np


def check_gradient(f, x, delta=1e-5, tol=1e-4):
    '''
    Checks the implementation of analytical gradient by comparing
    it to numerical gradient using two-point formula

    Arguments:
      f: function that receives x and computes value and gradient
      x: np array, initial point where gradient is checked
      delta: step to compute numerical gradient
      tol: tolerance for comparing numerical and analytical gradient

    Return:`
      bool indicating whether gradients match or not
    '''
    
    assert isinstance(x, np.ndarray)
    assert x.dtype == np.float

    orig_x = x.copy()
    fx, analytic_grad = f(x)

    assert np.all(np.isclose(orig_x, x, tol)), "Functions shouldn't modify input variables"

    # Одномерный случай приводим к многомерному
    if len(analytic_grad.shape) == 1:
        analytic_grad = analytic_grad[None, :]
    if len(x.shape) == 1:
        x = x[None, :]

    assert analytic_grad.shape == x.shape
    analytic_grad = analytic_grad.copy()

    # We will go through every dimension of x and compute numeric
    # derivative for it
    it = np.nditer(x, flags=['multi_index'], op_flags=['readwrite'])
    while not it.finished:
        ix = it.multi_index
        analytic_grad_at_ix = analytic_grad[ix]

        # TODO compute value of numeric gradient of f to idx
        x_minus_delta = it.operands[0].copy()
        x_minus_delta[ix] -= delta
        x_plus_delta = it.operands[0].copy()
        x_plus_delta[ix] += delta

        left_point_val, _ = f(x_plus_delta.squeeze())
        right_point_val, _ = f(x_minus_delta.squeeze())
        numeric_grad_at_ix = np.sum(left_point_val - right_point_val) / (2 * delta)

        if not np.isclose(numeric_grad_at_ix, analytic_grad_at_ix, tol):
            print("Gradients are different at %s. Analytic: %2.5f, Numeric: %2.5f" % (ix, analytic_grad_at_ix, numeric_grad_at_ix))
            return False

        it.iternext()

    print("Gradient check passed!")
    return True
