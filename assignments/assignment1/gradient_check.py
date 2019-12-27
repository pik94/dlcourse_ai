import numpy as np


def check_gradient(f, x: np.ndarray, delta: float = 1e-5, tol: float = 1e-4):
    """
    Checks the implementation of analytical gradient by comparing
    it to numerical gradient using two-point formula

    Arguments:
      f: function that receives x and computes value and gradient
      x: np array, initial point where gradient is checked
      delta: step to compute numerical gradient
      tol: tolerance for comparing numerical and analytical gradient

    Return:
      bool indicating whether gradients match or not
    """
    
    assert isinstance(x, np.ndarray)
    assert x.dtype == np.float
    
    orig_x = x.copy()
    fx, analytic_grad = f(x)
    assert np.all(np.isclose(orig_x, x, tol)), \
        "Functions shouldn't modify input variables"

    assert analytic_grad.shape == x.shape
    analytic_grad = analytic_grad.copy()

    # We will go through every dimension of x and compute numeric
    # derivative for it
    it = np.nditer(x, flags=['multi_index'], op_flags=['readwrite'])
    while not it.finished:
        ix = it.multi_index
        analytic_grad_at_ix = analytic_grad[ix]

        # TODO compute value of numeric gradient of f to idx
        value = x[ix].copy()
        x[ix] = value + delta
        fx_r = f(x)[0]
        x[ix] = value - delta
        fx_l = f(x)[0]
        numeric_grad_at_ix = (fx_r - fx_l) / (2*delta)
        x[ix] = value
        if not np.isclose(numeric_grad_at_ix, analytic_grad_at_ix, tol):
            print((f'Gradients are different at {ix}. '
                   f'Analytic: {analytic_grad_at_ix: .5f}, '
                   f'Numeric: {numeric_grad_at_ix: .5f}'))
            return False

        it.iternext()

    print('Gradient check passed!')
    return True
