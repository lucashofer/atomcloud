def create_spence(npt):
    """
    Create a function that calculates the spence function using either
    scipy.special.spence or a jaxified version of the same function. I
    ported the Cephes library algorithm to jax, but which is equivalent for
    real numbers.

    Args:
        npt: numpy or jax

    Returns:
        spence function

    """

    def _spence_poly(w):
        A = npt.array(
            [
                4.65128586073990045278e-5,
                7.31589045238094711071e-3,
                1.33847639578309018650e-1,
                8.79691311754530315341e-1,
                2.71149851196553469920e0,
                4.25697156008121755724e0,
                3.29771340985225106936e0,
                1.00000000000000000126e0,
            ]
        )

        B = npt.array(
            [
                6.90990488912553276999e-4,
                2.54043763932544379113e-2,
                2.82974860602568089943e-1,
                1.41172597751831069617e0,
                3.63800533345137075418e0,
                5.03278880143316990390e0,
                3.54771340985225096217e0,
                9.99999999999999998740e-1,
            ]
        )
        return -w * npt.polyval(A, w) / npt.polyval(B, w)

    def _spence_calc(x):
        x2_bool = x > 2.0
        x = npt.piecewise(x, [x2_bool], [lambda x: 1.0 / x, lambda x: x])

        x1_5_bool = x > 1.5
        x_5_bool = x < 0.5
        x2_bool = x2_bool | x1_5_bool

        w_conds = [x1_5_bool, x_5_bool]
        w_funcs = [lambda x: 1.0 / x - 1.0, lambda x: -x, lambda x: x - 1.0]

        w = npt.piecewise(x, w_conds, w_funcs)
        y = _spence_poly(w)
        y_flag_one = npt.pi**2 / 6.0 - npt.log(x) * npt.log(1.0 - x) - y
        y = npt.where(x_5_bool, y_flag_one, y)
        y_flag_two = -0.5 * npt.log(x) ** 2 - y
        y = npt.where(x2_bool, y_flag_two, y)
        return y

    def spence(x):
        condlist = [x < 0.0, x == 1.0, x == 0.0]
        funclist = [npt.nan, 0, npt.pi**2 / 6, _spence_calc]
        return npt.piecewise(x, condlist, funclist)

    return spence


def create_polylog2d(npt):
    spence = create_spence(npt)

    def polylog2d(z):
        return spence(1 - z)

    return polylog2d
