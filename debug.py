from sympy.parsing.mathematica import parse_mathematica


def n1(
    n,
    Sigma12,
    Sigma21,
    Sigma13,
    Sigma31,
    Sigma24,
    Sigma35,
    Tau21,
    Tau32,
    Tau43,
    Tau54,
):
    n * (
        Sigma21 * (Sigma31 * Tau21 * Tau32 + Tau21)
        + Sigma31 * (Sigma24 * Tau21 * Tau32 + Tau32)
        + 1
    ) / (
        Sigma12
        * Tau21
        * (
            Sigma24
            * (
                Tau32 * (Sigma31 * Tau43 + Sigma35 * Tau43 + Sigma35 * Tau54 + 1)
                + Tau43
            )
            + Sigma31 * Tau32
            + 1
        )
        + Sigma13
        * (
            Sigma21 * Tau21 * (Sigma35 * Tau32 * (Tau43 + Tau54) + Tau32)
            + Sigma24 * Tau21 * (Sigma35 * Tau32 * (Tau43 + Tau54) + Tau32 + Tau43)
            + Sigma35 * Tau32 * (Tau43 + Tau54)
            + Tau21
            + Tau32
        )
        + Sigma21 * Sigma31 * Tau21 * Tau32
        + Sigma21 * Tau21
        + Sigma24 * Sigma31 * Tau21 * Tau32
        + Sigma31 * Tau32
        + 1
    )


def n2(
    n,
    Sigma12,
    Sigma21,
    Sigma13,
    Sigma31,
    Sigma24,
    Sigma35,
    Tau21,
    Tau32,
    Tau43,
    Tau54,
):
    Tau21 * n * (Sigma12 * Sigma31 * Tau32 + Sigma12 + Sigma13) / (
        Sigma12
        * Tau21
        * (
            Sigma24
            * (
                Tau32 * (Sigma31 * Tau43 + Sigma35 * Tau43 + Sigma35 * Tau54 + 1)
                + Tau43
            )
            + Sigma31 * Tau32
            + 1
        )
        + Sigma13
        * (
            Sigma21 * Tau21 * (Sigma35 * Tau32 * (Tau43 + Tau54) + Tau32)
            + Sigma24 * Tau21 * (Sigma35 * Tau32 * (Tau43 + Tau54) + Tau32 + Tau43)
            + Sigma35 * Tau32 * (Tau43 + Tau54)
            + Tau21
            + Tau32
        )
        + Sigma21 * Sigma31 * Tau21 * Tau32
        + Sigma21 * Tau21
        + Sigma24 * Sigma31 * Tau21 * Tau32
        + Sigma31 * Tau32
        + 1
    )


def n3(
    n,
    Sigma12,
    Sigma21,
    Sigma13,
    Sigma31,
    Sigma24,
    Sigma35,
    Tau21,
    Tau32,
    Tau43,
    Tau54,
):
    Tau32 * n * (
        Sigma12 * Sigma24 * Tau21
        + Sigma13 * Sigma21 * Tau21
        + Sigma13 * Sigma24 * Tau21
        + Sigma13
    ) / (
        Sigma12
        * Tau21
        * (
            Sigma24
            * (
                Tau32 * (Sigma31 * Tau43 + Sigma35 * Tau43 + Sigma35 * Tau54 + 1)
                + Tau43
            )
            + Sigma31 * Tau32
            + 1
        )
        + Sigma13
        * (
            Sigma21 * Tau21 * (Sigma35 * Tau32 * (Tau43 + Tau54) + Tau32)
            + Sigma24 * Tau21 * (Sigma35 * Tau32 * (Tau43 + Tau54) + Tau32 + Tau43)
            + Sigma35 * Tau32 * (Tau43 + Tau54)
            + Tau21
            + Tau32
        )
        + Sigma21 * Sigma31 * Tau21 * Tau32
        + Sigma21 * Tau21
        + Sigma24 * Sigma31 * Tau21 * Tau32
        + Sigma31 * Tau32
        + 1
    )
