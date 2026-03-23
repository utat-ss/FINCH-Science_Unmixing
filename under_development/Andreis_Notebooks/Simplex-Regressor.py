"""
- Andrei
"""


import marimo

__generated_with = "0.19.4"
app = marimo.App(width="medium")


@app.cell
def _():
    import spectral_library
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import mpltern
    return np, plt, spectral_library


@app.cell
def _(spectral_library):
    df = spectral_library.open_file('unmixing/simpler_data.csv')
    fractions, nr900to1700, _ = spectral_library.take_subset(df, 900, 1700)
    return fractions, nr900to1700


@app.cell
def _(np):
    def calculate_angle(npv,gv,soil):
        return sum(
            0 * npv,
            2*np.pi/3 * gv,
            4*np.pi/3 * soil
        )
    def calculate_distance(npv,gv,soil):
        return (npv**2 + gv**2 + soil**2) ** (1/2)
    return (calculate_distance,)


@app.cell
def _(fractions):
    np_fractions = fractions.to_numpy()
    return (np_fractions,)


@app.cell
def _(calculate_distance, np_fractions, nr900to1700, plt):
    def _():
        wavelength = nr900to1700['1400'].to_numpy()

        ax = plt.subplot()
        ax.scatter(wavelength, [calculate_distance(f[0],f[1],f[2]) for f in np_fractions])
        return plt.show()


    _()
    return


@app.cell
def _(plt):
    ax = plt.subplot(projection="ternary")

    ax.scatter([0.5],[0.25],[0.25])

    ax.set_tlabel("npv")
    ax.set_llabel("gv")
    ax.set_rlabel("soil")

    plt.show()
    return


if __name__ == "__main__":
    app.run()
