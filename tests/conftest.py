import matplotlib

# plot_spectra imports pyplot lazily, so the backend has to be pinned before the suite
# reaches it. Agg is headless: no window, no display required on CI.
matplotlib.use("Agg")
