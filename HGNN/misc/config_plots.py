import matplotlib

def global_settings():
    pgf_with_latex = {                      # setup matplotlib to use latex for output
        "pgf.texsystem": "pdflatex",        # change this if using xetex or lautex
        "text.usetex": True,                # use LaTeX to write all text
        "figure.figsize": (9, 5),     # default fig size of 0.9 textwidth
        "figure.dpi": 300,
        "font.family": "serif",
        "font.serif": [],                   # blank entries should cause plots to inherit fonts from the document
        "font.sans-serif": [],
        "font.monospace": [],
        "axes.labelsize": 25,               # LaTeX default is 10pt font.
        'axes.titlesize': 25,
        "font.size": 25,
        "legend.fontsize": 25,               # Make the legend/label fonts a little smaller
        "xtick.labelsize": 25, #23?
        "ytick.labelsize": 25,
        }

    matplotlib.rcParams.update(pgf_with_latex)