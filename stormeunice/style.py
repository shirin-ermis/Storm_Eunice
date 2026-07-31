"""Shared figure style for the Storm Babet paper figures.

Every ``PAPER*.ipynb`` notebook pulls its fonts, panel-label sizing, method
colours and save settings from here, so that the published figures read as one
visual system.

The reference figure is PAPER1 (``figures/PAPER1_attribution_methods_figure``):
a 7.2 in wide canvas with 12 pt bold panel labels on a ``cmcrameri.bamako``
method ramp. Everything below is expressed relative to that figure.

Typical use in a notebook::

    import babet as bb

    bb.style.set_style()
    ...
    bb.style.panel_labels(axes_with_data)          # a, b, c, ...
    bb.style.save(fig, 'PAPER3_map_plots')
"""
import os

import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import to_hex
from cmcrameri import cm

# --- typeface ---------------------------------------------------------------
FONT = "Nimbus Sans"          # Helvetica-metric clone; "Arial" also works
FONT_STACK = [FONT, "Helvetica", "Arial", "DejaVu Sans"]

INK = "#1A1A1A"
MUTED = "#5C5C60"

# --- type sizes -------------------------------------------------------------
# Quoted in points at REF_WIDTH. Figures drawn on a wider canvas are scaled to
# column width in the paper, so their font sizes have to be scaled up by the
# same factor for the printed type size to come out identical. Use scaled().
REF_WIDTH = 7.2               # inches, the PAPER1 canvas width

FS_PANEL = 12.0               # subplot labels a, b, c ...
FS_HEADER = 9.4               # column / row headers
FS_TITLE = 9.5                # panel titles
FS_LABEL = 9.0                # axis and colourbar labels
FS_TICK = 8.0                 # tick labels
FS_ANNOT = 8.5                # in-plot annotation and legends

# Dense multi-panel grids (PAPER3's 7x4 of maps) cannot carry type at the sizes
# above without headers running into each other. These are for those figures.
FS_SMALL = 7.0                # headers and labels on a dense grid
FS_XSMALL = 6.0               # tick labels on a dense grid

# --- method colours ---------------------------------------------------------
# One ramp for all attribution methods, sampled at the position each method
# family occupies on the conditioning axis of the PAPER1 figure.
# RAMP = sns.color_palette("Greens_d", as_cmap=True)
# RAMP = sns.color_palette(palette='cividis', as_cmap=True)
RAMP = cm.bamako               # cmcrameri: dark teal-green (0) -> light sand (1)

# Span of the conditioning axis (0..1) each method family covers in the PAPER1
# spectrum. This is figure geometry only -- it does not set the colours.
METHOD_RANGE = {
    "probabilistic": (0.020, 0.265),
    "flow": (0.290, 0.655),
    "forecast": (0.605, 0.850),
    "pgw": (0.680, 0.925),
}

# Colours are sampled at equal spacing along the ramp rather than at the centre
# of each span, so that the four families stay distinguishable where colour is
# the only thing telling them apart (y tick labels, column headers).
METHOD_ORDER = ["probabilistic", "flow", "forecast", "pgw"]
METHOD_POSITION = {k: (i + 0.5) / len(METHOD_ORDER)
                   for i, k in enumerate(METHOD_ORDER)}

METHOD_COLOR = {k: to_hex(RAMP(v)) for k, v in METHOD_POSITION.items()}

# The light end of the ramp is fine as a fill but too pale to read as text on
# white. Where the method colour is carried by type -- y tick labels, column
# headers -- sample the same ramp over its darker half instead, keeping the
# four families in the same order and equally spaced.
#
# bamako runs dark -> light, so its darker half is the *low* end and the range
# below runs downwards. The upper bound stops at 0.5 (#637a0a), the palest
# point still clearing 4.5:1 contrast on white.
TEXT_RAMP_RANGE = (0.0, 0.5)


def _to_text_position(p):
    lo, hi = TEXT_RAMP_RANGE
    span = max(METHOD_POSITION.values()) - min(METHOD_POSITION.values())
    t = (p - min(METHOD_POSITION.values())) / span
    return lo + t * (hi - lo)


METHOD_TEXT_POSITION = {k: _to_text_position(v) for k, v in METHOD_POSITION.items()}
METHOD_TEXT_COLOR = {k: to_hex(RAMP(v)) for k, v in METHOD_TEXT_POSITION.items()}


def shade(x, factor=0.62):
    """Darker version of the ramp colour at position x (for borders)."""
    r, g, b = RAMP(x)[:3]
    return (r * factor, g * factor, b * factor)


def tint(x, alpha=0.16):
    """Very light version of the ramp colour (for caveated chips)."""
    r, g, b = RAMP(x)[:3]
    return tuple(1 - alpha * (1 - v) for v in (r, g, b))


def ink_on(x):
    """Readable text colour on top of the ramp colour at position x."""
    r, g, b = RAMP(x)[:3]
    lum = 0.2126 * r + 0.7152 * g + 0.0722 * b
    return INK if lum > 0.55 else "white"


def method_family(name):
    """Map a method label used in the papers onto one of the four families.

    Returns None for labels that are not an attribution method (e.g. the
    "ERA5 event" column showing the observed storm).
    """
    n = " ".join(str(name).lower().replace(",", " ").replace("-", " ").split())
    if "probabilistic" in n:
        return "probabilistic"
    if "pgw" in n or "pseudo" in n:
        return "pgw"
    if "fba" in n or "forecast" in n or "ifs" in n or "micas" in n:
        return "forecast"
    if "analogue" in n or "racmo" in n:
        return "flow"
    # "ERA5 analogues" is caught above; a bare "ERA5" column means the same
    # thing, but "ERA5 event" is the observed storm rather than a method.
    if "era5" in n and "event" not in n:
        return "flow"
    return None


def method_color(name, default=INK):
    """Fill colour for a method label; `default` if the label is not a method.

    Use this for patches and swatches. For coloured type use method_text_color.
    """
    family = method_family(name)
    return default if family is None else METHOD_COLOR[family]


def method_colors(names, default=INK):
    """Fill colours for a sequence of method labels, in the order given."""
    return [method_color(n, default=default) for n in names]


def method_text_color(name, default=INK):
    """Colour for a method name set as type -- dark enough to read on white."""
    family = method_family(name)
    return default if family is None else METHOD_TEXT_COLOR[family]


def method_text_colors(names, default=INK):
    """Text colours for a sequence of method labels, in the order given."""
    return [method_text_color(n, default=default) for n in names]


# --- climate scenario colours -----------------------------------------------
# The scenarios are drawn from moarpalettes' ``classic_tab20`` (indices 0, 1,
# 14, 2, 6), which is the classic matplotlib tab20 at full saturation. On the
# page that reads brighter than the palettes in the moarpalettes reference --
# those are the *new* Tableau ramps, which are muted to begin with. Rather than
# change hue, pull the saturation down with seaborn's desaturate, which is the
# same knob as the ``desat`` argument of ``sns.color_palette``.
#
# SCENARIO_DESAT is the one number to turn: 1.0 is the raw palette, 0.0 is
# grey. 0.7 is roughly the weight of the new Tableau ramps.
SCENARIO_DESAT = 0.7

SCENARIO_BASE = {
    "1870": "#1f77b4",        # classic_tab20[0]   blue
    "1950": "#aec7e8",        # classic_tab20[1]   light blue
    "present": "#7f7f7f",     # classic_tab20[14]  grey
    "future1": "#ff7f0e",     # classic_tab20[2]   orange
    "future2": "#d62728",     # classic_tab20[6]   red
}

CLIMATE_COLOR = {k: to_hex(sns.desaturate(v, SCENARIO_DESAT))
                 for k, v in SCENARIO_BASE.items()}


def climate_color(name, default=INK):
    """Colour for a climate scenario label ('1870', '1950', 'present', ...)."""
    return CLIMATE_COLOR.get(str(name), default)


def climate_colors(names, default=INK):
    """Colours for a sequence of scenario labels, in the order given."""
    return [climate_color(n, default=default) for n in names]


# --- accent colours ---------------------------------------------------------
# For anything that is neither a method nor a scenario: observations drawn over
# a model ensemble, a threshold or reference line, a highlighted member.
#
# The scenarios take blue, grey, orange and red, and the bamako ramp runs
# across the whole teal-green -> olive -> sand band, which leaves the violet
# end of the wheel as the only hue nothing else is using. The three below sit
# there and are separated from each other mainly by lightness, so they stay
# apart in greyscale as well as in colour.
#
# Chosen by maximising the smallest CIELab distance to the scenario colours and
# to 21 samples along RAMP, under normal vision and under simulated deutan,
# protan and tritan vision: worst case dE 17 to anything already in use, dE 18
# between the accents themselves. All three clear 4.5:1 on white, so they can
# carry type as well as lines.
ACCENT_ORDER = ["violet", "orchid", "indigo"]

ACCENT_COLOR = {
    "violet": "#4A1C6E",      # deep plum; darkest, reads almost as ink
    "orchid": "#9B57C9",      # light; the one to use on a dark fill
    "indigo": "#3A32A8",      # blue-violet; clear of the 1870 blue
    "grey": "#8b8b8b",        # classic_tab20[14]  grey; same as "present"
}


def accent(name, default=INK):
    """Accent colour by name ('violet', 'orchid', 'indigo') or by index."""
    if isinstance(name, int):
        return ACCENT_COLOR[ACCENT_ORDER[name % len(ACCENT_ORDER)]]
    return ACCENT_COLOR.get(str(name), default)


def accents(n=None):
    """The accent colours in order; the first `n` of them if `n` is given."""
    colors = [ACCENT_COLOR[k] for k in ACCENT_ORDER]
    return colors if n is None else [accent(i) for i in range(n)]


# --- sizing helpers ---------------------------------------------------------
def scaled(size, fig=None):
    """Convert a size quoted at REF_WIDTH to points on `fig`'s canvas.

    A label drawn at ``scaled(FS_PANEL, fig)`` prints at FS_PANEL points once
    the figure is scaled to column width, whatever the canvas size.
    """
    fig = plt.gcf() if fig is None else fig
    return size * fig.get_figwidth() / REF_WIDTH


def set_style():
    """Apply the shared rcParams. Call once, at the top of a notebook."""
    sns.set_theme(style="white")
    sns.set_style("white")
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": FONT_STACK,
        "pdf.fonttype": 42,        # editable text in Illustrator
        "ps.fonttype": 42,
        "svg.fonttype": "none",
        "text.color": INK,
        "axes.edgecolor": INK,
        "axes.labelcolor": INK,
        "xtick.color": INK,
        "ytick.color": INK,
        "figure.facecolor": "white",
        "savefig.facecolor": "white",
        "savefig.dpi": 600,
        "savefig.bbox": "tight",
    })


# --- panel labels -----------------------------------------------------------
LETTERS = "abcdefghijklmnopqrstuvwxyz"


def panel_label(ax, letter, x=0.03, y=0.95, color=INK, zorder=20, **kwargs):
    """Draw a bold subplot label in the top-left corner of `ax`.

    The font size is scaled to the figure canvas so that every panel label in
    the paper prints at the same size. Pass ``color='white'`` where the label
    would otherwise sit on a dark fill.
    """
    fig = ax.get_figure()
    kwargs.setdefault("ha", "left")
    kwargs.setdefault("va", "top")
    return ax.text(x, y, letter, transform=ax.transAxes,
                   fontsize=scaled(FS_PANEL, fig), fontweight="bold",
                   color=color, zorder=zorder, **kwargs)


def panel_labels(axes, letters=None, colors=None, **kwargs):
    """Label a sequence of axes a, b, c, ... in the order given.

    `letters` overrides the default sequence (e.g. to continue at "m" in a
    second panel of the same figure). `colors` is an optional per-axis colour,
    for panels whose corner sits on a dark fill.
    """
    axes = list(axes)
    letters = LETTERS[:len(axes)] if letters is None else letters
    if colors is None:
        colors = [INK] * len(axes)
    return [panel_label(ax, letter, color=color, **kwargs)
            for ax, letter, color in zip(axes, letters, colors)]


# --- saving -----------------------------------------------------------------
def save(fig, stem, directory="../figures", formats=("png", "pdf"), **kwargs):
    """Save `fig` as ``<directory>/<stem>.<fmt>`` for each format.

    Uses the shared dpi and bounding box so every figure lands in the paper
    with the same margins.
    """
    kwargs.setdefault("dpi", 600)
    kwargs.setdefault("bbox_inches", "tight")
    kwargs.setdefault("facecolor", "white")
    paths = []
    for fmt in formats:
        path = os.path.join(directory, f"{stem}.{fmt}")
        fig.savefig(path, **kwargs)
        paths.append(path)
    return paths
