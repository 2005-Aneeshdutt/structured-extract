"""Chart primitives for the README figures.

Design rules encoded here (rather than left to per-chart taste):

* **Categorical hues assigned in fixed slot order, never cycled.** Slot 1 = blue,
  2 = orange, 3 = aqua, 4 = yellow. The ordering is the colorblind-safety
  mechanism -- this exact order was validated for adjacent-pair separation under
  protan/deutan/tritan simulation in both light and dark mode (worst adjacent CVD
  dE 9.1 light / 8.4 dark, against a >= 8 target).
* **Direct value labels on every bar.** Two of the light-mode slots sit below 3:1
  contrast against the light surface, so identity must not rest on color alone;
  labels are the required relief. They also make the PNG readable in a README
  without squinting at an axis.
* **Light AND dark variants.** GitHub renders READMEs in the viewer's theme; a
  single light PNG is unreadable for half the audience. The dark steps are
  separately chosen for the dark surface, not an inverted copy.
* **Recessive chrome.** Hairline horizontal gridlines only, muted tick ink, no
  top/right spines, no chartjunk.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")  # headless: this runs in CI and over SSH
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Patch


@dataclass(frozen=True)
class Theme:
    name: str
    surface: str
    text_primary: str
    text_secondary: str
    muted: str
    grid: str
    baseline: str
    series: tuple[str, ...]


LIGHT = Theme(
    name="light",
    surface="#fcfcfb",
    text_primary="#0b0b0b",
    text_secondary="#52514e",
    muted="#898781",
    grid="#e1e0d9",
    baseline="#c3c2b7",
    series=("#2a78d6", "#eb6834", "#1baf7a", "#eda100"),
)

DARK = Theme(
    name="dark",
    surface="#1a1a19",
    text_primary="#ffffff",
    text_secondary="#c3c2b7",
    muted="#898781",
    grid="#2c2c2a",
    baseline="#383835",
    series=("#3987e5", "#d95926", "#199e70", "#c98500"),
)

THEMES = (LIGHT, DARK)

_FONT = ["DejaVu Sans", "Segoe UI", "Helvetica", "sans-serif"]


def _style(ax: Any, theme: Theme, *, ygrid: bool = True, xgrid: bool = False) -> None:
    ax.set_facecolor(theme.surface)
    ax.figure.set_facecolor(theme.surface)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(theme.baseline)
        ax.spines[side].set_linewidth(1.0)
    ax.tick_params(colors=theme.muted, labelsize=9, length=0)
    for lbl in ax.get_xticklabels() + ax.get_yticklabels():
        lbl.set_color(theme.text_secondary)
    # Hairline grid, behind the marks, on the value axis only. Passing line
    # properties with a False first arg silently *enables* the grid in
    # matplotlib, so the enable flag is checked before styling.
    if ygrid:
        ax.grid(True, axis="y", color=theme.grid, linewidth=0.8)
    if xgrid:
        ax.grid(True, axis="x", color=theme.grid, linewidth=0.8)
    ax.set_axisbelow(True)


def _legend(ax: Any, theme: Theme, labels: Sequence[str], *, ncol: int, y: float) -> None:
    """Legend built from explicit color proxies.

    Necessary because the bars are FancyBboxPatch objects added straight to the
    axes; an empty `ax.bar([], [])` call registers a container with no artist, so
    matplotlib falls back to a default color and every swatch comes out the same
    blue. Explicit Patch handles keep swatch color bound to slot order.
    """
    handles = [Patch(facecolor=theme.series[i % len(theme.series)], edgecolor="none", label=lab)
               for i, lab in enumerate(labels)]
    leg = ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, y),
                    ncol=ncol, frameon=False, fontsize=9.5, handlelength=1.1,
                    handleheight=1.1, columnspacing=1.6)
    for text in leg.get_texts():
        text.set_color(theme.text_secondary)
        text.set_fontfamily(_FONT)


def _rounded_bars(ax: Any, xs: Sequence[float], heights: Sequence[float], width: float,
                  color: str, theme: Theme, horizontal: bool = False) -> None:
    """Bars with rounded data-ends, square against the baseline.

    Implemented with FancyBboxPatch extended *past* the baseline and then clipped
    to the positive side: that is what makes the far end round while the end
    anchored at zero stays flat, which is the anatomy rule (the data-end is
    rounded; the baseline end is not, because a rounded baseline reads as a value
    above zero).

    The radius is a fraction of bar thickness rather than a literal pixel count.
    These PNGs render at 2000px wide and are displayed at ~800px in a README, so
    a hard-coded 4px radius would be invisible at both sizes; scaling with the
    mark keeps the intended weight at any output size.
    """
    ax.figure.canvas.draw()  # a real bbox is needed before aspect can be computed
    bbox = ax.get_window_extent()
    x0, x1 = ax.get_xlim()
    y0, y1 = ax.get_ylim()
    px_per_x = bbox.width / max(x1 - x0, 1e-9)
    px_per_y = bbox.height / max(y1 - y0, 1e-9)

    # strict=True throughout this module: a length mismatch between positions and
    # values means a bar is silently missing from a results chart, which is worse
    # than a crash.
    for x, h in zip(xs, heights, strict=True):
        if horizontal:
            # aspect converts the y-unit radius into an equal-looking x radius
            aspect = (px_per_y / px_per_x) if px_per_x else 1.0
            r_y = min(width * 0.22, abs(h) / max(aspect, 1e-9) * 0.5)
            patch = FancyBboxPatch(
                (0 - r_y * aspect, x - width / 2), abs(h) + r_y * aspect, width,
                boxstyle=f"round,pad=0,rounding_size={r_y}",
                mutation_aspect=1 / aspect if aspect else 1.0,
                linewidth=0, facecolor=color, clip_on=True, zorder=2,
            )
            ax.add_patch(patch)
            patch.set_clip_path(plt.Rectangle((0, y0), x1, y1 - y0,
                                              transform=ax.transData, visible=False))
        else:
            aspect = (px_per_y / px_per_x) if px_per_x else 1.0
            r_x = min(width * 0.22, abs(h) / max(aspect, 1e-9) * 0.5)
            patch = FancyBboxPatch(
                (x - width / 2, 0 - r_x * aspect), width, abs(h) + r_x * aspect,
                boxstyle=f"round,pad=0,rounding_size={r_x}",
                mutation_aspect=aspect,
                linewidth=0, facecolor=color, clip_on=True, zorder=2,
            )
            ax.add_patch(patch)
            patch.set_clip_path(plt.Rectangle((x0, 0), x1 - x0, y1,
                                              transform=ax.transData, visible=False))


def grouped_bars(
    categories: Sequence[str],
    series: dict[str, Sequence[float]],
    theme: Theme,
    *,
    title: str,
    subtitle: str = "",
    ylabel: str = "",
    errors: dict[str, Sequence[tuple[float, float]]] | None = None,
    value_fmt: str = "{:.0%}",
    ylim: tuple[float, float] = (0.0, 1.0),
    figsize: tuple[float, float] = (10.0, 5.2),
) -> Any:
    """Grouped vertical bars: one group per category, one bar per series.

    A single value axis, always. Two measures on different scales get two charts,
    never a second y-axis -- a dual-axis chart lets the author choose the visual
    conclusion by picking scales, which is why it is banned outright.
    """
    fig, ax = plt.subplots(figsize=figsize, dpi=200)
    _style(ax, theme)

    n_series = len(series)
    group_w = 0.78
    bar_w = group_w / n_series
    gap = bar_w * 0.06  # ~2px visual separation between adjacent fills
    xs = list(range(len(categories)))

    ax.set_xlim(-0.6, len(categories) - 0.4)
    # Headroom above the tallest bar so value labels and error-bar caps never
    # collide with the subtitle. Ticks still stop at the data ceiling, so the
    # padding is invisible.
    span = ylim[1] - ylim[0]
    ax.set_ylim(ylim[0], ylim[1] + span * 0.10)
    ax.set_yticks([ylim[0] + span * f for f in (0, 0.2, 0.4, 0.6, 0.8, 1.0)])

    for i, (label, values) in enumerate(series.items()):
        color = theme.series[i % len(theme.series)]
        offs = [x - group_w / 2 + bar_w * (i + 0.5) for x in xs]
        _rounded_bars(ax, offs, values, bar_w - gap, color, theme)

        if errors and label in errors:
            for xo, (lo, hi) in zip(offs, errors[label], strict=True):
                ax.plot([xo, xo], [lo, hi], color=theme.text_secondary, linewidth=1.2,
                        solid_capstyle="butt", zorder=5)
                for y in (lo, hi):
                    ax.plot([xo - bar_w * 0.15, xo + bar_w * 0.15], [y, y],
                            color=theme.text_secondary, linewidth=1.2, zorder=5)

        for idx, (xo, v) in enumerate(zip(offs, values, strict=True)):
            top = max(v, errors[label][idx][1]) if (errors and label in errors) else v
            ax.text(xo, top + span * 0.015, value_fmt.format(v),
                    ha="center", va="bottom", fontsize=8, color=theme.text_secondary,
                    fontfamily=_FONT)

    ax.set_xticks(xs)
    ax.set_xticklabels(categories, fontsize=9.5, fontfamily=_FONT)
    ax.set_ylabel(ylabel, color=theme.text_secondary, fontsize=9.5, fontfamily=_FONT)
    ax.set_title(title, color=theme.text_primary, fontsize=13, fontweight="600",
                 loc="left", pad=26, fontfamily=_FONT)
    if subtitle:
        ax.text(0, 1.028, subtitle, transform=ax.transAxes, color=theme.text_secondary,
                fontsize=9.5, va="bottom", fontfamily=_FONT)

    _legend(ax, theme, list(series), ncol=min(len(series), 4), y=-0.10)
    fig.tight_layout()
    return fig


def horizontal_bars(
    categories: Sequence[str],
    series: dict[str, Sequence[float]],
    theme: Theme,
    *,
    title: str,
    subtitle: str = "",
    value_fmt: str = "{:.2f}",
    xlim: tuple[float, float] = (0.0, 1.0),
    figsize: tuple[float, float] = (10.0, 9.0),
) -> Any:
    """Horizontal grouped bars -- used when there are many long category labels
    (the 18 leaf fields), where vertical bars would force rotated text."""
    fig, ax = plt.subplots(figsize=figsize, dpi=200)
    _style(ax, theme, ygrid=False, xgrid=True)

    n_series = len(series)
    group_h = 0.78
    bar_h = group_h / n_series
    gap = bar_h * 0.06
    ys = list(range(len(categories)))

    ax.set_ylim(-0.6, len(categories) - 0.4)
    xspan = xlim[1] - xlim[0]
    ax.set_xlim(xlim[0], xlim[1] + xspan * 0.08)  # room for the end labels
    ax.set_xticks([xlim[0] + xspan * f for f in (0, 0.25, 0.5, 0.75, 1.0)])

    for i, values in enumerate(series.values()):
        color = theme.series[i % len(theme.series)]
        offs = [y - group_h / 2 + bar_h * (i + 0.5) for y in ys]
        _rounded_bars(ax, offs, values, bar_h - gap, color, theme, horizontal=True)
        for yo, v in zip(offs, values, strict=True):
            ax.text(v + xspan * 0.012, yo, value_fmt.format(v),
                    ha="left", va="center", fontsize=7.5, color=theme.text_secondary,
                    fontfamily=_FONT)

    ax.set_yticks(ys)
    ax.set_yticklabels(categories, fontsize=9, fontfamily=_FONT)
    ax.invert_yaxis()
    ax.set_title(title, color=theme.text_primary, fontsize=13, fontweight="600",
                 loc="left", pad=24, fontfamily=_FONT)
    if subtitle:
        ax.text(0, 1.015, subtitle, transform=ax.transAxes, color=theme.text_secondary,
                fontsize=9.5, va="bottom", fontfamily=_FONT)
    _legend(ax, theme, list(series), ncol=min(len(series), 4), y=-0.045)
    fig.tight_layout()
    return fig


def save_both_themes(build, out_dir: Path, stem: str) -> list[Path]:
    """Render `build(theme)` once per theme and write `<stem>_light|dark.png`.

    The README pairs them with a <picture> element so GitHub serves whichever
    matches the reader's theme.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for theme in THEMES:
        fig = build(theme)
        path = out_dir / f"{stem}_{theme.name}.png"
        fig.savefig(path, facecolor=theme.surface, bbox_inches="tight", dpi=200)
        plt.close(fig)
        paths.append(path)
    return paths
