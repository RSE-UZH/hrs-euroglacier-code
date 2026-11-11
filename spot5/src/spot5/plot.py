import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio
from matplotlib import colors as mcolors
from rasterio.transform import from_origin, xy
from shapely.geometry import Point
from shapely.strtree import STRtree

from spot5.utils.dotdict import DotDict


def plot_pairs_temporal_distribution(pairs_df: pd.DataFrame, plot_config: dict = None):
    """
    Create a temporal distribution plot of SPOT pairs with ablation/accumulation periods.

    Parameters:
        pairs_df (pd.DataFrame): DataFrame containing pairs data with 'date', and 'count' columns
        plot_config (dict): Plot configuration object containing style settings

    Returns:
        tuple: (fig, ax) matplotlib figure and axis objects
    """

    _default_config_dict = {
        "figure": {
            "figsize": (7, 4),
            "dpi": 150,
            "style": "seaborn-v0_8-bright",
            "font_family": "sans-serif",
            "font_size": 12,
            "title": "Temporal distribution of SPOT pairs",
            "frame": {
                "visible": False,  # Hide the outer frame
                "top": False,  # Hide top spine
                "right": False,  # Hide right spine
                "left": False,  # Show left spine
                "bottom": False,  # Show bottom spine
            },
        },
        "scatter": {
            "alpha": 0.8,
            "size": 40,
            "edge_color": "white",
            "edge_width": 0.4,
            "cmap": "viridis",
            "color_bins": 7,
            "categorize_months": False,
            "use_jitter": False,
            "jitter": 0.2,
        },
        "periods": {
            "start": 10,  # September
            "end": 6,  # June
            "lines": {
                "style": "--",
                "alpha": 0.7,
                "width": 0.8,
                "color": "#ff9999",
            },
            "bands": {
                "accumulation": {"alpha": 0.1, "color": "cornflowerblue"},
                "ablation": {"alpha": 0.1, "color": "tomato"},
            },
        },
    }

    # Merge the default config with the passed config
    config = DotDict(_default_config_dict)
    if plot_config is not None:
        config.merge(plot_config)

    # Apply style settings
    plt.style.use(config.figure.style)
    plt.rcParams["font.family"] = config.figure.font_family

    # Create figure
    fig, ax = plt.subplots(figsize=config.figure.figsize, dpi=config.figure.dpi)

    # Extract the year
    year = pairs_df["date"].dt.year

    # Extract numerical month values (1-12)
    month_num = pairs_df["date"].dt.month

    # Categorize months if needed
    if config.scatter.categorize_months:
        if config.scatter.use_jitter:
            # Add jitter to reduce overlap
            jitter = config.scatter.jitter
            y_values = month_num + np.random.uniform(-jitter, jitter, len(pairs_df))
        else:
            y_values = month_num

        ylims = (0.5, 12.5)
    else:
        y_values = pairs_df["date"].dt.month + pairs_df["date"].dt.day / 30
        ylims = (0.5, 13.5)

    # Create scatter plot
    if "count" in pairs_df.columns:
        count = pairs_df["count"]
        bins = np.linspace(count.min(), count.max(), config.scatter.color_bins)
        cmap = plt.cm.get_cmap(config.scatter.cmap, config.scatter.color_bins)
        norm = mcolors.BoundaryNorm(bins, cmap.N)

        scatter = ax.scatter(
            year,
            y_values,
            c=count,
            cmap=cmap,
            norm=norm,
            alpha=config.scatter.alpha,
            s=config.scatter.size,
            edgecolor=config.scatter.edge_color,
            linewidth=config.scatter.edge_width,
        )
    else:
        scatter = ax.scatter(
            year,
            y_values,
            c="blue",
            cmap=config.scatter.cmap,
            alpha=config.scatter.alpha,
            s=config.scatter.size,
            edgecolor=config.scatter.edge_color,
            linewidth=config.scatter.edge_width,
        )

    # Calculate statistics
    summer_mask = (month_num >= config.periods.end) & (
        month_num <= config.periods.start
    )
    winter_mask = ~summer_mask

    n_summer = len(pairs_df[summer_mask])
    n_winter = len(pairs_df[winter_mask])
    total_scenes = len(pairs_df)

    # Add statistics text
    stats_text = (
        f"Total scenes: {total_scenes}\n"
        f"Summer scenes: {n_summer} ({n_summer / total_scenes:.1%})\n"
        f"Winter scenes: {n_winter} ({n_winter / total_scenes:.1%})"
    )
    ax.text(
        0.98,
        0.98,
        stats_text,
        transform=ax.transAxes,
        verticalalignment="top",
        horizontalalignment="right",
        bbox=dict(facecolor="white", alpha=0.8, edgecolor="none"),
        fontsize=8,
    )

    # Mark ablation period with boundary lines
    ax.axhline(
        y=config.periods.start,
        color=config.periods.lines.color,
        linestyle=config.periods.lines.style,
        alpha=config.periods.lines.alpha,
        linewidth=config.periods.lines.width,
        label="Ablation period",
    )
    ax.axhline(
        y=config.periods.end,
        color=config.periods.lines.color,
        linestyle=config.periods.lines.style,
        alpha=config.periods.lines.alpha,
        linewidth=config.periods.lines.width,
    )

    # Add background bands for the ablation period
    ax.axhspan(
        config.periods.end,
        config.periods.start,
        alpha=config.periods.bands.ablation.alpha,
        color=config.periods.bands.ablation.color,
    )

    # Add background bands for the accumulation period
    ax.axhspan(
        config.periods.start,
        ylims[1],
        alpha=config.periods.bands.accumulation.alpha,
        color=config.periods.bands.accumulation.color,
    )
    ax.axhspan(
        ylims[0],
        config.periods.end,
        alpha=config.periods.bands.accumulation.alpha,
        color=config.periods.bands.accumulation.color,
    )

    # Customize x-axis
    min_year = year.min()
    max_year = year.max()
    years = range(min_year, max_year + 1)
    ax.set_xticks(years)
    ax.set_xticklabels(years, rotation=0)
    ax.set_xlim(min(years) - 0.5, max(years) + 0.5)
    ax.set_xlabel("Year", fontsize=config.figure.font_size)
    # ax.xaxis.set_major_formatter(DateFormatter("%Y"))  # "%b %Y"

    # Set y-axis limits and labels
    ax.set_ylim(ylims)
    month_abbr = [
        "Jan",
        "Feb",
        "Mar",
        "Apr",
        "May",
        "Jun",
        "Jul",
        "Aug",
        "Sep",
        "Oct",
        "Nov",
        "Dec",
    ]
    ax.set_yticks(range(1, 13), labels=month_abbr)
    ax.set_ylabel("Month", fontsize=config.figure.font_size)

    # Add grid
    ax.grid(True, linestyle="--", alpha=0.4)

    # Add colorbar
    cbar = plt.colorbar(scatter, ax=ax, pad=0.01)
    cbar.set_label("Number of pairs per day")

    # Add title
    if config.figure.title:
        ax.set_title(config.figure.title, fontsize=config.figure.font_size)

    # Add legend with better positioning
    ax.legend()

    # Hide frame based on configuration using a single if statement
    if hasattr(config.figure, "frame") and not config.figure.frame.visible:
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_visible(
            config.figure.frame.left
            if hasattr(config.figure.frame, "left")
            else ax.spines["left"].get_visible()
        )
        ax.spines["bottom"].set_visible(
            config.figure.frame.bottom
            if hasattr(config.figure.frame, "bottom")
            else ax.spines["bottom"].get_visible()
        )

    plt.tight_layout()

    return fig, ax


def plot_scenes_polar_distribution(pairs_df: pd.DataFrame, plot_config: dict):
    """
    Create a polar plot showing the distribution of scenes by month, split into summer and winter.
    Improves readability by adding spacing, edges, and a clear legend placement.
    """
    from math import pi

    _default_config_dict = {
        "figure": {"figsize": (6, 6), "dpi": 100, "style": "seaborn-v0_8-bright"},
        "periods": {"start": 9, "end": 6},
        "colors": {"summer": "tomato", "winter": "cornflowerblue"},
    }

    config = DotDict(_default_config_dict)
    if plot_config is not None:
        config.merge(plot_config)

    plt.style.use(config.figure.style)

    # Define month and season masks
    month_num = pairs_df["date"].dt.month
    summer_mask = (month_num >= config.periods.end) & (
        month_num <= config.periods.start
    )
    winter_mask = ~summer_mask

    # Count by month for each season
    summer_counts = pairs_df[summer_mask]["date"].dt.month.value_counts().sort_index()
    winter_counts = pairs_df[winter_mask]["date"].dt.month.value_counts().sort_index()
    summer_counts = summer_counts.reindex(range(1, 13), fill_value=0)
    winter_counts = winter_counts.reindex(range(1, 13), fill_value=0)

    # Angles and bar width
    angles = np.linspace(0, 2 * pi, 12, endpoint=False)
    bar_width = 0.95 * (2 * pi / 12)  # Slightly reduced width for spacing

    # Create polar plot
    fig, ax = plt.subplots(
        figsize=config.figure.figsize,
        dpi=config.figure.dpi,
        subplot_kw={"projection": "polar"},
    )

    # Make the plot clockwise and place January at the top
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)

    # Plot winter part first, then summer on top
    ax.bar(
        angles,
        winter_counts,
        width=bar_width,
        color=config.colors.winter,
        alpha=0.7,
        label="Winter Scenes",
        edgecolor="white",
        linewidth=1,
    )
    ax.bar(
        angles,
        summer_counts,
        width=bar_width,
        bottom=winter_counts,
        color=config.colors.summer,
        alpha=0.7,
        label="Summer Scenes",
        edgecolor="white",
        linewidth=1,
    )

    # Improve the style of radial (magnitude) labels
    max_value = (winter_counts + summer_counts).max()
    r_ticks = np.linspace(0, max_value, 6)  # 6 intervals from 0 to max_value
    ax.set_rgrids(
        r_ticks,
        angle=0,
        labels=[f"{int(t)}" for t in r_ticks],
        fontsize=10,
        color="gray",
    )

    # Set month labels
    month_labels = [
        "Jan",
        "Feb",
        "Mar",
        "Apr",
        "May",
        "Jun",
        "Jul",
        "Aug",
        "Sep",
        "Oct",
        "Nov",
        "Dec",
    ]
    ax.set_xticks(angles)
    ax.set_xticklabels(month_labels, fontsize=12)
    ax.tick_params(axis="x", pad=10)

    # Improve title and spacing
    ax.set_title("Distribution of Scenes by Month", pad=40, fontsize=14)

    # Place legend below the plot to avoid overlap
    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.18),
        fancybox=True,
        shadow=False,
        ncol=2,
    )

    return fig, ax


def plot_scene_distribution(
    footprints, group_by="cluster_id", ax=None, figsize=(12, 6), dpi=100
):
    """
    Create a bar plot showing the distribution of scenes by either cluster_id or grid reference.

    Parameters:
    -----------
    footprints : pandas.DataFrame
        DataFrame containing the scenes data.
    group_by : str, optional
        Field name to group scenes by. If "cluster_id" is chosen but not present in the dataframe,
        the function falls back to grouping by "hrs1_grid_reference". Default is "cluster_id".
    ax : matplotlib.axes.Axes, optional
        Axes object to plot on. If None, a new figure is created.
    figsize : tuple, optional
        Figure size (width, height) in inches.
    dpi : int, optional
        Figure resolution.

    Returns:
    --------
    fig, ax : tuple
        The figure and axes objects used for plotting.
    """
    # Determine the field for grouping
    if group_by == "cluster_id" and "cluster_id" in footprints.columns:
        field = "cluster_id"
        xlabel = "Cluster ID"
    else:
        field = "hrs1_grid_reference"
        xlabel = "Grid Reference"

    # Group data
    indexes = sorted(footprints[field].unique())
    counts = footprints[field].value_counts().sort_index()

    # Create figure if not provided
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    else:
        fig = ax.figure

    # Create bar plot with styling
    bars = ax.bar(
        indexes, counts, color="skyblue", edgecolor="navy", alpha=0.7, width=0.8
    )

    # Add value labels on top of each bar
    for bar in bars:
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            height,
            f"{int(height)}",
            ha="center",
            va="bottom",
        )

    # Customize the plot
    ax.set_title(
        "Distribution of SPOT5 Scenes by " + xlabel,
        pad=20,
        fontsize=14,
        fontweight="bold",
    )
    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel("Number of Scenes", fontsize=12)

    # Improve readability on the x-axis
    plt.setp(ax.get_xticklabels(), rotation=45, ha="center", fontsize=10)

    # Add grid for better readability
    ax.grid(True, axis="y", linestyle="--", alpha=0.7)

    # Remove top and right spines
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Add total count as text
    total_scenes = counts.sum()
    ax.text(
        0.98,
        0.98,
        f"Total scenes: {total_scenes}",
        transform=ax.transAxes,
        ha="right",
        va="top",
        bbox=dict(facecolor="white", edgecolor="none", alpha=0.8),
    )

    plt.tight_layout()
    return fig, ax


def create_count_raster(
    footprints: gpd.GeoDataFrame,
    output_path: str,
    resolution: float,
    crs: str,
    query_predicate: str = "intersects",
):
    """
    Creates a raster with the count of intersecting footprints for each cell.

    Args:
        footprints: GeoDataFrame of footprint polygons.
        output_path: Path to save the output raster.
        resolution: Resolution of the raster in units of the CRS.
        crs: Coordinate Reference System for the raster.
    """

    # Get the bounding box of the footprints
    bbox = footprints.total_bounds

    # Calculate the raster dimensions
    width = int((bbox[2] - bbox[0]) / resolution)
    height = int((bbox[3] - bbox[1]) / resolution)

    # Create an empty raster
    raster = np.zeros((height, width), dtype=np.uint16)

    # Create the transform
    transform = from_origin(bbox[0], bbox[3], resolution, resolution)

    # Create an STRtree for fast spatial queries
    footprints_tree = STRtree(footprints.geometry)

    # Create arrays of pixel indices
    rows, cols = np.indices((height, width))

    # Get the coordinates for all pixels at once
    xs, ys = xy(transform, rows, cols)

    # Process the pixels
    for i, (x, y) in enumerate(zip(xs.flat, ys.flat, strict=True)):
        # Get indices of potentially intersecting geometries
        intersecting_geoms = list(
            footprints_tree.query(Point(x, y), predicate=query_predicate)
        )

        # Count the number of intersecting geometries
        count = len(intersecting_geoms)

        # Write the count to the raster
        raster.flat[i] = count

    # Write the raster to a file
    with rasterio.open(
        output_path,
        "w",
        driver="GTiff",
        height=height,
        width=width,
        count=1,
        dtype=raster.dtype,
        crs=crs,
        transform=transform,
    ) as dst:
        dst.write(raster, 1)
