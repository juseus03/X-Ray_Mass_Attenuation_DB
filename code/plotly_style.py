import plotly.graph_objects as go
import plotly.io as pio


def setup_custom_template():
    """Setup custom Plotly template matching matplotlib style"""

    custom_template = go.layout.Template()

    # Layout settings
    custom_template.layout = go.Layout(
        # Font settings
        font=dict(family="sans-serif", size=14, color="black"),
        # Title and labels font sizes
        title_font=dict(size=17),
        xaxis=dict(
            title_font=dict(size=17),
            tickfont=dict(size=17),
            linewidth=0.5,
            linecolor="black",
            showline=True,
            ticks="outside",
            mirror=False,
            showgrid=False,
        ),
        yaxis=dict(
            title_font=dict(size=17),
            tickfont=dict(size=17),
            linewidth=0.5,
            linecolor="black",
            showline=True,
            ticks="outside",
            mirror=False,
            showgrid=False,
        ),
        # Transparent background
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        # Legend settings
        legend=dict(font=dict(size=14), bgcolor="rgba(0,0,0,0)", borderwidth=0),
        # Figure size
        width=600,
        height=450,
        # Margins
        margin=dict(l=60, r=20, t=40, b=50, autoexpand=True),
        # Color sequence
        colorway=[
            "#000000",
            "#1f77b4",
            "#ff7f0e",
            "#2ca02c",
            "#d62728",
            "#9467bd",
            "#8c564b",
            "#e377c2",
            "#7f7f7f",
            "#bcbd22",
            "#17becf",
        ],
    )

    # Line settings
    custom_template.data.scatter = [go.Scatter(line=dict(width=1))]

    # Register and set as default
    pio.templates["custom_style"] = custom_template
    pio.templates.default = "custom_style"


def apply_style(fig):
    """Apply additional styling to a figure"""
    fig.update_xaxes(showline=True, linewidth=0.5, linecolor="black", mirror=False)
    fig.update_yaxes(showline=True, linewidth=0.5, linecolor="black", mirror=False)
    return fig
