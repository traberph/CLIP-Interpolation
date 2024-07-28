def fig_post(fig, marker_size=2, hide_axes=True, size=(1500,700)):
    """
    Applies post-processing to a plotly figure object.

    Args:
        fig (plotly.graph_objects.Figure): The figure object to be processed.
        marker_size (int, optional): The size of the markers in the figure. Defaults to 2.
        hide_axes (bool, optional): Whether to hide the axes in the figure. Defaults to True.
        size (tuple, optional): The size of the figure in pixels. Defaults to (1500, 700).
    """
    fig.update_layout({
        'plot_bgcolor': 'rgba(0, 0, 0, 0)',
        'paper_bgcolor': 'rgba(0, 0, 0, 0)',
        'font': dict(color='lightgrey'), 
    })
    if hide_axes:
        fig.update_scenes(xaxis_visible=False, yaxis_visible=False,zaxis_visible=False)
    fig.update_traces(marker=dict(size=marker_size), textposition='top center')
    fig.update_layout(autosize=False, width=size[0], height=size[1])
    fig.show()