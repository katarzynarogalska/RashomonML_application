import plotly.express as px
import pandas as pd


def plot_class_distribution(dataset, target_column):
 
    counts = dataset[target_column].value_counts().reset_index()
    counts.columns = ['Class', 'Count']

    unique_classes = dataset[target_column].unique()

    fig = px.bar(
        counts,
        x='Class',
        y='Count',
        color_discrete_sequence=['#426c85'],
        title='Number of observations per class'
    )

    fig.update_traces(
        hovertemplate='Class: %{x}<br>Number of observations: %{y}<extra></extra>'
    )

    fig.update_layout(
        xaxis_title='Class',
        yaxis_title='Number of observations',
        plot_bgcolor='white',
        paper_bgcolor='white',
        font=dict(color='black'),
        title_x=0.5
    )

    fig.update_xaxes(
        tickmode='array',
        tickvals=unique_classes,
        showline=True,
        linecolor='black',
        gridcolor='lightgray'
    )
    
    fig.update_yaxes(
        showline=True,
        linecolor='black',
        gridcolor='lightgray'
    )

    return fig