import pandas as pd
import plotly.graph_objects as go
import plotly.express as px


def plot_class_distribution(data, target_column):
    counts = data[target_column].astype(str).value_counts()
    fig = go.Figure(data=[go.Bar(
        x=counts.index,
        y=counts.values,
        marker_color = '#92c5de'
    )])
    fig.update_layout(
        title=dict(
            text="Class Distribution plot",
            font=dict(
                family="Inter, sans-serif",  
                size=14,
                color="white"
            ),
            x=0.1),
        plot_bgcolor='rgba(0,0,0,0)',   
        paper_bgcolor='rgba(0,0,0,0)',
        yaxis=dict(
           title=dict(
                text="Count",
                font=dict(color='white')
            ),
            tickfont=dict(color='white'),
            gridcolor='lightgray',
        ),
        xaxis=dict(
            title=dict(
                text="Class",
                font=dict(color='white')
            ),
            tickfont=dict(color='white'),
            gridcolor='lightgray',
            type='category'          
        ),
        margin=dict(l=60, r=10, t=80, b=60)
    )
    return fig 

# #for numeric features
def plot_correlation_matrix(data):
    numeric_data = data.select_dtypes(include='number')
    corr = numeric_data.corr()
    fig = go.Figure(data=go.Heatmap(
        z=corr.values,
        x=corr.columns,
        y=corr.columns,
        colorscale='RdBu_r',
        zmin=-1,
        zmax=1,
        colorbar=dict(
            title='corr',
            thickness=15,   
            tickfont=dict(color='white'),
        )
    ))
    fig.update_layout(
        title=dict(
            text="Pearsons Correlation Matrix",
            font=dict(
                family="Inter, sans-serif",
                size=14,
                color="white"
            ),
            x=0.1
        ),
        xaxis=dict(
            tickfont=dict(color='white'),
            title=dict(font=dict(color='white')),
            gridcolor='lightgray',
            side='bottom'
        ),
        yaxis=dict(
            tickfont=dict(color='white'),
            title=dict(font=dict(color='white')),
            gridcolor='lightgray'
        ),
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        margin=dict(l=10, r=0, t=80, b=60)
    )
    return fig

def return_missing(dataset):
    total_cells = dataset.size
    nan_cells = dataset.isna().sum().sum()
    return (nan_cells / total_cells) * 100








