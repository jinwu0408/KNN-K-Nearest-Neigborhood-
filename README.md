<div>
<img src="attachment:thumbnail.png" width="800">
</div>

# Import All necessary library


```python
import pandas as pd
import numpy as np
from math import sqrt
import plotly.express as px
```

# Data Cleaning and Processing



```python
df = pd.read_csv('iris.csv')
df = df.drop('Id', axis=1)
print(df.shape)
df = df.sample(frac=1).reset_index(drop=True)
df.head()
```

    (150, 5)





<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>SepalLengthCm</th>
      <th>SepalWidthCm</th>
      <th>PetalLengthCm</th>
      <th>PetalWidthCm</th>
      <th>Species</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>6.7</td>
      <td>2.5</td>
      <td>5.8</td>
      <td>1.8</td>
      <td>Iris-virginica</td>
    </tr>
    <tr>
      <td>1</td>
      <td>5.7</td>
      <td>2.8</td>
      <td>4.5</td>
      <td>1.3</td>
      <td>Iris-versicolor</td>
    </tr>
    <tr>
      <td>2</td>
      <td>6.0</td>
      <td>2.7</td>
      <td>5.1</td>
      <td>1.6</td>
      <td>Iris-versicolor</td>
    </tr>
    <tr>
      <td>3</td>
      <td>5.8</td>
      <td>2.7</td>
      <td>5.1</td>
      <td>1.9</td>
      <td>Iris-virginica</td>
    </tr>
    <tr>
      <td>4</td>
      <td>6.9</td>
      <td>3.2</td>
      <td>5.7</td>
      <td>2.3</td>
      <td>Iris-virginica</td>
    </tr>
  </tbody>
</table>
</div>




```python
percent_of_training = 0.8
cutoff_ind = int(0.8*df.shape[0])

training_data = df.loc[0:cutoff_ind].values
testing_data = df.loc[cutoff_ind:].values

print('Training data: ',training_data.shape,type(training_data))
print('Testing data: ',testing_data.shape,type(training_data))
```

    Training data:  (121, 5) <class 'numpy.ndarray'>
    Testing data:  (30, 5) <class 'numpy.ndarray'>


# Data Visualization


```python
# fig = px.scatter(x=df['SepalLengthCm'], y=df['SepalWidthCm'],color = df['Species'])
# fig.show()
```


<div>


            <div id="51857f6a-12a8-4372-904c-6bc073dedb1d" class="plotly-graph-div" style="height:525px; width:100%;"></div>
            <script type="text/javascript">
                require(["plotly"], function(Plotly) {
                    window.PLOTLYENV=window.PLOTLYENV || {};

                if (document.getElementById("51857f6a-12a8-4372-904c-6bc073dedb1d")) {
                    Plotly.newPlot(
                        '51857f6a-12a8-4372-904c-6bc073dedb1d',
                        [{"hoverlabel": {"namelength": 0}, "hovertemplate": "color=Iris-virginica<br>x=%{x}<br>y=%{y}", "legendgroup": "color=Iris-virginica", "marker": {"color": "#636efa", "symbol": "circle"}, "mode": "markers", "name": "color=Iris-virginica", "showlegend": true, "type": "scatter", "x": [6.7, 5.8, 6.9, 7.7, 6.4, 5.8, 7.7, 6.2, 6.3, 6.7, 6.7, 6.2, 6.5, 6.9, 6.1, 7.1, 6.8, 7.7, 7.9, 7.2, 6.3, 6.3, 5.9, 6.0, 4.9, 6.4, 6.3, 6.0, 5.7, 7.2, 6.9, 6.4, 7.2, 6.4, 6.5, 6.3, 6.5, 5.6, 6.3, 6.1, 6.5, 6.8, 6.7, 6.7, 7.6, 5.8, 7.4, 7.3, 7.7, 6.4], "xaxis": "x", "y": [2.5, 2.7, 3.2, 3.8, 3.1, 2.8, 2.6, 2.8, 2.8, 3.1, 3.3, 3.4, 3.0, 3.1, 3.0, 3.0, 3.0, 3.0, 3.8, 3.2, 3.4, 2.9, 3.0, 2.2, 2.5, 2.8, 2.7, 3.0, 2.5, 3.0, 3.1, 3.2, 3.6, 2.8, 3.0, 3.3, 3.0, 2.8, 2.5, 2.6, 3.2, 3.2, 3.0, 3.3, 3.0, 2.7, 2.8, 2.9, 2.8, 2.7], "yaxis": "y"}, {"hoverlabel": {"namelength": 0}, "hovertemplate": "color=Iris-versicolor<br>x=%{x}<br>y=%{y}", "legendgroup": "color=Iris-versicolor", "marker": {"color": "#EF553B", "symbol": "circle"}, "mode": "markers", "name": "color=Iris-versicolor", "showlegend": true, "type": "scatter", "x": [5.7, 6.0, 7.0, 5.7, 5.6, 5.7, 6.1, 6.1, 6.2, 6.3, 5.5, 6.0, 5.5, 6.3, 6.0, 6.6, 5.8, 5.5, 5.6, 5.8, 6.1, 6.0, 5.0, 6.4, 5.4, 6.5, 5.1, 5.6, 6.7, 6.3, 5.6, 6.7, 6.1, 5.5, 6.9, 5.5, 5.7, 4.9, 6.7, 6.4, 5.2, 5.0, 6.8, 5.7, 5.9, 5.9, 6.2, 5.6, 5.8, 6.6], "xaxis": "x", "y": [2.8, 2.7, 3.2, 2.6, 2.9, 3.0, 2.8, 2.9, 2.2, 2.5, 2.6, 3.4, 2.5, 3.3, 2.2, 2.9, 2.7, 2.4, 3.0, 2.7, 3.0, 2.9, 2.0, 3.2, 3.0, 2.8, 2.5, 2.5, 3.1, 2.3, 3.0, 3.0, 2.8, 2.4, 3.1, 2.3, 2.9, 2.4, 3.1, 2.9, 2.7, 2.3, 2.8, 2.8, 3.0, 3.2, 2.9, 2.7, 2.6, 3.0], "yaxis": "y"}, {"hoverlabel": {"namelength": 0}, "hovertemplate": "color=Iris-setosa<br>x=%{x}<br>y=%{y}", "legendgroup": "color=Iris-setosa", "marker": {"color": "#00cc96", "symbol": "circle"}, "mode": "markers", "name": "color=Iris-setosa", "showlegend": true, "type": "scatter", "x": [4.3, 5.0, 5.4, 5.2, 4.4, 5.1, 5.1, 5.2, 5.0, 5.4, 5.1, 5.1, 5.0, 4.4, 5.2, 5.0, 5.4, 4.6, 5.8, 5.4, 5.0, 4.8, 4.6, 5.0, 4.9, 5.5, 5.7, 4.8, 5.1, 5.3, 5.5, 4.7, 4.9, 4.8, 5.1, 4.8, 4.4, 4.8, 5.0, 4.6, 5.7, 4.9, 4.5, 4.6, 5.1, 5.4, 5.1, 5.0, 4.9, 4.7], "xaxis": "x", "y": [3.0, 3.3, 3.9, 3.5, 2.9, 3.3, 3.8, 3.4, 3.4, 3.4, 3.4, 3.5, 3.4, 3.0, 4.1, 3.5, 3.4, 3.2, 4.0, 3.9, 3.0, 3.4, 3.1, 3.2, 3.0, 3.5, 4.4, 3.0, 3.8, 3.7, 4.2, 3.2, 3.1, 3.1, 3.5, 3.4, 3.2, 3.0, 3.6, 3.6, 3.8, 3.1, 2.3, 3.4, 3.8, 3.7, 3.7, 3.5, 3.1, 3.2], "yaxis": "y"}],
                        {"legend": {"tracegroupgap": 0}, "margin": {"t": 60}, "template": {"data": {"bar": [{"error_x": {"color": "#2a3f5f"}, "error_y": {"color": "#2a3f5f"}, "marker": {"line": {"color": "#E5ECF6", "width": 0.5}}, "type": "bar"}], "barpolar": [{"marker": {"line": {"color": "#E5ECF6", "width": 0.5}}, "type": "barpolar"}], "carpet": [{"aaxis": {"endlinecolor": "#2a3f5f", "gridcolor": "white", "linecolor": "white", "minorgridcolor": "white", "startlinecolor": "#2a3f5f"}, "baxis": {"endlinecolor": "#2a3f5f", "gridcolor": "white", "linecolor": "white", "minorgridcolor": "white", "startlinecolor": "#2a3f5f"}, "type": "carpet"}], "choropleth": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "type": "choropleth"}], "contour": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "colorscale": [[0.0, "#0d0887"], [0.1111111111111111, "#46039f"], [0.2222222222222222, "#7201a8"], [0.3333333333333333, "#9c179e"], [0.4444444444444444, "#bd3786"], [0.5555555555555556, "#d8576b"], [0.6666666666666666, "#ed7953"], [0.7777777777777778, "#fb9f3a"], [0.8888888888888888, "#fdca26"], [1.0, "#f0f921"]], "type": "contour"}], "contourcarpet": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "type": "contourcarpet"}], "heatmap": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "colorscale": [[0.0, "#0d0887"], [0.1111111111111111, "#46039f"], [0.2222222222222222, "#7201a8"], [0.3333333333333333, "#9c179e"], [0.4444444444444444, "#bd3786"], [0.5555555555555556, "#d8576b"], [0.6666666666666666, "#ed7953"], [0.7777777777777778, "#fb9f3a"], [0.8888888888888888, "#fdca26"], [1.0, "#f0f921"]], "type": "heatmap"}], "heatmapgl": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "colorscale": [[0.0, "#0d0887"], [0.1111111111111111, "#46039f"], [0.2222222222222222, "#7201a8"], [0.3333333333333333, "#9c179e"], [0.4444444444444444, "#bd3786"], [0.5555555555555556, "#d8576b"], [0.6666666666666666, "#ed7953"], [0.7777777777777778, "#fb9f3a"], [0.8888888888888888, "#fdca26"], [1.0, "#f0f921"]], "type": "heatmapgl"}], "histogram": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "histogram"}], "histogram2d": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "colorscale": [[0.0, "#0d0887"], [0.1111111111111111, "#46039f"], [0.2222222222222222, "#7201a8"], [0.3333333333333333, "#9c179e"], [0.4444444444444444, "#bd3786"], [0.5555555555555556, "#d8576b"], [0.6666666666666666, "#ed7953"], [0.7777777777777778, "#fb9f3a"], [0.8888888888888888, "#fdca26"], [1.0, "#f0f921"]], "type": "histogram2d"}], "histogram2dcontour": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "colorscale": [[0.0, "#0d0887"], [0.1111111111111111, "#46039f"], [0.2222222222222222, "#7201a8"], [0.3333333333333333, "#9c179e"], [0.4444444444444444, "#bd3786"], [0.5555555555555556, "#d8576b"], [0.6666666666666666, "#ed7953"], [0.7777777777777778, "#fb9f3a"], [0.8888888888888888, "#fdca26"], [1.0, "#f0f921"]], "type": "histogram2dcontour"}], "mesh3d": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "type": "mesh3d"}], "parcoords": [{"line": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "parcoords"}], "pie": [{"automargin": true, "type": "pie"}], "scatter": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scatter"}], "scatter3d": [{"line": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scatter3d"}], "scattercarpet": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scattercarpet"}], "scattergeo": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scattergeo"}], "scattergl": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scattergl"}], "scattermapbox": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scattermapbox"}], "scatterpolar": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scatterpolar"}], "scatterpolargl": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scatterpolargl"}], "scatterternary": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scatterternary"}], "surface": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "colorscale": [[0.0, "#0d0887"], [0.1111111111111111, "#46039f"], [0.2222222222222222, "#7201a8"], [0.3333333333333333, "#9c179e"], [0.4444444444444444, "#bd3786"], [0.5555555555555556, "#d8576b"], [0.6666666666666666, "#ed7953"], [0.7777777777777778, "#fb9f3a"], [0.8888888888888888, "#fdca26"], [1.0, "#f0f921"]], "type": "surface"}], "table": [{"cells": {"fill": {"color": "#EBF0F8"}, "line": {"color": "white"}}, "header": {"fill": {"color": "#C8D4E3"}, "line": {"color": "white"}}, "type": "table"}]}, "layout": {"annotationdefaults": {"arrowcolor": "#2a3f5f", "arrowhead": 0, "arrowwidth": 1}, "coloraxis": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "colorscale": {"diverging": [[0, "#8e0152"], [0.1, "#c51b7d"], [0.2, "#de77ae"], [0.3, "#f1b6da"], [0.4, "#fde0ef"], [0.5, "#f7f7f7"], [0.6, "#e6f5d0"], [0.7, "#b8e186"], [0.8, "#7fbc41"], [0.9, "#4d9221"], [1, "#276419"]], "sequential": [[0.0, "#0d0887"], [0.1111111111111111, "#46039f"], [0.2222222222222222, "#7201a8"], [0.3333333333333333, "#9c179e"], [0.4444444444444444, "#bd3786"], [0.5555555555555556, "#d8576b"], [0.6666666666666666, "#ed7953"], [0.7777777777777778, "#fb9f3a"], [0.8888888888888888, "#fdca26"], [1.0, "#f0f921"]], "sequentialminus": [[0.0, "#0d0887"], [0.1111111111111111, "#46039f"], [0.2222222222222222, "#7201a8"], [0.3333333333333333, "#9c179e"], [0.4444444444444444, "#bd3786"], [0.5555555555555556, "#d8576b"], [0.6666666666666666, "#ed7953"], [0.7777777777777778, "#fb9f3a"], [0.8888888888888888, "#fdca26"], [1.0, "#f0f921"]]}, "colorway": ["#636efa", "#EF553B", "#00cc96", "#ab63fa", "#FFA15A", "#19d3f3", "#FF6692", "#B6E880", "#FF97FF", "#FECB52"], "font": {"color": "#2a3f5f"}, "geo": {"bgcolor": "white", "lakecolor": "white", "landcolor": "#E5ECF6", "showlakes": true, "showland": true, "subunitcolor": "white"}, "hoverlabel": {"align": "left"}, "hovermode": "closest", "mapbox": {"style": "light"}, "paper_bgcolor": "white", "plot_bgcolor": "#E5ECF6", "polar": {"angularaxis": {"gridcolor": "white", "linecolor": "white", "ticks": ""}, "bgcolor": "#E5ECF6", "radialaxis": {"gridcolor": "white", "linecolor": "white", "ticks": ""}}, "scene": {"xaxis": {"backgroundcolor": "#E5ECF6", "gridcolor": "white", "gridwidth": 2, "linecolor": "white", "showbackground": true, "ticks": "", "zerolinecolor": "white"}, "yaxis": {"backgroundcolor": "#E5ECF6", "gridcolor": "white", "gridwidth": 2, "linecolor": "white", "showbackground": true, "ticks": "", "zerolinecolor": "white"}, "zaxis": {"backgroundcolor": "#E5ECF6", "gridcolor": "white", "gridwidth": 2, "linecolor": "white", "showbackground": true, "ticks": "", "zerolinecolor": "white"}}, "shapedefaults": {"line": {"color": "#2a3f5f"}}, "ternary": {"aaxis": {"gridcolor": "white", "linecolor": "white", "ticks": ""}, "baxis": {"gridcolor": "white", "linecolor": "white", "ticks": ""}, "bgcolor": "#E5ECF6", "caxis": {"gridcolor": "white", "linecolor": "white", "ticks": ""}}, "title": {"x": 0.05}, "xaxis": {"automargin": true, "gridcolor": "white", "linecolor": "white", "ticks": "", "title": {"standoff": 15}, "zerolinecolor": "white", "zerolinewidth": 2}, "yaxis": {"automargin": true, "gridcolor": "white", "linecolor": "white", "ticks": "", "title": {"standoff": 15}, "zerolinecolor": "white", "zerolinewidth": 2}}}, "xaxis": {"anchor": "y", "domain": [0.0, 1.0], "title": {"text": "x"}}, "yaxis": {"anchor": "x", "domain": [0.0, 1.0], "title": {"text": "y"}}},
                        {"responsive": true}
                    ).then(function(){

var gd = document.getElementById('51857f6a-12a8-4372-904c-6bc073dedb1d');
var x = new MutationObserver(function (mutations, observer) {{
        var display = window.getComputedStyle(gd).display;
        if (!display || display === 'none') {{
            console.log([gd, 'removed!']);
            Plotly.purge(gd);
            observer.disconnect();
        }}
}});

// Listen for the removal of the full notebook cells
var notebookContainer = gd.closest('#notebook-container');
if (notebookContainer) {{
    x.observe(notebookContainer, {childList: true});
}}

// Listen for the clearing of the current output cell
var outputEl = gd.closest('.output');
if (outputEl) {{
    x.observe(outputEl, {childList: true});
}}

                        })
                };
                });
            </script>
        </div>



```python
# fig = px.scatter(x=df['PetalLengthCm'], y=df['PetalWidthCm'],color = df['Species'])
# fig.show()
```


<div>


            <div id="cb5f820b-68d6-4c20-bc46-d80a3ad36194" class="plotly-graph-div" style="height:525px; width:100%;"></div>
            <script type="text/javascript">
                require(["plotly"], function(Plotly) {
                    window.PLOTLYENV=window.PLOTLYENV || {};

                if (document.getElementById("cb5f820b-68d6-4c20-bc46-d80a3ad36194")) {
                    Plotly.newPlot(
                        'cb5f820b-68d6-4c20-bc46-d80a3ad36194',
                        [{"hoverlabel": {"namelength": 0}, "hovertemplate": "color=Iris-virginica<br>x=%{x}<br>y=%{y}", "legendgroup": "color=Iris-virginica", "marker": {"color": "#636efa", "symbol": "circle"}, "mode": "markers", "name": "color=Iris-virginica", "showlegend": true, "type": "scatter", "x": [5.8, 5.1, 5.7, 6.7, 5.5, 5.1, 6.9, 4.8, 5.1, 5.6, 5.7, 5.4, 5.5, 5.4, 4.9, 5.9, 5.5, 6.1, 6.4, 6.0, 5.6, 5.6, 5.1, 5.0, 4.5, 5.6, 4.9, 4.8, 5.0, 5.8, 5.1, 5.3, 6.1, 5.6, 5.2, 6.0, 5.8, 4.9, 5.0, 5.6, 5.1, 5.9, 5.2, 5.7, 6.6, 5.1, 6.1, 6.3, 6.7, 5.3], "xaxis": "x", "y": [1.8, 1.9, 2.3, 2.2, 1.8, 2.4, 2.3, 1.8, 1.5, 2.4, 2.1, 2.3, 1.8, 2.1, 1.8, 2.1, 2.1, 2.3, 2.0, 1.8, 2.4, 1.8, 1.8, 1.5, 1.7, 2.1, 1.8, 1.8, 2.0, 1.6, 2.3, 2.3, 2.5, 2.2, 2.0, 2.5, 2.2, 2.0, 1.9, 1.4, 2.0, 2.3, 2.3, 2.5, 2.1, 1.9, 1.9, 1.8, 2.0, 1.9], "yaxis": "y"}, {"hoverlabel": {"namelength": 0}, "hovertemplate": "color=Iris-versicolor<br>x=%{x}<br>y=%{y}", "legendgroup": "color=Iris-versicolor", "marker": {"color": "#EF553B", "symbol": "circle"}, "mode": "markers", "name": "color=Iris-versicolor", "showlegend": true, "type": "scatter", "x": [4.5, 5.1, 4.7, 3.5, 3.6, 4.2, 4.7, 4.7, 4.5, 4.9, 4.4, 4.5, 4.0, 4.7, 4.0, 4.6, 3.9, 3.8, 4.1, 4.1, 4.6, 4.5, 3.5, 4.5, 4.5, 4.6, 3.0, 3.9, 4.4, 4.4, 4.5, 5.0, 4.0, 3.7, 4.9, 4.0, 4.2, 3.3, 4.7, 4.3, 3.9, 3.3, 4.8, 4.1, 4.2, 4.8, 4.3, 4.2, 4.0, 4.4], "xaxis": "x", "y": [1.3, 1.6, 1.4, 1.0, 1.3, 1.2, 1.2, 1.4, 1.5, 1.5, 1.2, 1.6, 1.3, 1.6, 1.0, 1.3, 1.2, 1.1, 1.3, 1.0, 1.4, 1.5, 1.0, 1.5, 1.5, 1.5, 1.1, 1.1, 1.4, 1.3, 1.5, 1.7, 1.3, 1.0, 1.5, 1.3, 1.3, 1.0, 1.5, 1.3, 1.4, 1.0, 1.4, 1.3, 1.5, 1.8, 1.3, 1.3, 1.2, 1.4], "yaxis": "y"}, {"hoverlabel": {"namelength": 0}, "hovertemplate": "color=Iris-setosa<br>x=%{x}<br>y=%{y}", "legendgroup": "color=Iris-setosa", "marker": {"color": "#00cc96", "symbol": "circle"}, "mode": "markers", "name": "color=Iris-setosa", "showlegend": true, "type": "scatter", "x": [1.1, 1.4, 1.7, 1.5, 1.4, 1.7, 1.9, 1.4, 1.6, 1.5, 1.5, 1.4, 1.5, 1.3, 1.5, 1.6, 1.7, 1.4, 1.2, 1.3, 1.6, 1.6, 1.5, 1.2, 1.4, 1.3, 1.5, 1.4, 1.6, 1.5, 1.4, 1.6, 1.5, 1.6, 1.4, 1.9, 1.3, 1.4, 1.4, 1.0, 1.7, 1.5, 1.3, 1.4, 1.5, 1.5, 1.5, 1.3, 1.5, 1.3], "xaxis": "x", "y": [0.1, 0.2, 0.4, 0.2, 0.2, 0.5, 0.4, 0.2, 0.4, 0.4, 0.2, 0.2, 0.2, 0.2, 0.1, 0.6, 0.2, 0.2, 0.2, 0.4, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.4, 0.3, 0.2, 0.2, 0.2, 0.2, 0.1, 0.2, 0.3, 0.2, 0.2, 0.1, 0.2, 0.2, 0.3, 0.1, 0.3, 0.3, 0.3, 0.2, 0.4, 0.3, 0.1, 0.2], "yaxis": "y"}],
                        {"legend": {"tracegroupgap": 0}, "margin": {"t": 60}, "template": {"data": {"bar": [{"error_x": {"color": "#2a3f5f"}, "error_y": {"color": "#2a3f5f"}, "marker": {"line": {"color": "#E5ECF6", "width": 0.5}}, "type": "bar"}], "barpolar": [{"marker": {"line": {"color": "#E5ECF6", "width": 0.5}}, "type": "barpolar"}], "carpet": [{"aaxis": {"endlinecolor": "#2a3f5f", "gridcolor": "white", "linecolor": "white", "minorgridcolor": "white", "startlinecolor": "#2a3f5f"}, "baxis": {"endlinecolor": "#2a3f5f", "gridcolor": "white", "linecolor": "white", "minorgridcolor": "white", "startlinecolor": "#2a3f5f"}, "type": "carpet"}], "choropleth": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "type": "choropleth"}], "contour": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "colorscale": [[0.0, "#0d0887"], [0.1111111111111111, "#46039f"], [0.2222222222222222, "#7201a8"], [0.3333333333333333, "#9c179e"], [0.4444444444444444, "#bd3786"], [0.5555555555555556, "#d8576b"], [0.6666666666666666, "#ed7953"], [0.7777777777777778, "#fb9f3a"], [0.8888888888888888, "#fdca26"], [1.0, "#f0f921"]], "type": "contour"}], "contourcarpet": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "type": "contourcarpet"}], "heatmap": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "colorscale": [[0.0, "#0d0887"], [0.1111111111111111, "#46039f"], [0.2222222222222222, "#7201a8"], [0.3333333333333333, "#9c179e"], [0.4444444444444444, "#bd3786"], [0.5555555555555556, "#d8576b"], [0.6666666666666666, "#ed7953"], [0.7777777777777778, "#fb9f3a"], [0.8888888888888888, "#fdca26"], [1.0, "#f0f921"]], "type": "heatmap"}], "heatmapgl": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "colorscale": [[0.0, "#0d0887"], [0.1111111111111111, "#46039f"], [0.2222222222222222, "#7201a8"], [0.3333333333333333, "#9c179e"], [0.4444444444444444, "#bd3786"], [0.5555555555555556, "#d8576b"], [0.6666666666666666, "#ed7953"], [0.7777777777777778, "#fb9f3a"], [0.8888888888888888, "#fdca26"], [1.0, "#f0f921"]], "type": "heatmapgl"}], "histogram": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "histogram"}], "histogram2d": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "colorscale": [[0.0, "#0d0887"], [0.1111111111111111, "#46039f"], [0.2222222222222222, "#7201a8"], [0.3333333333333333, "#9c179e"], [0.4444444444444444, "#bd3786"], [0.5555555555555556, "#d8576b"], [0.6666666666666666, "#ed7953"], [0.7777777777777778, "#fb9f3a"], [0.8888888888888888, "#fdca26"], [1.0, "#f0f921"]], "type": "histogram2d"}], "histogram2dcontour": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "colorscale": [[0.0, "#0d0887"], [0.1111111111111111, "#46039f"], [0.2222222222222222, "#7201a8"], [0.3333333333333333, "#9c179e"], [0.4444444444444444, "#bd3786"], [0.5555555555555556, "#d8576b"], [0.6666666666666666, "#ed7953"], [0.7777777777777778, "#fb9f3a"], [0.8888888888888888, "#fdca26"], [1.0, "#f0f921"]], "type": "histogram2dcontour"}], "mesh3d": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "type": "mesh3d"}], "parcoords": [{"line": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "parcoords"}], "pie": [{"automargin": true, "type": "pie"}], "scatter": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scatter"}], "scatter3d": [{"line": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scatter3d"}], "scattercarpet": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scattercarpet"}], "scattergeo": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scattergeo"}], "scattergl": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scattergl"}], "scattermapbox": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scattermapbox"}], "scatterpolar": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scatterpolar"}], "scatterpolargl": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scatterpolargl"}], "scatterternary": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scatterternary"}], "surface": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "colorscale": [[0.0, "#0d0887"], [0.1111111111111111, "#46039f"], [0.2222222222222222, "#7201a8"], [0.3333333333333333, "#9c179e"], [0.4444444444444444, "#bd3786"], [0.5555555555555556, "#d8576b"], [0.6666666666666666, "#ed7953"], [0.7777777777777778, "#fb9f3a"], [0.8888888888888888, "#fdca26"], [1.0, "#f0f921"]], "type": "surface"}], "table": [{"cells": {"fill": {"color": "#EBF0F8"}, "line": {"color": "white"}}, "header": {"fill": {"color": "#C8D4E3"}, "line": {"color": "white"}}, "type": "table"}]}, "layout": {"annotationdefaults": {"arrowcolor": "#2a3f5f", "arrowhead": 0, "arrowwidth": 1}, "coloraxis": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "colorscale": {"diverging": [[0, "#8e0152"], [0.1, "#c51b7d"], [0.2, "#de77ae"], [0.3, "#f1b6da"], [0.4, "#fde0ef"], [0.5, "#f7f7f7"], [0.6, "#e6f5d0"], [0.7, "#b8e186"], [0.8, "#7fbc41"], [0.9, "#4d9221"], [1, "#276419"]], "sequential": [[0.0, "#0d0887"], [0.1111111111111111, "#46039f"], [0.2222222222222222, "#7201a8"], [0.3333333333333333, "#9c179e"], [0.4444444444444444, "#bd3786"], [0.5555555555555556, "#d8576b"], [0.6666666666666666, "#ed7953"], [0.7777777777777778, "#fb9f3a"], [0.8888888888888888, "#fdca26"], [1.0, "#f0f921"]], "sequentialminus": [[0.0, "#0d0887"], [0.1111111111111111, "#46039f"], [0.2222222222222222, "#7201a8"], [0.3333333333333333, "#9c179e"], [0.4444444444444444, "#bd3786"], [0.5555555555555556, "#d8576b"], [0.6666666666666666, "#ed7953"], [0.7777777777777778, "#fb9f3a"], [0.8888888888888888, "#fdca26"], [1.0, "#f0f921"]]}, "colorway": ["#636efa", "#EF553B", "#00cc96", "#ab63fa", "#FFA15A", "#19d3f3", "#FF6692", "#B6E880", "#FF97FF", "#FECB52"], "font": {"color": "#2a3f5f"}, "geo": {"bgcolor": "white", "lakecolor": "white", "landcolor": "#E5ECF6", "showlakes": true, "showland": true, "subunitcolor": "white"}, "hoverlabel": {"align": "left"}, "hovermode": "closest", "mapbox": {"style": "light"}, "paper_bgcolor": "white", "plot_bgcolor": "#E5ECF6", "polar": {"angularaxis": {"gridcolor": "white", "linecolor": "white", "ticks": ""}, "bgcolor": "#E5ECF6", "radialaxis": {"gridcolor": "white", "linecolor": "white", "ticks": ""}}, "scene": {"xaxis": {"backgroundcolor": "#E5ECF6", "gridcolor": "white", "gridwidth": 2, "linecolor": "white", "showbackground": true, "ticks": "", "zerolinecolor": "white"}, "yaxis": {"backgroundcolor": "#E5ECF6", "gridcolor": "white", "gridwidth": 2, "linecolor": "white", "showbackground": true, "ticks": "", "zerolinecolor": "white"}, "zaxis": {"backgroundcolor": "#E5ECF6", "gridcolor": "white", "gridwidth": 2, "linecolor": "white", "showbackground": true, "ticks": "", "zerolinecolor": "white"}}, "shapedefaults": {"line": {"color": "#2a3f5f"}}, "ternary": {"aaxis": {"gridcolor": "white", "linecolor": "white", "ticks": ""}, "baxis": {"gridcolor": "white", "linecolor": "white", "ticks": ""}, "bgcolor": "#E5ECF6", "caxis": {"gridcolor": "white", "linecolor": "white", "ticks": ""}}, "title": {"x": 0.05}, "xaxis": {"automargin": true, "gridcolor": "white", "linecolor": "white", "ticks": "", "title": {"standoff": 15}, "zerolinecolor": "white", "zerolinewidth": 2}, "yaxis": {"automargin": true, "gridcolor": "white", "linecolor": "white", "ticks": "", "title": {"standoff": 15}, "zerolinecolor": "white", "zerolinewidth": 2}}}, "xaxis": {"anchor": "y", "domain": [0.0, 1.0], "title": {"text": "x"}}, "yaxis": {"anchor": "x", "domain": [0.0, 1.0], "title": {"text": "y"}}},
                        {"responsive": true}
                    ).then(function(){

var gd = document.getElementById('cb5f820b-68d6-4c20-bc46-d80a3ad36194');
var x = new MutationObserver(function (mutations, observer) {{
        var display = window.getComputedStyle(gd).display;
        if (!display || display === 'none') {{
            console.log([gd, 'removed!']);
            Plotly.purge(gd);
            observer.disconnect();
        }}
}});

// Listen for the removal of the full notebook cells
var notebookContainer = gd.closest('#notebook-container');
if (notebookContainer) {{
    x.observe(notebookContainer, {childList: true});
}}

// Listen for the clearing of the current output cell
var outputEl = gd.closest('.output');
if (outputEl) {{
    x.observe(outputEl, {childList: true});
}}

                        })
                };
                });
            </script>
        </div>



```python
# fig = px.scatter_3d(x=df['SepalLengthCm'], y=df['PetalLengthCm'],z=df['PetalWidthCm'],color = df['Species'])
# fig.show()
```


<div>


            <div id="3553d2ac-18bc-41df-85da-8bd32e9877cd" class="plotly-graph-div" style="height:525px; width:100%;"></div>
            <script type="text/javascript">
                require(["plotly"], function(Plotly) {
                    window.PLOTLYENV=window.PLOTLYENV || {};

                if (document.getElementById("3553d2ac-18bc-41df-85da-8bd32e9877cd")) {
                    Plotly.newPlot(
                        '3553d2ac-18bc-41df-85da-8bd32e9877cd',
                        [{"hoverlabel": {"namelength": 0}, "hovertemplate": "color=Iris-virginica<br>x=%{x}<br>y=%{y}<br>z=%{z}", "legendgroup": "color=Iris-virginica", "marker": {"color": "#636efa", "symbol": "circle"}, "mode": "markers", "name": "color=Iris-virginica", "scene": "scene", "showlegend": true, "type": "scatter3d", "x": [6.7, 5.8, 6.9, 7.7, 6.4, 5.8, 7.7, 6.2, 6.3, 6.7, 6.7, 6.2, 6.5, 6.9, 6.1, 7.1, 6.8, 7.7, 7.9, 7.2, 6.3, 6.3, 5.9, 6.0, 4.9, 6.4, 6.3, 6.0, 5.7, 7.2, 6.9, 6.4, 7.2, 6.4, 6.5, 6.3, 6.5, 5.6, 6.3, 6.1, 6.5, 6.8, 6.7, 6.7, 7.6, 5.8, 7.4, 7.3, 7.7, 6.4], "y": [5.8, 5.1, 5.7, 6.7, 5.5, 5.1, 6.9, 4.8, 5.1, 5.6, 5.7, 5.4, 5.5, 5.4, 4.9, 5.9, 5.5, 6.1, 6.4, 6.0, 5.6, 5.6, 5.1, 5.0, 4.5, 5.6, 4.9, 4.8, 5.0, 5.8, 5.1, 5.3, 6.1, 5.6, 5.2, 6.0, 5.8, 4.9, 5.0, 5.6, 5.1, 5.9, 5.2, 5.7, 6.6, 5.1, 6.1, 6.3, 6.7, 5.3], "z": [1.8, 1.9, 2.3, 2.2, 1.8, 2.4, 2.3, 1.8, 1.5, 2.4, 2.1, 2.3, 1.8, 2.1, 1.8, 2.1, 2.1, 2.3, 2.0, 1.8, 2.4, 1.8, 1.8, 1.5, 1.7, 2.1, 1.8, 1.8, 2.0, 1.6, 2.3, 2.3, 2.5, 2.2, 2.0, 2.5, 2.2, 2.0, 1.9, 1.4, 2.0, 2.3, 2.3, 2.5, 2.1, 1.9, 1.9, 1.8, 2.0, 1.9]}, {"hoverlabel": {"namelength": 0}, "hovertemplate": "color=Iris-versicolor<br>x=%{x}<br>y=%{y}<br>z=%{z}", "legendgroup": "color=Iris-versicolor", "marker": {"color": "#EF553B", "symbol": "circle"}, "mode": "markers", "name": "color=Iris-versicolor", "scene": "scene", "showlegend": true, "type": "scatter3d", "x": [5.7, 6.0, 7.0, 5.7, 5.6, 5.7, 6.1, 6.1, 6.2, 6.3, 5.5, 6.0, 5.5, 6.3, 6.0, 6.6, 5.8, 5.5, 5.6, 5.8, 6.1, 6.0, 5.0, 6.4, 5.4, 6.5, 5.1, 5.6, 6.7, 6.3, 5.6, 6.7, 6.1, 5.5, 6.9, 5.5, 5.7, 4.9, 6.7, 6.4, 5.2, 5.0, 6.8, 5.7, 5.9, 5.9, 6.2, 5.6, 5.8, 6.6], "y": [4.5, 5.1, 4.7, 3.5, 3.6, 4.2, 4.7, 4.7, 4.5, 4.9, 4.4, 4.5, 4.0, 4.7, 4.0, 4.6, 3.9, 3.8, 4.1, 4.1, 4.6, 4.5, 3.5, 4.5, 4.5, 4.6, 3.0, 3.9, 4.4, 4.4, 4.5, 5.0, 4.0, 3.7, 4.9, 4.0, 4.2, 3.3, 4.7, 4.3, 3.9, 3.3, 4.8, 4.1, 4.2, 4.8, 4.3, 4.2, 4.0, 4.4], "z": [1.3, 1.6, 1.4, 1.0, 1.3, 1.2, 1.2, 1.4, 1.5, 1.5, 1.2, 1.6, 1.3, 1.6, 1.0, 1.3, 1.2, 1.1, 1.3, 1.0, 1.4, 1.5, 1.0, 1.5, 1.5, 1.5, 1.1, 1.1, 1.4, 1.3, 1.5, 1.7, 1.3, 1.0, 1.5, 1.3, 1.3, 1.0, 1.5, 1.3, 1.4, 1.0, 1.4, 1.3, 1.5, 1.8, 1.3, 1.3, 1.2, 1.4]}, {"hoverlabel": {"namelength": 0}, "hovertemplate": "color=Iris-setosa<br>x=%{x}<br>y=%{y}<br>z=%{z}", "legendgroup": "color=Iris-setosa", "marker": {"color": "#00cc96", "symbol": "circle"}, "mode": "markers", "name": "color=Iris-setosa", "scene": "scene", "showlegend": true, "type": "scatter3d", "x": [4.3, 5.0, 5.4, 5.2, 4.4, 5.1, 5.1, 5.2, 5.0, 5.4, 5.1, 5.1, 5.0, 4.4, 5.2, 5.0, 5.4, 4.6, 5.8, 5.4, 5.0, 4.8, 4.6, 5.0, 4.9, 5.5, 5.7, 4.8, 5.1, 5.3, 5.5, 4.7, 4.9, 4.8, 5.1, 4.8, 4.4, 4.8, 5.0, 4.6, 5.7, 4.9, 4.5, 4.6, 5.1, 5.4, 5.1, 5.0, 4.9, 4.7], "y": [1.1, 1.4, 1.7, 1.5, 1.4, 1.7, 1.9, 1.4, 1.6, 1.5, 1.5, 1.4, 1.5, 1.3, 1.5, 1.6, 1.7, 1.4, 1.2, 1.3, 1.6, 1.6, 1.5, 1.2, 1.4, 1.3, 1.5, 1.4, 1.6, 1.5, 1.4, 1.6, 1.5, 1.6, 1.4, 1.9, 1.3, 1.4, 1.4, 1.0, 1.7, 1.5, 1.3, 1.4, 1.5, 1.5, 1.5, 1.3, 1.5, 1.3], "z": [0.1, 0.2, 0.4, 0.2, 0.2, 0.5, 0.4, 0.2, 0.4, 0.4, 0.2, 0.2, 0.2, 0.2, 0.1, 0.6, 0.2, 0.2, 0.2, 0.4, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.4, 0.3, 0.2, 0.2, 0.2, 0.2, 0.1, 0.2, 0.3, 0.2, 0.2, 0.1, 0.2, 0.2, 0.3, 0.1, 0.3, 0.3, 0.3, 0.2, 0.4, 0.3, 0.1, 0.2]}],
                        {"legend": {"tracegroupgap": 0}, "margin": {"t": 60}, "scene": {"domain": {"x": [0.0, 1.0], "y": [0.0, 1.0]}, "xaxis": {"title": {"text": "x"}}, "yaxis": {"title": {"text": "y"}}, "zaxis": {"title": {"text": "z"}}}, "template": {"data": {"bar": [{"error_x": {"color": "#2a3f5f"}, "error_y": {"color": "#2a3f5f"}, "marker": {"line": {"color": "#E5ECF6", "width": 0.5}}, "type": "bar"}], "barpolar": [{"marker": {"line": {"color": "#E5ECF6", "width": 0.5}}, "type": "barpolar"}], "carpet": [{"aaxis": {"endlinecolor": "#2a3f5f", "gridcolor": "white", "linecolor": "white", "minorgridcolor": "white", "startlinecolor": "#2a3f5f"}, "baxis": {"endlinecolor": "#2a3f5f", "gridcolor": "white", "linecolor": "white", "minorgridcolor": "white", "startlinecolor": "#2a3f5f"}, "type": "carpet"}], "choropleth": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "type": "choropleth"}], "contour": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "colorscale": [[0.0, "#0d0887"], [0.1111111111111111, "#46039f"], [0.2222222222222222, "#7201a8"], [0.3333333333333333, "#9c179e"], [0.4444444444444444, "#bd3786"], [0.5555555555555556, "#d8576b"], [0.6666666666666666, "#ed7953"], [0.7777777777777778, "#fb9f3a"], [0.8888888888888888, "#fdca26"], [1.0, "#f0f921"]], "type": "contour"}], "contourcarpet": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "type": "contourcarpet"}], "heatmap": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "colorscale": [[0.0, "#0d0887"], [0.1111111111111111, "#46039f"], [0.2222222222222222, "#7201a8"], [0.3333333333333333, "#9c179e"], [0.4444444444444444, "#bd3786"], [0.5555555555555556, "#d8576b"], [0.6666666666666666, "#ed7953"], [0.7777777777777778, "#fb9f3a"], [0.8888888888888888, "#fdca26"], [1.0, "#f0f921"]], "type": "heatmap"}], "heatmapgl": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "colorscale": [[0.0, "#0d0887"], [0.1111111111111111, "#46039f"], [0.2222222222222222, "#7201a8"], [0.3333333333333333, "#9c179e"], [0.4444444444444444, "#bd3786"], [0.5555555555555556, "#d8576b"], [0.6666666666666666, "#ed7953"], [0.7777777777777778, "#fb9f3a"], [0.8888888888888888, "#fdca26"], [1.0, "#f0f921"]], "type": "heatmapgl"}], "histogram": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "histogram"}], "histogram2d": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "colorscale": [[0.0, "#0d0887"], [0.1111111111111111, "#46039f"], [0.2222222222222222, "#7201a8"], [0.3333333333333333, "#9c179e"], [0.4444444444444444, "#bd3786"], [0.5555555555555556, "#d8576b"], [0.6666666666666666, "#ed7953"], [0.7777777777777778, "#fb9f3a"], [0.8888888888888888, "#fdca26"], [1.0, "#f0f921"]], "type": "histogram2d"}], "histogram2dcontour": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "colorscale": [[0.0, "#0d0887"], [0.1111111111111111, "#46039f"], [0.2222222222222222, "#7201a8"], [0.3333333333333333, "#9c179e"], [0.4444444444444444, "#bd3786"], [0.5555555555555556, "#d8576b"], [0.6666666666666666, "#ed7953"], [0.7777777777777778, "#fb9f3a"], [0.8888888888888888, "#fdca26"], [1.0, "#f0f921"]], "type": "histogram2dcontour"}], "mesh3d": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "type": "mesh3d"}], "parcoords": [{"line": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "parcoords"}], "pie": [{"automargin": true, "type": "pie"}], "scatter": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scatter"}], "scatter3d": [{"line": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scatter3d"}], "scattercarpet": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scattercarpet"}], "scattergeo": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scattergeo"}], "scattergl": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scattergl"}], "scattermapbox": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scattermapbox"}], "scatterpolar": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scatterpolar"}], "scatterpolargl": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scatterpolargl"}], "scatterternary": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scatterternary"}], "surface": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "colorscale": [[0.0, "#0d0887"], [0.1111111111111111, "#46039f"], [0.2222222222222222, "#7201a8"], [0.3333333333333333, "#9c179e"], [0.4444444444444444, "#bd3786"], [0.5555555555555556, "#d8576b"], [0.6666666666666666, "#ed7953"], [0.7777777777777778, "#fb9f3a"], [0.8888888888888888, "#fdca26"], [1.0, "#f0f921"]], "type": "surface"}], "table": [{"cells": {"fill": {"color": "#EBF0F8"}, "line": {"color": "white"}}, "header": {"fill": {"color": "#C8D4E3"}, "line": {"color": "white"}}, "type": "table"}]}, "layout": {"annotationdefaults": {"arrowcolor": "#2a3f5f", "arrowhead": 0, "arrowwidth": 1}, "coloraxis": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "colorscale": {"diverging": [[0, "#8e0152"], [0.1, "#c51b7d"], [0.2, "#de77ae"], [0.3, "#f1b6da"], [0.4, "#fde0ef"], [0.5, "#f7f7f7"], [0.6, "#e6f5d0"], [0.7, "#b8e186"], [0.8, "#7fbc41"], [0.9, "#4d9221"], [1, "#276419"]], "sequential": [[0.0, "#0d0887"], [0.1111111111111111, "#46039f"], [0.2222222222222222, "#7201a8"], [0.3333333333333333, "#9c179e"], [0.4444444444444444, "#bd3786"], [0.5555555555555556, "#d8576b"], [0.6666666666666666, "#ed7953"], [0.7777777777777778, "#fb9f3a"], [0.8888888888888888, "#fdca26"], [1.0, "#f0f921"]], "sequentialminus": [[0.0, "#0d0887"], [0.1111111111111111, "#46039f"], [0.2222222222222222, "#7201a8"], [0.3333333333333333, "#9c179e"], [0.4444444444444444, "#bd3786"], [0.5555555555555556, "#d8576b"], [0.6666666666666666, "#ed7953"], [0.7777777777777778, "#fb9f3a"], [0.8888888888888888, "#fdca26"], [1.0, "#f0f921"]]}, "colorway": ["#636efa", "#EF553B", "#00cc96", "#ab63fa", "#FFA15A", "#19d3f3", "#FF6692", "#B6E880", "#FF97FF", "#FECB52"], "font": {"color": "#2a3f5f"}, "geo": {"bgcolor": "white", "lakecolor": "white", "landcolor": "#E5ECF6", "showlakes": true, "showland": true, "subunitcolor": "white"}, "hoverlabel": {"align": "left"}, "hovermode": "closest", "mapbox": {"style": "light"}, "paper_bgcolor": "white", "plot_bgcolor": "#E5ECF6", "polar": {"angularaxis": {"gridcolor": "white", "linecolor": "white", "ticks": ""}, "bgcolor": "#E5ECF6", "radialaxis": {"gridcolor": "white", "linecolor": "white", "ticks": ""}}, "scene": {"xaxis": {"backgroundcolor": "#E5ECF6", "gridcolor": "white", "gridwidth": 2, "linecolor": "white", "showbackground": true, "ticks": "", "zerolinecolor": "white"}, "yaxis": {"backgroundcolor": "#E5ECF6", "gridcolor": "white", "gridwidth": 2, "linecolor": "white", "showbackground": true, "ticks": "", "zerolinecolor": "white"}, "zaxis": {"backgroundcolor": "#E5ECF6", "gridcolor": "white", "gridwidth": 2, "linecolor": "white", "showbackground": true, "ticks": "", "zerolinecolor": "white"}}, "shapedefaults": {"line": {"color": "#2a3f5f"}}, "ternary": {"aaxis": {"gridcolor": "white", "linecolor": "white", "ticks": ""}, "baxis": {"gridcolor": "white", "linecolor": "white", "ticks": ""}, "bgcolor": "#E5ECF6", "caxis": {"gridcolor": "white", "linecolor": "white", "ticks": ""}}, "title": {"x": 0.05}, "xaxis": {"automargin": true, "gridcolor": "white", "linecolor": "white", "ticks": "", "title": {"standoff": 15}, "zerolinecolor": "white", "zerolinewidth": 2}, "yaxis": {"automargin": true, "gridcolor": "white", "linecolor": "white", "ticks": "", "title": {"standoff": 15}, "zerolinecolor": "white", "zerolinewidth": 2}}}},
                        {"responsive": true}
                    ).then(function(){

var gd = document.getElementById('3553d2ac-18bc-41df-85da-8bd32e9877cd');
var x = new MutationObserver(function (mutations, observer) {{
        var display = window.getComputedStyle(gd).display;
        if (!display || display === 'none') {{
            console.log([gd, 'removed!']);
            Plotly.purge(gd);
            observer.disconnect();
        }}
}});

// Listen for the removal of the full notebook cells
var notebookContainer = gd.closest('#notebook-container');
if (notebookContainer) {{
    x.observe(notebookContainer, {childList: true});
}}

// Listen for the clearing of the current output cell
var outputEl = gd.closest('.output');
if (outputEl) {{
    x.observe(outputEl, {childList: true});
}}

                        })
                };
                });
            </script>
        </div>


# Build Model


```python
def euclidean_distance(pt1, pt2):
    """
    Calculate the Euclidean Distance between two points.
    """
    distance = 0.0
    for i in range(len(pt1)-1):
        distance += (pt1[i] - pt2[i])**2
    return sqrt(distance)

pt1 = [0,0]
pt2 = [1,0]
d = euclidean_distance(pt1,pt2)
print('The Euclidean Distance between {} and {} is {}'.format(pt1,pt2,d))
```

    The Euclidean Distance between [0, 0] and [1, 0] is 1.0



```python
def most_freq_cls(neighbors):
    """
    Return the predict class given K neighbors
    """
    max_freq = 0
    res = neighbors[0]
    for i in neighbors:
        freq = neighbors.count(i)
        if freq > max_freq:
            max_freq = freq
            res = i
    return res
    
neighbors = ['Iris-virginica','Iris-virginica','Iris-virginica'
             'Iris-setosa','Iris-setosa','Iris-setosa','Iris-setosa','Iris-setosa','Iris-setosa',
             'Iris-versicolor','Iris-versicolor']

print('The Predict Class is {}'.format(most_freq_cls(neighbors)))
```

    The Predict Class is Iris-setosa



```python
def knn(input_pt, training_data, num_neighbors):
    """
    Return the label of input points based on 
    training data using KNN
    """
    distances = list()
    
    for pt in training_data:
        dist = euclidean_distance(input_pt,pt)
        distances.append((pt, dist))
        
    distances.sort(key=lambda x:x[1])
    neighbors = list()

    for neighbor in distances[0:num_neighbors]:
        neighbors.append(neighbor[0][-1])

    predicted_label = most_freq_cls(neighbors)
    return predicted_label

input_pt = [6.1,2.8,4.7,1.2]
k=1
knn(input_pt, training_data, k) # True class is 'Iris-versicolor'
```




    'Iris-versicolor'




# Make Prediction


```python
k=1

train_prediction =[]
for row in training_data[:,0:4]:
    predict_class = knn(row,training_data,k)
    train_prediction.append(predict_class)
    
train_accuracy = sum(train_prediction == training_data[:,4])/len(train_prediction)
print('Training Accuracy is {}'.format(train_accuracy))


test_prediction =[]
for row in testing_data:
    predict_class = knn(row,training_data,k)
    test_prediction.append(predict_class)
    
test_accuracy = sum(test_prediction == testing_data[:,4])/len(test_prediction)
print('Testing Accuracy is {}'.format(test_accuracy))
```

    Training Accuracy is 1.0
    Testing Accuracy is 0.9666666666666667


# Visualiza Training Result


```python
fig = px.scatter(x=training_data[:,2],y=training_data[:,3],color = training_data[:,4])
fig.show()
```


<div>


            <div id="9903a970-d894-425e-a347-6648dbd42cc2" class="plotly-graph-div" style="height:525px; width:100%;"></div>
            <script type="text/javascript">
                require(["plotly"], function(Plotly) {
                    window.PLOTLYENV=window.PLOTLYENV || {};

                if (document.getElementById("9903a970-d894-425e-a347-6648dbd42cc2")) {
                    Plotly.newPlot(
                        '9903a970-d894-425e-a347-6648dbd42cc2',
                        [{"hoverlabel": {"namelength": 0}, "hovertemplate": "color=Iris-virginica<br>x=%{x}<br>y=%{y}", "legendgroup": "color=Iris-virginica", "marker": {"color": "#636efa", "symbol": "circle"}, "mode": "markers", "name": "color=Iris-virginica", "showlegend": true, "type": "scatter", "x": [5.8, 5.1, 5.7, 6.7, 5.5, 5.1, 6.9, 4.8, 5.1, 5.6, 5.7, 5.4, 5.5, 5.4, 4.9, 5.9, 5.5, 6.1, 6.4, 6.0, 5.6, 5.6, 5.1, 5.0, 4.5, 5.6, 4.9, 4.8, 5.0, 5.8, 5.1, 5.3, 6.1, 5.6, 5.2, 6.0, 5.8, 4.9, 5.0, 5.6, 5.1, 5.9], "xaxis": "x", "y": [1.8, 1.9, 2.3, 2.2, 1.8, 2.4, 2.3, 1.8, 1.5, 2.4, 2.1, 2.3, 1.8, 2.1, 1.8, 2.1, 2.1, 2.3, 2.0, 1.8, 2.4, 1.8, 1.8, 1.5, 1.7, 2.1, 1.8, 1.8, 2.0, 1.6, 2.3, 2.3, 2.5, 2.2, 2.0, 2.5, 2.2, 2.0, 1.9, 1.4, 2.0, 2.3], "yaxis": "y"}, {"hoverlabel": {"namelength": 0}, "hovertemplate": "color=Iris-versicolor<br>x=%{x}<br>y=%{y}", "legendgroup": "color=Iris-versicolor", "marker": {"color": "#EF553B", "symbol": "circle"}, "mode": "markers", "name": "color=Iris-versicolor", "showlegend": true, "type": "scatter", "x": [4.5, 5.1, 4.7, 3.5, 3.6, 4.2, 4.7, 4.7, 4.5, 4.9, 4.4, 4.5, 4.0, 4.7, 4.0, 4.6, 3.9, 3.8, 4.1, 4.1, 4.6, 4.5, 3.5, 4.5, 4.5, 4.6, 3.0, 3.9, 4.4, 4.4, 4.5, 5.0, 4.0, 3.7, 4.9, 4.0, 4.2, 3.3, 4.7, 4.3, 3.9], "xaxis": "x", "y": [1.3, 1.6, 1.4, 1.0, 1.3, 1.2, 1.2, 1.4, 1.5, 1.5, 1.2, 1.6, 1.3, 1.6, 1.0, 1.3, 1.2, 1.1, 1.3, 1.0, 1.4, 1.5, 1.0, 1.5, 1.5, 1.5, 1.1, 1.1, 1.4, 1.3, 1.5, 1.7, 1.3, 1.0, 1.5, 1.3, 1.3, 1.0, 1.5, 1.3, 1.4], "yaxis": "y"}, {"hoverlabel": {"namelength": 0}, "hovertemplate": "color=Iris-setosa<br>x=%{x}<br>y=%{y}", "legendgroup": "color=Iris-setosa", "marker": {"color": "#00cc96", "symbol": "circle"}, "mode": "markers", "name": "color=Iris-setosa", "showlegend": true, "type": "scatter", "x": [1.1, 1.4, 1.7, 1.5, 1.4, 1.7, 1.9, 1.4, 1.6, 1.5, 1.5, 1.4, 1.5, 1.3, 1.5, 1.6, 1.7, 1.4, 1.2, 1.3, 1.6, 1.6, 1.5, 1.2, 1.4, 1.3, 1.5, 1.4, 1.6, 1.5, 1.4, 1.6, 1.5, 1.6, 1.4, 1.9, 1.3, 1.4], "xaxis": "x", "y": [0.1, 0.2, 0.4, 0.2, 0.2, 0.5, 0.4, 0.2, 0.4, 0.4, 0.2, 0.2, 0.2, 0.2, 0.1, 0.6, 0.2, 0.2, 0.2, 0.4, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.4, 0.3, 0.2, 0.2, 0.2, 0.2, 0.1, 0.2, 0.3, 0.2, 0.2, 0.1], "yaxis": "y"}],
                        {"legend": {"tracegroupgap": 0}, "margin": {"t": 60}, "template": {"data": {"bar": [{"error_x": {"color": "#2a3f5f"}, "error_y": {"color": "#2a3f5f"}, "marker": {"line": {"color": "#E5ECF6", "width": 0.5}}, "type": "bar"}], "barpolar": [{"marker": {"line": {"color": "#E5ECF6", "width": 0.5}}, "type": "barpolar"}], "carpet": [{"aaxis": {"endlinecolor": "#2a3f5f", "gridcolor": "white", "linecolor": "white", "minorgridcolor": "white", "startlinecolor": "#2a3f5f"}, "baxis": {"endlinecolor": "#2a3f5f", "gridcolor": "white", "linecolor": "white", "minorgridcolor": "white", "startlinecolor": "#2a3f5f"}, "type": "carpet"}], "choropleth": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "type": "choropleth"}], "contour": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "colorscale": [[0.0, "#0d0887"], [0.1111111111111111, "#46039f"], [0.2222222222222222, "#7201a8"], [0.3333333333333333, "#9c179e"], [0.4444444444444444, "#bd3786"], [0.5555555555555556, "#d8576b"], [0.6666666666666666, "#ed7953"], [0.7777777777777778, "#fb9f3a"], [0.8888888888888888, "#fdca26"], [1.0, "#f0f921"]], "type": "contour"}], "contourcarpet": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "type": "contourcarpet"}], "heatmap": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "colorscale": [[0.0, "#0d0887"], [0.1111111111111111, "#46039f"], [0.2222222222222222, "#7201a8"], [0.3333333333333333, "#9c179e"], [0.4444444444444444, "#bd3786"], [0.5555555555555556, "#d8576b"], [0.6666666666666666, "#ed7953"], [0.7777777777777778, "#fb9f3a"], [0.8888888888888888, "#fdca26"], [1.0, "#f0f921"]], "type": "heatmap"}], "heatmapgl": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "colorscale": [[0.0, "#0d0887"], [0.1111111111111111, "#46039f"], [0.2222222222222222, "#7201a8"], [0.3333333333333333, "#9c179e"], [0.4444444444444444, "#bd3786"], [0.5555555555555556, "#d8576b"], [0.6666666666666666, "#ed7953"], [0.7777777777777778, "#fb9f3a"], [0.8888888888888888, "#fdca26"], [1.0, "#f0f921"]], "type": "heatmapgl"}], "histogram": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "histogram"}], "histogram2d": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "colorscale": [[0.0, "#0d0887"], [0.1111111111111111, "#46039f"], [0.2222222222222222, "#7201a8"], [0.3333333333333333, "#9c179e"], [0.4444444444444444, "#bd3786"], [0.5555555555555556, "#d8576b"], [0.6666666666666666, "#ed7953"], [0.7777777777777778, "#fb9f3a"], [0.8888888888888888, "#fdca26"], [1.0, "#f0f921"]], "type": "histogram2d"}], "histogram2dcontour": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "colorscale": [[0.0, "#0d0887"], [0.1111111111111111, "#46039f"], [0.2222222222222222, "#7201a8"], [0.3333333333333333, "#9c179e"], [0.4444444444444444, "#bd3786"], [0.5555555555555556, "#d8576b"], [0.6666666666666666, "#ed7953"], [0.7777777777777778, "#fb9f3a"], [0.8888888888888888, "#fdca26"], [1.0, "#f0f921"]], "type": "histogram2dcontour"}], "mesh3d": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "type": "mesh3d"}], "parcoords": [{"line": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "parcoords"}], "pie": [{"automargin": true, "type": "pie"}], "scatter": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scatter"}], "scatter3d": [{"line": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scatter3d"}], "scattercarpet": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scattercarpet"}], "scattergeo": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scattergeo"}], "scattergl": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scattergl"}], "scattermapbox": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scattermapbox"}], "scatterpolar": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scatterpolar"}], "scatterpolargl": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scatterpolargl"}], "scatterternary": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scatterternary"}], "surface": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "colorscale": [[0.0, "#0d0887"], [0.1111111111111111, "#46039f"], [0.2222222222222222, "#7201a8"], [0.3333333333333333, "#9c179e"], [0.4444444444444444, "#bd3786"], [0.5555555555555556, "#d8576b"], [0.6666666666666666, "#ed7953"], [0.7777777777777778, "#fb9f3a"], [0.8888888888888888, "#fdca26"], [1.0, "#f0f921"]], "type": "surface"}], "table": [{"cells": {"fill": {"color": "#EBF0F8"}, "line": {"color": "white"}}, "header": {"fill": {"color": "#C8D4E3"}, "line": {"color": "white"}}, "type": "table"}]}, "layout": {"annotationdefaults": {"arrowcolor": "#2a3f5f", "arrowhead": 0, "arrowwidth": 1}, "coloraxis": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "colorscale": {"diverging": [[0, "#8e0152"], [0.1, "#c51b7d"], [0.2, "#de77ae"], [0.3, "#f1b6da"], [0.4, "#fde0ef"], [0.5, "#f7f7f7"], [0.6, "#e6f5d0"], [0.7, "#b8e186"], [0.8, "#7fbc41"], [0.9, "#4d9221"], [1, "#276419"]], "sequential": [[0.0, "#0d0887"], [0.1111111111111111, "#46039f"], [0.2222222222222222, "#7201a8"], [0.3333333333333333, "#9c179e"], [0.4444444444444444, "#bd3786"], [0.5555555555555556, "#d8576b"], [0.6666666666666666, "#ed7953"], [0.7777777777777778, "#fb9f3a"], [0.8888888888888888, "#fdca26"], [1.0, "#f0f921"]], "sequentialminus": [[0.0, "#0d0887"], [0.1111111111111111, "#46039f"], [0.2222222222222222, "#7201a8"], [0.3333333333333333, "#9c179e"], [0.4444444444444444, "#bd3786"], [0.5555555555555556, "#d8576b"], [0.6666666666666666, "#ed7953"], [0.7777777777777778, "#fb9f3a"], [0.8888888888888888, "#fdca26"], [1.0, "#f0f921"]]}, "colorway": ["#636efa", "#EF553B", "#00cc96", "#ab63fa", "#FFA15A", "#19d3f3", "#FF6692", "#B6E880", "#FF97FF", "#FECB52"], "font": {"color": "#2a3f5f"}, "geo": {"bgcolor": "white", "lakecolor": "white", "landcolor": "#E5ECF6", "showlakes": true, "showland": true, "subunitcolor": "white"}, "hoverlabel": {"align": "left"}, "hovermode": "closest", "mapbox": {"style": "light"}, "paper_bgcolor": "white", "plot_bgcolor": "#E5ECF6", "polar": {"angularaxis": {"gridcolor": "white", "linecolor": "white", "ticks": ""}, "bgcolor": "#E5ECF6", "radialaxis": {"gridcolor": "white", "linecolor": "white", "ticks": ""}}, "scene": {"xaxis": {"backgroundcolor": "#E5ECF6", "gridcolor": "white", "gridwidth": 2, "linecolor": "white", "showbackground": true, "ticks": "", "zerolinecolor": "white"}, "yaxis": {"backgroundcolor": "#E5ECF6", "gridcolor": "white", "gridwidth": 2, "linecolor": "white", "showbackground": true, "ticks": "", "zerolinecolor": "white"}, "zaxis": {"backgroundcolor": "#E5ECF6", "gridcolor": "white", "gridwidth": 2, "linecolor": "white", "showbackground": true, "ticks": "", "zerolinecolor": "white"}}, "shapedefaults": {"line": {"color": "#2a3f5f"}}, "ternary": {"aaxis": {"gridcolor": "white", "linecolor": "white", "ticks": ""}, "baxis": {"gridcolor": "white", "linecolor": "white", "ticks": ""}, "bgcolor": "#E5ECF6", "caxis": {"gridcolor": "white", "linecolor": "white", "ticks": ""}}, "title": {"x": 0.05}, "xaxis": {"automargin": true, "gridcolor": "white", "linecolor": "white", "ticks": "", "title": {"standoff": 15}, "zerolinecolor": "white", "zerolinewidth": 2}, "yaxis": {"automargin": true, "gridcolor": "white", "linecolor": "white", "ticks": "", "title": {"standoff": 15}, "zerolinecolor": "white", "zerolinewidth": 2}}}, "xaxis": {"anchor": "y", "domain": [0.0, 1.0], "title": {"text": "x"}}, "yaxis": {"anchor": "x", "domain": [0.0, 1.0], "title": {"text": "y"}}},
                        {"responsive": true}
                    ).then(function(){

var gd = document.getElementById('9903a970-d894-425e-a347-6648dbd42cc2');
var x = new MutationObserver(function (mutations, observer) {{
        var display = window.getComputedStyle(gd).display;
        if (!display || display === 'none') {{
            console.log([gd, 'removed!']);
            Plotly.purge(gd);
            observer.disconnect();
        }}
}});

// Listen for the removal of the full notebook cells
var notebookContainer = gd.closest('#notebook-container');
if (notebookContainer) {{
    x.observe(notebookContainer, {childList: true});
}}

// Listen for the clearing of the current output cell
var outputEl = gd.closest('.output');
if (outputEl) {{
    x.observe(outputEl, {childList: true});
}}

                        })
                };
                });
            </script>
        </div>



```python
fig = px.scatter(x=training_data[:,2],y=training_data[:,3],color = train_prediction)
fig.show()
```


<div>


            <div id="7e92fb9d-37f0-4037-8d65-6fe1acc889d4" class="plotly-graph-div" style="height:525px; width:100%;"></div>
            <script type="text/javascript">
                require(["plotly"], function(Plotly) {
                    window.PLOTLYENV=window.PLOTLYENV || {};

                if (document.getElementById("7e92fb9d-37f0-4037-8d65-6fe1acc889d4")) {
                    Plotly.newPlot(
                        '7e92fb9d-37f0-4037-8d65-6fe1acc889d4',
                        [{"hoverlabel": {"namelength": 0}, "hovertemplate": "color=Iris-virginica<br>x=%{x}<br>y=%{y}", "legendgroup": "color=Iris-virginica", "marker": {"color": "#636efa", "symbol": "circle"}, "mode": "markers", "name": "color=Iris-virginica", "showlegend": true, "type": "scatter", "x": [5.8, 5.1, 5.7, 6.7, 5.5, 5.1, 6.9, 4.8, 5.1, 5.6, 5.7, 5.4, 5.5, 5.4, 4.9, 5.9, 5.5, 6.1, 6.4, 6.0, 5.6, 5.6, 5.1, 5.0, 4.5, 5.6, 4.9, 4.8, 5.0, 5.8, 5.1, 5.3, 6.1, 5.6, 5.2, 6.0, 5.8, 4.9, 5.0, 5.6, 5.1, 5.9], "xaxis": "x", "y": [1.8, 1.9, 2.3, 2.2, 1.8, 2.4, 2.3, 1.8, 1.5, 2.4, 2.1, 2.3, 1.8, 2.1, 1.8, 2.1, 2.1, 2.3, 2.0, 1.8, 2.4, 1.8, 1.8, 1.5, 1.7, 2.1, 1.8, 1.8, 2.0, 1.6, 2.3, 2.3, 2.5, 2.2, 2.0, 2.5, 2.2, 2.0, 1.9, 1.4, 2.0, 2.3], "yaxis": "y"}, {"hoverlabel": {"namelength": 0}, "hovertemplate": "color=Iris-versicolor<br>x=%{x}<br>y=%{y}", "legendgroup": "color=Iris-versicolor", "marker": {"color": "#EF553B", "symbol": "circle"}, "mode": "markers", "name": "color=Iris-versicolor", "showlegend": true, "type": "scatter", "x": [4.5, 5.1, 4.7, 3.5, 3.6, 4.2, 4.7, 4.7, 4.5, 4.9, 4.4, 4.5, 4.0, 4.7, 4.0, 4.6, 3.9, 3.8, 4.1, 4.1, 4.6, 4.5, 3.5, 4.5, 4.5, 4.6, 3.0, 3.9, 4.4, 4.4, 4.5, 5.0, 4.0, 3.7, 4.9, 4.0, 4.2, 3.3, 4.7, 4.3, 3.9], "xaxis": "x", "y": [1.3, 1.6, 1.4, 1.0, 1.3, 1.2, 1.2, 1.4, 1.5, 1.5, 1.2, 1.6, 1.3, 1.6, 1.0, 1.3, 1.2, 1.1, 1.3, 1.0, 1.4, 1.5, 1.0, 1.5, 1.5, 1.5, 1.1, 1.1, 1.4, 1.3, 1.5, 1.7, 1.3, 1.0, 1.5, 1.3, 1.3, 1.0, 1.5, 1.3, 1.4], "yaxis": "y"}, {"hoverlabel": {"namelength": 0}, "hovertemplate": "color=Iris-setosa<br>x=%{x}<br>y=%{y}", "legendgroup": "color=Iris-setosa", "marker": {"color": "#00cc96", "symbol": "circle"}, "mode": "markers", "name": "color=Iris-setosa", "showlegend": true, "type": "scatter", "x": [1.1, 1.4, 1.7, 1.5, 1.4, 1.7, 1.9, 1.4, 1.6, 1.5, 1.5, 1.4, 1.5, 1.3, 1.5, 1.6, 1.7, 1.4, 1.2, 1.3, 1.6, 1.6, 1.5, 1.2, 1.4, 1.3, 1.5, 1.4, 1.6, 1.5, 1.4, 1.6, 1.5, 1.6, 1.4, 1.9, 1.3, 1.4], "xaxis": "x", "y": [0.1, 0.2, 0.4, 0.2, 0.2, 0.5, 0.4, 0.2, 0.4, 0.4, 0.2, 0.2, 0.2, 0.2, 0.1, 0.6, 0.2, 0.2, 0.2, 0.4, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.4, 0.3, 0.2, 0.2, 0.2, 0.2, 0.1, 0.2, 0.3, 0.2, 0.2, 0.1], "yaxis": "y"}],
                        {"legend": {"tracegroupgap": 0}, "margin": {"t": 60}, "template": {"data": {"bar": [{"error_x": {"color": "#2a3f5f"}, "error_y": {"color": "#2a3f5f"}, "marker": {"line": {"color": "#E5ECF6", "width": 0.5}}, "type": "bar"}], "barpolar": [{"marker": {"line": {"color": "#E5ECF6", "width": 0.5}}, "type": "barpolar"}], "carpet": [{"aaxis": {"endlinecolor": "#2a3f5f", "gridcolor": "white", "linecolor": "white", "minorgridcolor": "white", "startlinecolor": "#2a3f5f"}, "baxis": {"endlinecolor": "#2a3f5f", "gridcolor": "white", "linecolor": "white", "minorgridcolor": "white", "startlinecolor": "#2a3f5f"}, "type": "carpet"}], "choropleth": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "type": "choropleth"}], "contour": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "colorscale": [[0.0, "#0d0887"], [0.1111111111111111, "#46039f"], [0.2222222222222222, "#7201a8"], [0.3333333333333333, "#9c179e"], [0.4444444444444444, "#bd3786"], [0.5555555555555556, "#d8576b"], [0.6666666666666666, "#ed7953"], [0.7777777777777778, "#fb9f3a"], [0.8888888888888888, "#fdca26"], [1.0, "#f0f921"]], "type": "contour"}], "contourcarpet": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "type": "contourcarpet"}], "heatmap": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "colorscale": [[0.0, "#0d0887"], [0.1111111111111111, "#46039f"], [0.2222222222222222, "#7201a8"], [0.3333333333333333, "#9c179e"], [0.4444444444444444, "#bd3786"], [0.5555555555555556, "#d8576b"], [0.6666666666666666, "#ed7953"], [0.7777777777777778, "#fb9f3a"], [0.8888888888888888, "#fdca26"], [1.0, "#f0f921"]], "type": "heatmap"}], "heatmapgl": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "colorscale": [[0.0, "#0d0887"], [0.1111111111111111, "#46039f"], [0.2222222222222222, "#7201a8"], [0.3333333333333333, "#9c179e"], [0.4444444444444444, "#bd3786"], [0.5555555555555556, "#d8576b"], [0.6666666666666666, "#ed7953"], [0.7777777777777778, "#fb9f3a"], [0.8888888888888888, "#fdca26"], [1.0, "#f0f921"]], "type": "heatmapgl"}], "histogram": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "histogram"}], "histogram2d": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "colorscale": [[0.0, "#0d0887"], [0.1111111111111111, "#46039f"], [0.2222222222222222, "#7201a8"], [0.3333333333333333, "#9c179e"], [0.4444444444444444, "#bd3786"], [0.5555555555555556, "#d8576b"], [0.6666666666666666, "#ed7953"], [0.7777777777777778, "#fb9f3a"], [0.8888888888888888, "#fdca26"], [1.0, "#f0f921"]], "type": "histogram2d"}], "histogram2dcontour": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "colorscale": [[0.0, "#0d0887"], [0.1111111111111111, "#46039f"], [0.2222222222222222, "#7201a8"], [0.3333333333333333, "#9c179e"], [0.4444444444444444, "#bd3786"], [0.5555555555555556, "#d8576b"], [0.6666666666666666, "#ed7953"], [0.7777777777777778, "#fb9f3a"], [0.8888888888888888, "#fdca26"], [1.0, "#f0f921"]], "type": "histogram2dcontour"}], "mesh3d": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "type": "mesh3d"}], "parcoords": [{"line": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "parcoords"}], "pie": [{"automargin": true, "type": "pie"}], "scatter": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scatter"}], "scatter3d": [{"line": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scatter3d"}], "scattercarpet": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scattercarpet"}], "scattergeo": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scattergeo"}], "scattergl": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scattergl"}], "scattermapbox": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scattermapbox"}], "scatterpolar": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scatterpolar"}], "scatterpolargl": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scatterpolargl"}], "scatterternary": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scatterternary"}], "surface": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "colorscale": [[0.0, "#0d0887"], [0.1111111111111111, "#46039f"], [0.2222222222222222, "#7201a8"], [0.3333333333333333, "#9c179e"], [0.4444444444444444, "#bd3786"], [0.5555555555555556, "#d8576b"], [0.6666666666666666, "#ed7953"], [0.7777777777777778, "#fb9f3a"], [0.8888888888888888, "#fdca26"], [1.0, "#f0f921"]], "type": "surface"}], "table": [{"cells": {"fill": {"color": "#EBF0F8"}, "line": {"color": "white"}}, "header": {"fill": {"color": "#C8D4E3"}, "line": {"color": "white"}}, "type": "table"}]}, "layout": {"annotationdefaults": {"arrowcolor": "#2a3f5f", "arrowhead": 0, "arrowwidth": 1}, "coloraxis": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "colorscale": {"diverging": [[0, "#8e0152"], [0.1, "#c51b7d"], [0.2, "#de77ae"], [0.3, "#f1b6da"], [0.4, "#fde0ef"], [0.5, "#f7f7f7"], [0.6, "#e6f5d0"], [0.7, "#b8e186"], [0.8, "#7fbc41"], [0.9, "#4d9221"], [1, "#276419"]], "sequential": [[0.0, "#0d0887"], [0.1111111111111111, "#46039f"], [0.2222222222222222, "#7201a8"], [0.3333333333333333, "#9c179e"], [0.4444444444444444, "#bd3786"], [0.5555555555555556, "#d8576b"], [0.6666666666666666, "#ed7953"], [0.7777777777777778, "#fb9f3a"], [0.8888888888888888, "#fdca26"], [1.0, "#f0f921"]], "sequentialminus": [[0.0, "#0d0887"], [0.1111111111111111, "#46039f"], [0.2222222222222222, "#7201a8"], [0.3333333333333333, "#9c179e"], [0.4444444444444444, "#bd3786"], [0.5555555555555556, "#d8576b"], [0.6666666666666666, "#ed7953"], [0.7777777777777778, "#fb9f3a"], [0.8888888888888888, "#fdca26"], [1.0, "#f0f921"]]}, "colorway": ["#636efa", "#EF553B", "#00cc96", "#ab63fa", "#FFA15A", "#19d3f3", "#FF6692", "#B6E880", "#FF97FF", "#FECB52"], "font": {"color": "#2a3f5f"}, "geo": {"bgcolor": "white", "lakecolor": "white", "landcolor": "#E5ECF6", "showlakes": true, "showland": true, "subunitcolor": "white"}, "hoverlabel": {"align": "left"}, "hovermode": "closest", "mapbox": {"style": "light"}, "paper_bgcolor": "white", "plot_bgcolor": "#E5ECF6", "polar": {"angularaxis": {"gridcolor": "white", "linecolor": "white", "ticks": ""}, "bgcolor": "#E5ECF6", "radialaxis": {"gridcolor": "white", "linecolor": "white", "ticks": ""}}, "scene": {"xaxis": {"backgroundcolor": "#E5ECF6", "gridcolor": "white", "gridwidth": 2, "linecolor": "white", "showbackground": true, "ticks": "", "zerolinecolor": "white"}, "yaxis": {"backgroundcolor": "#E5ECF6", "gridcolor": "white", "gridwidth": 2, "linecolor": "white", "showbackground": true, "ticks": "", "zerolinecolor": "white"}, "zaxis": {"backgroundcolor": "#E5ECF6", "gridcolor": "white", "gridwidth": 2, "linecolor": "white", "showbackground": true, "ticks": "", "zerolinecolor": "white"}}, "shapedefaults": {"line": {"color": "#2a3f5f"}}, "ternary": {"aaxis": {"gridcolor": "white", "linecolor": "white", "ticks": ""}, "baxis": {"gridcolor": "white", "linecolor": "white", "ticks": ""}, "bgcolor": "#E5ECF6", "caxis": {"gridcolor": "white", "linecolor": "white", "ticks": ""}}, "title": {"x": 0.05}, "xaxis": {"automargin": true, "gridcolor": "white", "linecolor": "white", "ticks": "", "title": {"standoff": 15}, "zerolinecolor": "white", "zerolinewidth": 2}, "yaxis": {"automargin": true, "gridcolor": "white", "linecolor": "white", "ticks": "", "title": {"standoff": 15}, "zerolinecolor": "white", "zerolinewidth": 2}}}, "xaxis": {"anchor": "y", "domain": [0.0, 1.0], "title": {"text": "x"}}, "yaxis": {"anchor": "x", "domain": [0.0, 1.0], "title": {"text": "y"}}},
                        {"responsive": true}
                    ).then(function(){

var gd = document.getElementById('7e92fb9d-37f0-4037-8d65-6fe1acc889d4');
var x = new MutationObserver(function (mutations, observer) {{
        var display = window.getComputedStyle(gd).display;
        if (!display || display === 'none') {{
            console.log([gd, 'removed!']);
            Plotly.purge(gd);
            observer.disconnect();
        }}
}});

// Listen for the removal of the full notebook cells
var notebookContainer = gd.closest('#notebook-container');
if (notebookContainer) {{
    x.observe(notebookContainer, {childList: true});
}}

// Listen for the clearing of the current output cell
var outputEl = gd.closest('.output');
if (outputEl) {{
    x.observe(outputEl, {childList: true});
}}

                        })
                };
                });
            </script>
        </div>


# Visualiza Testing Result


```python
fig = px.scatter(x=testing_data[:,2],y=testing_data[:,3],color = testing_data[:,4])
fig.show()
```


<div>


            <div id="ee28e4a0-9d14-4a8e-ac67-536bcd674fc0" class="plotly-graph-div" style="height:525px; width:100%;"></div>
            <script type="text/javascript">
                require(["plotly"], function(Plotly) {
                    window.PLOTLYENV=window.PLOTLYENV || {};

                if (document.getElementById("ee28e4a0-9d14-4a8e-ac67-536bcd674fc0")) {
                    Plotly.newPlot(
                        'ee28e4a0-9d14-4a8e-ac67-536bcd674fc0',
                        [{"hoverlabel": {"namelength": 0}, "hovertemplate": "color=Iris-setosa<br>x=%{x}<br>y=%{y}", "legendgroup": "color=Iris-setosa", "marker": {"color": "#636efa", "symbol": "circle"}, "mode": "markers", "name": "color=Iris-setosa", "showlegend": true, "type": "scatter", "x": [1.4, 1.4, 1.0, 1.7, 1.5, 1.3, 1.4, 1.5, 1.5, 1.5, 1.3, 1.5, 1.3], "xaxis": "x", "y": [0.1, 0.2, 0.2, 0.3, 0.1, 0.3, 0.3, 0.3, 0.2, 0.4, 0.3, 0.1, 0.2], "yaxis": "y"}, {"hoverlabel": {"namelength": 0}, "hovertemplate": "color=Iris-versicolor<br>x=%{x}<br>y=%{y}", "legendgroup": "color=Iris-versicolor", "marker": {"color": "#EF553B", "symbol": "circle"}, "mode": "markers", "name": "color=Iris-versicolor", "showlegend": true, "type": "scatter", "x": [3.3, 4.8, 4.1, 4.2, 4.8, 4.3, 4.2, 4.0, 4.4], "xaxis": "x", "y": [1.0, 1.4, 1.3, 1.5, 1.8, 1.3, 1.3, 1.2, 1.4], "yaxis": "y"}, {"hoverlabel": {"namelength": 0}, "hovertemplate": "color=Iris-virginica<br>x=%{x}<br>y=%{y}", "legendgroup": "color=Iris-virginica", "marker": {"color": "#00cc96", "symbol": "circle"}, "mode": "markers", "name": "color=Iris-virginica", "showlegend": true, "type": "scatter", "x": [5.2, 5.7, 6.6, 5.1, 6.1, 6.3, 6.7, 5.3], "xaxis": "x", "y": [2.3, 2.5, 2.1, 1.9, 1.9, 1.8, 2.0, 1.9], "yaxis": "y"}],
                        {"legend": {"tracegroupgap": 0}, "margin": {"t": 60}, "template": {"data": {"bar": [{"error_x": {"color": "#2a3f5f"}, "error_y": {"color": "#2a3f5f"}, "marker": {"line": {"color": "#E5ECF6", "width": 0.5}}, "type": "bar"}], "barpolar": [{"marker": {"line": {"color": "#E5ECF6", "width": 0.5}}, "type": "barpolar"}], "carpet": [{"aaxis": {"endlinecolor": "#2a3f5f", "gridcolor": "white", "linecolor": "white", "minorgridcolor": "white", "startlinecolor": "#2a3f5f"}, "baxis": {"endlinecolor": "#2a3f5f", "gridcolor": "white", "linecolor": "white", "minorgridcolor": "white", "startlinecolor": "#2a3f5f"}, "type": "carpet"}], "choropleth": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "type": "choropleth"}], "contour": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "colorscale": [[0.0, "#0d0887"], [0.1111111111111111, "#46039f"], [0.2222222222222222, "#7201a8"], [0.3333333333333333, "#9c179e"], [0.4444444444444444, "#bd3786"], [0.5555555555555556, "#d8576b"], [0.6666666666666666, "#ed7953"], [0.7777777777777778, "#fb9f3a"], [0.8888888888888888, "#fdca26"], [1.0, "#f0f921"]], "type": "contour"}], "contourcarpet": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "type": "contourcarpet"}], "heatmap": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "colorscale": [[0.0, "#0d0887"], [0.1111111111111111, "#46039f"], [0.2222222222222222, "#7201a8"], [0.3333333333333333, "#9c179e"], [0.4444444444444444, "#bd3786"], [0.5555555555555556, "#d8576b"], [0.6666666666666666, "#ed7953"], [0.7777777777777778, "#fb9f3a"], [0.8888888888888888, "#fdca26"], [1.0, "#f0f921"]], "type": "heatmap"}], "heatmapgl": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "colorscale": [[0.0, "#0d0887"], [0.1111111111111111, "#46039f"], [0.2222222222222222, "#7201a8"], [0.3333333333333333, "#9c179e"], [0.4444444444444444, "#bd3786"], [0.5555555555555556, "#d8576b"], [0.6666666666666666, "#ed7953"], [0.7777777777777778, "#fb9f3a"], [0.8888888888888888, "#fdca26"], [1.0, "#f0f921"]], "type": "heatmapgl"}], "histogram": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "histogram"}], "histogram2d": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "colorscale": [[0.0, "#0d0887"], [0.1111111111111111, "#46039f"], [0.2222222222222222, "#7201a8"], [0.3333333333333333, "#9c179e"], [0.4444444444444444, "#bd3786"], [0.5555555555555556, "#d8576b"], [0.6666666666666666, "#ed7953"], [0.7777777777777778, "#fb9f3a"], [0.8888888888888888, "#fdca26"], [1.0, "#f0f921"]], "type": "histogram2d"}], "histogram2dcontour": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "colorscale": [[0.0, "#0d0887"], [0.1111111111111111, "#46039f"], [0.2222222222222222, "#7201a8"], [0.3333333333333333, "#9c179e"], [0.4444444444444444, "#bd3786"], [0.5555555555555556, "#d8576b"], [0.6666666666666666, "#ed7953"], [0.7777777777777778, "#fb9f3a"], [0.8888888888888888, "#fdca26"], [1.0, "#f0f921"]], "type": "histogram2dcontour"}], "mesh3d": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "type": "mesh3d"}], "parcoords": [{"line": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "parcoords"}], "pie": [{"automargin": true, "type": "pie"}], "scatter": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scatter"}], "scatter3d": [{"line": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scatter3d"}], "scattercarpet": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scattercarpet"}], "scattergeo": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scattergeo"}], "scattergl": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scattergl"}], "scattermapbox": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scattermapbox"}], "scatterpolar": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scatterpolar"}], "scatterpolargl": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scatterpolargl"}], "scatterternary": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scatterternary"}], "surface": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "colorscale": [[0.0, "#0d0887"], [0.1111111111111111, "#46039f"], [0.2222222222222222, "#7201a8"], [0.3333333333333333, "#9c179e"], [0.4444444444444444, "#bd3786"], [0.5555555555555556, "#d8576b"], [0.6666666666666666, "#ed7953"], [0.7777777777777778, "#fb9f3a"], [0.8888888888888888, "#fdca26"], [1.0, "#f0f921"]], "type": "surface"}], "table": [{"cells": {"fill": {"color": "#EBF0F8"}, "line": {"color": "white"}}, "header": {"fill": {"color": "#C8D4E3"}, "line": {"color": "white"}}, "type": "table"}]}, "layout": {"annotationdefaults": {"arrowcolor": "#2a3f5f", "arrowhead": 0, "arrowwidth": 1}, "coloraxis": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "colorscale": {"diverging": [[0, "#8e0152"], [0.1, "#c51b7d"], [0.2, "#de77ae"], [0.3, "#f1b6da"], [0.4, "#fde0ef"], [0.5, "#f7f7f7"], [0.6, "#e6f5d0"], [0.7, "#b8e186"], [0.8, "#7fbc41"], [0.9, "#4d9221"], [1, "#276419"]], "sequential": [[0.0, "#0d0887"], [0.1111111111111111, "#46039f"], [0.2222222222222222, "#7201a8"], [0.3333333333333333, "#9c179e"], [0.4444444444444444, "#bd3786"], [0.5555555555555556, "#d8576b"], [0.6666666666666666, "#ed7953"], [0.7777777777777778, "#fb9f3a"], [0.8888888888888888, "#fdca26"], [1.0, "#f0f921"]], "sequentialminus": [[0.0, "#0d0887"], [0.1111111111111111, "#46039f"], [0.2222222222222222, "#7201a8"], [0.3333333333333333, "#9c179e"], [0.4444444444444444, "#bd3786"], [0.5555555555555556, "#d8576b"], [0.6666666666666666, "#ed7953"], [0.7777777777777778, "#fb9f3a"], [0.8888888888888888, "#fdca26"], [1.0, "#f0f921"]]}, "colorway": ["#636efa", "#EF553B", "#00cc96", "#ab63fa", "#FFA15A", "#19d3f3", "#FF6692", "#B6E880", "#FF97FF", "#FECB52"], "font": {"color": "#2a3f5f"}, "geo": {"bgcolor": "white", "lakecolor": "white", "landcolor": "#E5ECF6", "showlakes": true, "showland": true, "subunitcolor": "white"}, "hoverlabel": {"align": "left"}, "hovermode": "closest", "mapbox": {"style": "light"}, "paper_bgcolor": "white", "plot_bgcolor": "#E5ECF6", "polar": {"angularaxis": {"gridcolor": "white", "linecolor": "white", "ticks": ""}, "bgcolor": "#E5ECF6", "radialaxis": {"gridcolor": "white", "linecolor": "white", "ticks": ""}}, "scene": {"xaxis": {"backgroundcolor": "#E5ECF6", "gridcolor": "white", "gridwidth": 2, "linecolor": "white", "showbackground": true, "ticks": "", "zerolinecolor": "white"}, "yaxis": {"backgroundcolor": "#E5ECF6", "gridcolor": "white", "gridwidth": 2, "linecolor": "white", "showbackground": true, "ticks": "", "zerolinecolor": "white"}, "zaxis": {"backgroundcolor": "#E5ECF6", "gridcolor": "white", "gridwidth": 2, "linecolor": "white", "showbackground": true, "ticks": "", "zerolinecolor": "white"}}, "shapedefaults": {"line": {"color": "#2a3f5f"}}, "ternary": {"aaxis": {"gridcolor": "white", "linecolor": "white", "ticks": ""}, "baxis": {"gridcolor": "white", "linecolor": "white", "ticks": ""}, "bgcolor": "#E5ECF6", "caxis": {"gridcolor": "white", "linecolor": "white", "ticks": ""}}, "title": {"x": 0.05}, "xaxis": {"automargin": true, "gridcolor": "white", "linecolor": "white", "ticks": "", "title": {"standoff": 15}, "zerolinecolor": "white", "zerolinewidth": 2}, "yaxis": {"automargin": true, "gridcolor": "white", "linecolor": "white", "ticks": "", "title": {"standoff": 15}, "zerolinecolor": "white", "zerolinewidth": 2}}}, "xaxis": {"anchor": "y", "domain": [0.0, 1.0], "title": {"text": "x"}}, "yaxis": {"anchor": "x", "domain": [0.0, 1.0], "title": {"text": "y"}}},
                        {"responsive": true}
                    ).then(function(){

var gd = document.getElementById('ee28e4a0-9d14-4a8e-ac67-536bcd674fc0');
var x = new MutationObserver(function (mutations, observer) {{
        var display = window.getComputedStyle(gd).display;
        if (!display || display === 'none') {{
            console.log([gd, 'removed!']);
            Plotly.purge(gd);
            observer.disconnect();
        }}
}});

// Listen for the removal of the full notebook cells
var notebookContainer = gd.closest('#notebook-container');
if (notebookContainer) {{
    x.observe(notebookContainer, {childList: true});
}}

// Listen for the clearing of the current output cell
var outputEl = gd.closest('.output');
if (outputEl) {{
    x.observe(outputEl, {childList: true});
}}

                        })
                };
                });
            </script>
        </div>



```python
fig = px.scatter(x=testing_data[:,2],y=testing_data[:,3],color = test_prediction)
fig.show()
```


<div>


            <div id="f3d95fb1-683e-491b-89ac-a9679dab4c06" class="plotly-graph-div" style="height:525px; width:100%;"></div>
            <script type="text/javascript">
                require(["plotly"], function(Plotly) {
                    window.PLOTLYENV=window.PLOTLYENV || {};

                if (document.getElementById("f3d95fb1-683e-491b-89ac-a9679dab4c06")) {
                    Plotly.newPlot(
                        'f3d95fb1-683e-491b-89ac-a9679dab4c06',
                        [{"hoverlabel": {"namelength": 0}, "hovertemplate": "color=Iris-setosa<br>x=%{x}<br>y=%{y}", "legendgroup": "color=Iris-setosa", "marker": {"color": "#636efa", "symbol": "circle"}, "mode": "markers", "name": "color=Iris-setosa", "showlegend": true, "type": "scatter", "x": [1.4, 1.4, 1.0, 1.7, 1.5, 1.3, 1.4, 1.5, 1.5, 1.5, 1.3, 1.5, 1.3], "xaxis": "x", "y": [0.1, 0.2, 0.2, 0.3, 0.1, 0.3, 0.3, 0.3, 0.2, 0.4, 0.3, 0.1, 0.2], "yaxis": "y"}, {"hoverlabel": {"namelength": 0}, "hovertemplate": "color=Iris-versicolor<br>x=%{x}<br>y=%{y}", "legendgroup": "color=Iris-versicolor", "marker": {"color": "#EF553B", "symbol": "circle"}, "mode": "markers", "name": "color=Iris-versicolor", "showlegend": true, "type": "scatter", "x": [3.3, 4.8, 4.1, 4.2, 4.3, 4.2, 4.0, 4.4], "xaxis": "x", "y": [1.0, 1.4, 1.3, 1.5, 1.3, 1.3, 1.2, 1.4], "yaxis": "y"}, {"hoverlabel": {"namelength": 0}, "hovertemplate": "color=Iris-virginica<br>x=%{x}<br>y=%{y}", "legendgroup": "color=Iris-virginica", "marker": {"color": "#00cc96", "symbol": "circle"}, "mode": "markers", "name": "color=Iris-virginica", "showlegend": true, "type": "scatter", "x": [5.2, 5.7, 6.6, 5.1, 4.8, 6.1, 6.3, 6.7, 5.3], "xaxis": "x", "y": [2.3, 2.5, 2.1, 1.9, 1.8, 1.9, 1.8, 2.0, 1.9], "yaxis": "y"}],
                        {"legend": {"tracegroupgap": 0}, "margin": {"t": 60}, "template": {"data": {"bar": [{"error_x": {"color": "#2a3f5f"}, "error_y": {"color": "#2a3f5f"}, "marker": {"line": {"color": "#E5ECF6", "width": 0.5}}, "type": "bar"}], "barpolar": [{"marker": {"line": {"color": "#E5ECF6", "width": 0.5}}, "type": "barpolar"}], "carpet": [{"aaxis": {"endlinecolor": "#2a3f5f", "gridcolor": "white", "linecolor": "white", "minorgridcolor": "white", "startlinecolor": "#2a3f5f"}, "baxis": {"endlinecolor": "#2a3f5f", "gridcolor": "white", "linecolor": "white", "minorgridcolor": "white", "startlinecolor": "#2a3f5f"}, "type": "carpet"}], "choropleth": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "type": "choropleth"}], "contour": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "colorscale": [[0.0, "#0d0887"], [0.1111111111111111, "#46039f"], [0.2222222222222222, "#7201a8"], [0.3333333333333333, "#9c179e"], [0.4444444444444444, "#bd3786"], [0.5555555555555556, "#d8576b"], [0.6666666666666666, "#ed7953"], [0.7777777777777778, "#fb9f3a"], [0.8888888888888888, "#fdca26"], [1.0, "#f0f921"]], "type": "contour"}], "contourcarpet": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "type": "contourcarpet"}], "heatmap": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "colorscale": [[0.0, "#0d0887"], [0.1111111111111111, "#46039f"], [0.2222222222222222, "#7201a8"], [0.3333333333333333, "#9c179e"], [0.4444444444444444, "#bd3786"], [0.5555555555555556, "#d8576b"], [0.6666666666666666, "#ed7953"], [0.7777777777777778, "#fb9f3a"], [0.8888888888888888, "#fdca26"], [1.0, "#f0f921"]], "type": "heatmap"}], "heatmapgl": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "colorscale": [[0.0, "#0d0887"], [0.1111111111111111, "#46039f"], [0.2222222222222222, "#7201a8"], [0.3333333333333333, "#9c179e"], [0.4444444444444444, "#bd3786"], [0.5555555555555556, "#d8576b"], [0.6666666666666666, "#ed7953"], [0.7777777777777778, "#fb9f3a"], [0.8888888888888888, "#fdca26"], [1.0, "#f0f921"]], "type": "heatmapgl"}], "histogram": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "histogram"}], "histogram2d": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "colorscale": [[0.0, "#0d0887"], [0.1111111111111111, "#46039f"], [0.2222222222222222, "#7201a8"], [0.3333333333333333, "#9c179e"], [0.4444444444444444, "#bd3786"], [0.5555555555555556, "#d8576b"], [0.6666666666666666, "#ed7953"], [0.7777777777777778, "#fb9f3a"], [0.8888888888888888, "#fdca26"], [1.0, "#f0f921"]], "type": "histogram2d"}], "histogram2dcontour": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "colorscale": [[0.0, "#0d0887"], [0.1111111111111111, "#46039f"], [0.2222222222222222, "#7201a8"], [0.3333333333333333, "#9c179e"], [0.4444444444444444, "#bd3786"], [0.5555555555555556, "#d8576b"], [0.6666666666666666, "#ed7953"], [0.7777777777777778, "#fb9f3a"], [0.8888888888888888, "#fdca26"], [1.0, "#f0f921"]], "type": "histogram2dcontour"}], "mesh3d": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "type": "mesh3d"}], "parcoords": [{"line": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "parcoords"}], "pie": [{"automargin": true, "type": "pie"}], "scatter": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scatter"}], "scatter3d": [{"line": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scatter3d"}], "scattercarpet": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scattercarpet"}], "scattergeo": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scattergeo"}], "scattergl": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scattergl"}], "scattermapbox": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scattermapbox"}], "scatterpolar": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scatterpolar"}], "scatterpolargl": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scatterpolargl"}], "scatterternary": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scatterternary"}], "surface": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "colorscale": [[0.0, "#0d0887"], [0.1111111111111111, "#46039f"], [0.2222222222222222, "#7201a8"], [0.3333333333333333, "#9c179e"], [0.4444444444444444, "#bd3786"], [0.5555555555555556, "#d8576b"], [0.6666666666666666, "#ed7953"], [0.7777777777777778, "#fb9f3a"], [0.8888888888888888, "#fdca26"], [1.0, "#f0f921"]], "type": "surface"}], "table": [{"cells": {"fill": {"color": "#EBF0F8"}, "line": {"color": "white"}}, "header": {"fill": {"color": "#C8D4E3"}, "line": {"color": "white"}}, "type": "table"}]}, "layout": {"annotationdefaults": {"arrowcolor": "#2a3f5f", "arrowhead": 0, "arrowwidth": 1}, "coloraxis": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "colorscale": {"diverging": [[0, "#8e0152"], [0.1, "#c51b7d"], [0.2, "#de77ae"], [0.3, "#f1b6da"], [0.4, "#fde0ef"], [0.5, "#f7f7f7"], [0.6, "#e6f5d0"], [0.7, "#b8e186"], [0.8, "#7fbc41"], [0.9, "#4d9221"], [1, "#276419"]], "sequential": [[0.0, "#0d0887"], [0.1111111111111111, "#46039f"], [0.2222222222222222, "#7201a8"], [0.3333333333333333, "#9c179e"], [0.4444444444444444, "#bd3786"], [0.5555555555555556, "#d8576b"], [0.6666666666666666, "#ed7953"], [0.7777777777777778, "#fb9f3a"], [0.8888888888888888, "#fdca26"], [1.0, "#f0f921"]], "sequentialminus": [[0.0, "#0d0887"], [0.1111111111111111, "#46039f"], [0.2222222222222222, "#7201a8"], [0.3333333333333333, "#9c179e"], [0.4444444444444444, "#bd3786"], [0.5555555555555556, "#d8576b"], [0.6666666666666666, "#ed7953"], [0.7777777777777778, "#fb9f3a"], [0.8888888888888888, "#fdca26"], [1.0, "#f0f921"]]}, "colorway": ["#636efa", "#EF553B", "#00cc96", "#ab63fa", "#FFA15A", "#19d3f3", "#FF6692", "#B6E880", "#FF97FF", "#FECB52"], "font": {"color": "#2a3f5f"}, "geo": {"bgcolor": "white", "lakecolor": "white", "landcolor": "#E5ECF6", "showlakes": true, "showland": true, "subunitcolor": "white"}, "hoverlabel": {"align": "left"}, "hovermode": "closest", "mapbox": {"style": "light"}, "paper_bgcolor": "white", "plot_bgcolor": "#E5ECF6", "polar": {"angularaxis": {"gridcolor": "white", "linecolor": "white", "ticks": ""}, "bgcolor": "#E5ECF6", "radialaxis": {"gridcolor": "white", "linecolor": "white", "ticks": ""}}, "scene": {"xaxis": {"backgroundcolor": "#E5ECF6", "gridcolor": "white", "gridwidth": 2, "linecolor": "white", "showbackground": true, "ticks": "", "zerolinecolor": "white"}, "yaxis": {"backgroundcolor": "#E5ECF6", "gridcolor": "white", "gridwidth": 2, "linecolor": "white", "showbackground": true, "ticks": "", "zerolinecolor": "white"}, "zaxis": {"backgroundcolor": "#E5ECF6", "gridcolor": "white", "gridwidth": 2, "linecolor": "white", "showbackground": true, "ticks": "", "zerolinecolor": "white"}}, "shapedefaults": {"line": {"color": "#2a3f5f"}}, "ternary": {"aaxis": {"gridcolor": "white", "linecolor": "white", "ticks": ""}, "baxis": {"gridcolor": "white", "linecolor": "white", "ticks": ""}, "bgcolor": "#E5ECF6", "caxis": {"gridcolor": "white", "linecolor": "white", "ticks": ""}}, "title": {"x": 0.05}, "xaxis": {"automargin": true, "gridcolor": "white", "linecolor": "white", "ticks": "", "title": {"standoff": 15}, "zerolinecolor": "white", "zerolinewidth": 2}, "yaxis": {"automargin": true, "gridcolor": "white", "linecolor": "white", "ticks": "", "title": {"standoff": 15}, "zerolinecolor": "white", "zerolinewidth": 2}}}, "xaxis": {"anchor": "y", "domain": [0.0, 1.0], "title": {"text": "x"}}, "yaxis": {"anchor": "x", "domain": [0.0, 1.0], "title": {"text": "y"}}},
                        {"responsive": true}
                    ).then(function(){

var gd = document.getElementById('f3d95fb1-683e-491b-89ac-a9679dab4c06');
var x = new MutationObserver(function (mutations, observer) {{
        var display = window.getComputedStyle(gd).display;
        if (!display || display === 'none') {{
            console.log([gd, 'removed!']);
            Plotly.purge(gd);
            observer.disconnect();
        }}
}});

// Listen for the removal of the full notebook cells
var notebookContainer = gd.closest('#notebook-container');
if (notebookContainer) {{
    x.observe(notebookContainer, {childList: true});
}}

// Listen for the clearing of the current output cell
var outputEl = gd.closest('.output');
if (outputEl) {{
    x.observe(outputEl, {childList: true});
}}

                        })
                };
                });
            </script>
        </div>



```python

```


```python

```


```python

```
