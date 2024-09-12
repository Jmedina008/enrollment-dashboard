import pandas as pd
import plotly.express as px
import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import sys
import os
import traceback

# Add the current directory to Python's path
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

# Now try importing
from ml_applications import forecast_enrollment, cluster_demographics, detect_anomalies  # Import ML functions

# Load the dataset
file_path = r'C:\Users\j2medina\Documents\New Jersey Pre-K Stats.csv'
data = pd.read_csv(file_path)

# After loading the data
data['Year'] = data['Year'].astype(str)  # Ensure 'Year' is treated as a string

# Create a new column combining Demographic and Gender
data['Demographic_Gender'] = data['Demographic'] + ' (' + data['Gender'] + ')'

# Print data information
print(data.head())
print(data.columns)
print(data['Year'].dtype, data['State'].dtype, data['Gender'].dtype, data['Enrollment'].dtype)

# Debug prints
print("Data shape:", data.shape)
print("Data columns:", data.columns)
print("Data types:", data.dtypes)
print("Sample data:")
print(data.head())

# Check for null values
print("Null values:")
print(data.isnull().sum())

# Check unique values in categorical columns
for col in ['Year', 'State', 'Gender', 'Demographic']:
    print(f"Unique values in {col}:", data[col].nunique())

# Dash app creation
app = dash.Dash(__name__)

app.layout = html.Div([
    html.H1('Enrollment Dashboard'),
    
    # Existing dropdowns plus new Demographic dropdown
    html.Div([
        dcc.Dropdown(id='year-dropdown', options=[{'label': str(year), 'value': year} for year in sorted(data['Year'].unique())], multi=True, placeholder="Select Year(s)"),
        dcc.Dropdown(id='state-dropdown', options=[{'label': state, 'value': state} for state in sorted(data['State'].unique())], placeholder="Select State"),
        dcc.Dropdown(id='gender-dropdown', options=[{'label': gender, 'value': gender} for gender in sorted(data['Gender'].unique())], multi=True, placeholder="Select Gender(s)"),
        dcc.Dropdown(id='demographic-dropdown', options=[{'label': demo, 'value': demo} for demo in sorted(data['Demographic'].unique())], multi=True, placeholder="Select Demographic(s)")
    ]),
    
    # Main enrollment graph
    html.H2('Enrollment Over Time'),
    dcc.Graph(id='enrollment-graph'),
    dcc.Markdown(id='enrollment-graph-analysis'),
    
    # Bar chart
    html.H2('Enrollment by Demographic'),
    dcc.Graph(id='enrollment-bar-chart'),
    dcc.Markdown(id='enrollment-bar-chart-analysis'),
    
    # Pie chart
    html.H2('Enrollment Distribution'),
    dcc.Graph(id='enrollment-pie-chart'),
    dcc.Markdown(id='enrollment-pie-chart-analysis'),
    
    # Line trend chart
    html.H2('Enrollment Trends'),
    dcc.Graph(id='enrollment-trend-chart'),
    dcc.Markdown(id='enrollment-trend-chart-analysis'),
    
    # ML section
    html.H2('Machine Learning Analysis'),
    dcc.Dropdown(
        id='ml-dropdown',
        options=[
            {'label': 'Time Series Forecasting', 'value': 'forecast'},
            {'label': 'Clustering', 'value': 'clustering'},
            {'label': 'Anomaly Detection', 'value': 'anomaly'}
        ],
        value='forecast',
        style={'width': '50%'}
    ),
    dcc.Graph(id='ml-output-graph'),
    dcc.Markdown(id='ml-output', style={'whiteSpace': 'pre-wrap'}),
])

def log_error(message):
    print(f"ERROR: {message}", file=sys.stderr)
    print(traceback.format_exc(), file=sys.stderr)

# Existing callback for main enrollment graph
@app.callback(
    [Output('enrollment-graph', 'figure'),
     Output('enrollment-graph-analysis', 'children')],
    [Input('year-dropdown', 'value'),
     Input('state-dropdown', 'value'),
     Input('gender-dropdown', 'value'),
     Input('demographic-dropdown', 'value')]
)
def update_graph(selected_years, selected_state, selected_genders, selected_demographics):
    try:
        if not selected_years or not selected_state or not selected_genders:
            return {}, ""
        
        filtered_data = data[(data['Year'].isin([str(year) for year in selected_years])) & 
                             (data['State'] == selected_state) & 
                             (data['Gender'].isin(selected_genders))]
        
        if selected_demographics:
            filtered_data = filtered_data[filtered_data['Demographic'].isin(selected_demographics)]
        
        if filtered_data.empty:
            return {}, "No data available for the selected criteria."
        
        fig = px.line(filtered_data, x='Year', y='Enrollment', color='Demographic_Gender',
                      title=f'Line Chart: Enrollment in {selected_state}')
        fig.update_xaxes(type='category')
        
        # Generate analysis
        total_enrollment = filtered_data.groupby('Year')['Enrollment'].sum()
        start_enrollment = total_enrollment.iloc[0]
        end_enrollment = total_enrollment.iloc[-1]
        percent_change = ((end_enrollment - start_enrollment) / start_enrollment) * 100
        
        demographic_trends = filtered_data.groupby('Demographic')['Enrollment'].agg(['first', 'last'])
        demographic_trends['percent_change'] = (demographic_trends['last'] - demographic_trends['first']) / demographic_trends['first'] * 100
        fastest_growing = demographic_trends['percent_change'].idxmax()
        fastest_declining = demographic_trends['percent_change'].idxmin()
        
        analysis = f"""
        Enrollment Analysis for {selected_state} from {selected_years[0]} to {selected_years[-1]}:

        Overall, the total enrollment in {selected_state} has {'increased' if percent_change > 0 else 'decreased'} by {abs(percent_change):.2f}% over the selected period. The enrollment started at {start_enrollment:.0f} and ended at {end_enrollment:.0f}.

        Looking at specific demographic trends:
        - The fastest growing demographic was "{fastest_growing}", with a {demographic_trends.loc[fastest_growing, 'percent_change']:.2f}% increase.
        - The fastest declining demographic was "{fastest_declining}", with a {abs(demographic_trends.loc[fastest_declining, 'percent_change']):.2f}% decrease.

        The graph shows enrollment trends for different demographic groups and genders over time. You can observe how these trends differ across various demographics, which may reflect changes in population composition, educational policies, or other socio-economic factors specific to {selected_state}.
        """
        
        return fig, analysis
    except Exception as e:
        log_error(f"Error in update_graph: {str(e)}")
        return {}, f"An error occurred: {str(e)}"

# New callback for bar chart
@app.callback(
    [Output('enrollment-bar-chart', 'figure'),
     Output('enrollment-bar-chart-analysis', 'children')],
    [Input('year-dropdown', 'value'),
     Input('state-dropdown', 'value'),
     Input('gender-dropdown', 'value'),
     Input('demographic-dropdown', 'value')]
)
def update_bar_chart(selected_years, selected_state, selected_genders, selected_demographics):
    try:
        if not selected_years or not selected_state or not selected_genders:
            return {}, ""
        
        filtered_data = data[(data['Year'].isin([str(year) for year in selected_years])) & 
                             (data['State'] == selected_state) & 
                             (data['Gender'].isin(selected_genders))]
        
        if selected_demographics:
            filtered_data = filtered_data[filtered_data['Demographic'].isin(selected_demographics)]
        
        if filtered_data.empty:
            return {}, "No data available for the selected criteria."
        
        fig = px.bar(filtered_data, x='Demographic', y='Enrollment', color='Gender',
                     title=f'Bar Chart: Enrollment by Demographic in {selected_state}')
        
        # Generate analysis
        total_enrollment = filtered_data['Enrollment'].sum()
        demographic_enrollment = filtered_data.groupby('Demographic')['Enrollment'].sum().sort_values(ascending=False)
        top_demographic = demographic_enrollment.index[0]
        bottom_demographic = demographic_enrollment.index[-1]
        gender_enrollment = filtered_data.groupby('Gender')['Enrollment'].sum()
        
        gender_analysis = ""
        if 'Male' in gender_enrollment.index and 'Female' in gender_enrollment.index:
            gender_ratio = gender_enrollment['Male'] / gender_enrollment['Female']
            gender_analysis = f"In terms of gender distribution, {'male students outnumber female students' if gender_ratio > 1 else 'female students outnumber male students'} with a ratio of {gender_ratio:.2f} males to females."
        elif 'Male' in gender_enrollment.index:
            gender_analysis = "Only male students are present in the selected data."
        elif 'Female' in gender_enrollment.index:
            gender_analysis = "Only female students are present in the selected data."
        else:
            gender_analysis = "No gender-specific data is available for the selected criteria."
        
        analysis = f"""
        Demographic Enrollment Analysis for {selected_state} ({', '.join(selected_years)}):

        The total enrollment across all selected demographics in {selected_state} is {total_enrollment:.0f}. This bar chart provides a breakdown of enrollment by demographic and gender, offering insights into the composition of the student population.

        Key observations:
        - The demographic with the highest enrollment is "{top_demographic}", accounting for {(demographic_enrollment[top_demographic] / total_enrollment * 100):.2f}% of total enrollment.
        - The demographic with the lowest enrollment is "{bottom_demographic}", representing {(demographic_enrollment[bottom_demographic] / total_enrollment * 100):.2f}% of total enrollment.
        - {gender_analysis}

        These enrollment patterns may reflect demographic composition, cultural factors, or educational policies specific to {selected_state}. Consider investigating any significant disparities between demographics or genders to ensure equal educational opportunities.
        """
        
        return fig, analysis
    except Exception as e:
        log_error(f"Error in update_bar_chart: {str(e)}")
        return {}, f"An error occurred: {str(e)}"

# New callback for pie chart
@app.callback(
    [Output('enrollment-pie-chart', 'figure'),
     Output('enrollment-pie-chart-analysis', 'children')],
    [Input('year-dropdown', 'value'),
     Input('state-dropdown', 'value'),
     Input('gender-dropdown', 'value'),
     Input('demographic-dropdown', 'value')]
)
def update_pie_chart(selected_years, selected_state, selected_genders, selected_demographics):
    try:
        if not selected_years or not selected_state or not selected_genders:
            return {}, ""
        
        filtered_data = data[(data['Year'].isin([str(year) for year in selected_years])) & 
                             (data['State'] == selected_state) & 
                             (data['Gender'].isin(selected_genders))]
        
        if selected_demographics:
            filtered_data = filtered_data[filtered_data['Demographic'].isin(selected_demographics)]
        
        fig = px.pie(filtered_data, values='Enrollment', names='Demographic',
                     title=f'Pie Chart: Enrollment Distribution in {selected_state}')
        
        # Generate analysis
        total_enrollment = filtered_data['Enrollment'].sum()
        demographic_percentages = (filtered_data.groupby('Demographic')['Enrollment'].sum() / total_enrollment * 100).sort_values(ascending=False)
        
        analysis = f"""
        Enrollment Distribution Analysis for {selected_state} ({', '.join(selected_years)}):

        The pie chart visualizes the proportion of enrollment for each demographic group in {selected_state}, with a total enrollment of {total_enrollment:.0f} students across the selected demographics and time period.

        Key insights:
        1. The largest demographic group is "{demographic_percentages.index[0]}", representing {demographic_percentages.iloc[0]:.2f}% of the total enrollment.
        2. The second-largest group is "{demographic_percentages.index[1]}", accounting for {demographic_percentages.iloc[1]:.2f}% of enrollments.
        3. The smallest demographic group is "{demographic_percentages.index[-1]}", making up {demographic_percentages.iloc[-1]:.2f}% of the student population.

        This distribution provides valuable insights into the diversity of the student population in {selected_state}. It's important to consider whether this distribution aligns with the overall population demographics of the state. Any significant discrepancies might indicate areas where targeted educational outreach or support programs could be beneficial to ensure equal access to education across all demographic groups.
        """
        
        return fig, analysis
    except Exception as e:
        log_error(f"Error in update_pie_chart: {str(e)}")
        return {}, f"An error occurred: {str(e)}"

# New callback for line trend chart
@app.callback(
    [Output('enrollment-trend-chart', 'figure'),
     Output('enrollment-trend-chart-analysis', 'children')],
    [Input('state-dropdown', 'value'),
     Input('gender-dropdown', 'value'),
     Input('demographic-dropdown', 'value')]
)
def update_trend_chart(selected_state, selected_genders, selected_demographics):
    try:
        if not selected_state or not selected_genders:
            return {}, ""
        
        filtered_data = data[(data['State'] == selected_state) & 
                             (data['Gender'].isin(selected_genders))]
        
        if selected_demographics:
            filtered_data = filtered_data[filtered_data['Demographic'].isin(selected_demographics)]
        
        fig = px.line(filtered_data, x='Year', y='Enrollment', color='Demographic',
                      title=f'Line Chart: Enrollment Trends in {selected_state}')
        fig.update_xaxes(type='category')
        
        # Generate analysis
        start_year = filtered_data['Year'].min()
        end_year = filtered_data['Year'].max()
        total_change = filtered_data.groupby('Year')['Enrollment'].sum().pct_change().sum() * 100
        demographic_changes = filtered_data.groupby('Demographic').apply(lambda x: (x['Enrollment'].iloc[-1] - x['Enrollment'].iloc[0]) / x['Enrollment'].iloc[0] * 100)
        fastest_growing = demographic_changes.idxmax() if not demographic_changes.empty else "N/A"
        fastest_declining = demographic_changes.idxmin() if not demographic_changes.empty else "N/A"
        
        analysis = f"""
        Enrollment Trend Analysis for {selected_state} from {start_year} to {end_year}:

        Over the observed period, the overall enrollment in {selected_state} has {'increased' if total_change > 0 else 'decreased'} by approximately {abs(total_change):.2f}%. This trend reflects the changing educational landscape in the state, which may be influenced by factors such as population growth, migration patterns, or changes in educational policies.

        Key trends by demographic:
        1. The fastest growing demographic is "{fastest_growing}", with a {demographic_changes.get(fastest_growing, 0):.2f}% increase in enrollment over the period.
        2. The fastest declining demographic is "{fastest_declining}", showing a {abs(demographic_changes.get(fastest_declining, 0)):.2f}% decrease in enrollment.

        The line chart illustrates enrollment trends for each demographic group over time, allowing for comparison of growth rates and patterns. These trends can provide valuable insights for educational planning and resource allocation. For instance:
        - Rapidly growing demographics may require additional resources and infrastructure to accommodate increasing enrollment.
        - Declining enrollment in certain demographics might signal a need for targeted outreach or support programs.
        - Diverging trends between demographics could indicate changing population dynamics or potential disparities in educational access that may need to be addressed.

        It's important to consider these trends in the context of broader socio-economic factors in {selected_state} to fully understand the drivers behind these enrollment patterns.
        """
        
        return fig, analysis
    except Exception as e:
        log_error(f"Error in update_trend_chart: {str(e)}")
        return {}, f"An error occurred: {str(e)}"

# Add ML callback
@app.callback(
    [Output('ml-output', 'children'),
     Output('ml-output-graph', 'figure')],
    [Input('ml-dropdown', 'value'),
     Input('state-dropdown', 'value'),
     Input('gender-dropdown', 'value'),
     Input('demographic-dropdown', 'value')]
)
def update_ml_output(ml_task, selected_state, selected_genders, selected_demographics):
    try:
        print(f"Debug: update_ml_output called with {ml_task}, {selected_state}, {selected_genders}, {selected_demographics}")
        if not ml_task or not selected_state or not selected_genders:
            return "Please select all required inputs", {}

        if ml_task == 'forecast':
            print("Debug: Entering forecast task")
            # Filter data for the selected state, genders, and demographics
            filtered_data = data[(data['State'] == selected_state) & 
                                 (data['Gender'].isin(selected_genders))]
            
            if selected_demographics:
                filtered_data = filtered_data[filtered_data['Demographic'].isin(selected_demographics)]
            
            print(f"Debug: Filtered data shape: {filtered_data.shape}")
            
            # Aggregate enrollment across selected demographics
            aggregated_data = filtered_data.groupby('Year')['Enrollment'].sum().reset_index()
            
            print(f"Debug: Aggregated data shape: {aggregated_data.shape}")
            print(f"Debug: Aggregated data head: {aggregated_data.head()}")
            
            # Perform forecasting on aggregated data
            forecast_data = forecast_enrollment(aggregated_data, selected_state)
            
            print(f"Debug: Forecast data shape: {forecast_data.shape}")
            print(f"Debug: Forecast data head: {forecast_data.head()}")
            
            demographic_label = "All Demographics" if not selected_demographics else ", ".join(selected_demographics)
            fig = px.line(forecast_data, x='ds', y='yhat', 
                          title=f'Line Chart: Enrollment Forecast for {selected_state} ({demographic_label})')
            
            # Generate analysis
            last_actual = aggregated_data['Enrollment'].iloc[-1]
            last_forecast = forecast_data['yhat'].iloc[-1]
            percent_change = ((last_forecast - last_actual) / last_actual) * 100
            
            analysis = f"""
            Forecast Analysis for {selected_state} ({demographic_label}):

            The model predicts total enrollment will {'increase' if percent_change > 0 else 'decrease'} by approximately {abs(percent_change):.2f}% over the next 3 years.

            - The last actual total enrollment was {last_actual:.0f}.
            - The forecasted total enrollment for the final year is {last_forecast:.0f}.
            - This forecast is based on historical trends and assumes similar patterns will continue.
            - Note: This forecast aggregates data across the selected demographic groups and genders.
            """
            
            return analysis, fig
        elif ml_task == 'clustering':
            cluster_data = cluster_demographics(data)
            fig = px.scatter(cluster_data, x='Enrollment', y='Demographic_Gender', color='Cluster', title='Scatter Plot: Demographic Clustering')
            
            # Generate analysis
            cluster_counts = cluster_data['Cluster'].value_counts()
            largest_cluster = cluster_counts.index[0]
            smallest_cluster = cluster_counts.index[-1]
            
            analysis = f"""
            Clustering Analysis:

            The demographics have been grouped into {len(cluster_counts)} distinct clusters based on enrollment patterns.

            - The largest cluster (Cluster {largest_cluster}) contains {cluster_counts[largest_cluster]} demographics.
            - The smallest cluster (Cluster {smallest_cluster}) contains {cluster_counts[smallest_cluster]} demographics.
            - Demographics in the same cluster have similar enrollment characteristics.
            - This clustering can help identify groups of demographics that may benefit from similar educational strategies.
            """
            
            return analysis, fig
        elif ml_task == 'anomaly':
            anomaly_data = detect_anomalies(data)
            fig = px.scatter(anomaly_data, x='Year', y='Enrollment', color='Anomaly', title='Scatter Plot: Anomaly Detection')
            
            # Generate analysis
            anomaly_count = (anomaly_data['Anomaly'] == -1).sum()
            total_points = len(anomaly_data)
            anomaly_percentage = (anomaly_count / total_points) * 100
            
            analysis = f"""
            Anomaly Detection Analysis:

            The model has identified {anomaly_count} potential anomalies out of {total_points} total data points.
            This represents approximately {anomaly_percentage:.2f}% of the data.

            Anomalies are unusual enrollment patterns that deviate significantly from the norm.
            These could be due to various factors such as policy changes, demographic shifts, or data recording errors.
            Points marked as anomalies may warrant further investigation to understand the underlying causes.
            """
            
            return analysis, fig
        else:
            return "Select a ML task", {}
    except Exception as e:
        log_error(f"Error in update_ml_output: {str(e)}")
        return f"An error occurred: {str(e)}", {'data': [], 'layout': {'title': f'Error: {str(e)}'}}

# Run the app
if __name__ == '__main__':
    app.run_server(debug=True)
