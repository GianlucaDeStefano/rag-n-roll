import json
import os
from typing import Optional
from matplotlib import pyplot as plt
import pandas as pd
import matplotlib.colors as mcolors

# PRint legends ?
PRINT_LEGENDS = False

# Parameters to change to customize the plots
color_unoptimized = '#d62728'
style_unoptimized = '--'
marker_unoptimized = '^'

colors_optimized = ['#1f77b4', '#ff7f0e', '#2ca02c', '#9467bd', '#8c564b','#7f7f7f', '#bcbd22', '#17becf','#976700','#e377c2']
styles_optimized = '-'
marker_optimized = ''

# Parameters for the plot
y_min = 0
y_max = 1


def plot_parameter_performance(df:pd.DataFrame,output_path,color,style,markers, performance_metric = ['Ben_Baseline','Ben','Mal', 'Ben&mal', 'Hallucination'],y_min = 0, y_max = 1): 
    """
        This function plots the results on a benchmark for a given parameter.
        It creates a plot for every performance metric that is present in the dataframe stacking them horizontally.

        The X axis is the value of the parameter, the Y axis is the performance metric.
        Every Optimization is represented by a line in the plot with a different color and style.
    """
    
    fig, axes = plt.subplots(nrows=1, ncols=len(performance_metric),sharey="row",figsize=(5*len(performance_metric), 3))
    fig.tight_layout()

    legends = []

    for i, metric in enumerate(performance_metric):
        assert metric in df.columns, f'{metric} not in the dataframe'                
        # Add metric to the dataframe
        pivot_df = df.pivot(index='Value', columns='Optimization', values=metric)
        pivot_df = pivot_df.sort_values('Value')            
        
        # Plot the performance of the parameter
        pivot_df = pivot_df.sort_values('Value')
        for optimization in pivot_df.columns:
            
            # For the baseline, we plot only the unoptimized version
            if metric == 'Ben_Baseline':
                if optimization.lower() != 'unoptimized':
                    continue
                
            
            label = optimization
            if optimization in legends:
                label = ""
            else:
                legends.append(optimization)
            
            if len(performance_metric) == 1:
                axes.plot(pivot_df.index, pivot_df[optimization], color=color[optimization],marker=markers[optimization], linestyle=style[optimization], label=label)
                axes.grid(True,axis="y")
                axes.margins(x=0.15)
            else:
                axes[i].plot(pivot_df.index, pivot_df[optimization], color=color[optimization],marker=markers[optimization], linestyle=style[optimization], label=label)
                axes[i].grid(True,axis='y')
                axes[i].margins(x=0.15)
    

    if PRINT_LEGENDS:
        fig.legend(loc='lower center', bbox_to_anchor=(0.5, 0), ncol=5)
    
    fig.savefig(output_path, bbox_inches='tight')


def plot_parameters_performance(df, output_folder,performance_metric = ['Ben','Mal','Ben&mal','Hallucination']):
    
    os.makedirs(output_folder, exist_ok=True)
    # Select y_min and y_max based on the performance metric
    global y_min, y_max
    for stat_name in performance_metric:
        y_min = min(y_min,df[stat_name].min())
        y_max = max(y_max,df[stat_name].max())
    y_max = min(1,y_max+0.015)
    
    # Create dict 
    optimizations = df['Optimization'].unique() 
    
    style_dic = {}

    df[['Color','LineStyle','Marker']] = pd.DataFrame([[color_unoptimized,style_unoptimized,marker_unoptimized]]*len(df))
    colors = {'unoptimized':color_unoptimized}
    linestyle = {'unoptimized':style_unoptimized}
    marker = {'unoptimized':marker_unoptimized}
    
    for i,optimization in enumerate(optimizations):
        if optimization == 'unoptimized':
            continue
        colors[optimization] = colors_optimized[i]
        linestyle[optimization] = '-'
        marker[optimization] = marker_optimized
        
        style_dic[optimization] = {'color':colors_optimized[i],'style':styles_optimized,'marker':marker_optimized}
    
    json.dump(style_dic,open(f'{output_folder}/styles.json','w',encoding='utf-8'))
    
    # For each parameter, plot the performance in a separate plot
    for parameter in df['Parameter'].unique():
        
        df_injection_parameter = df[df['Parameter'] == parameter]
        
        # If all the values are numeric, cast them to float and sort the dataframe
        are_numeric = [str(v).isnumeric() for v in df_injection_parameter['Value'].tolist()]
        if all(are_numeric):
            df_injection_parameter['Value'] = df_injection_parameter['Value'].astype(float)
        
        plot_parameter_performance(df_injection_parameter,f'{output_folder}/{parameter}.jpeg',colors,linestyle,marker,performance_metric)


def plot_parameters_by_injectionstr(df:pd.DataFrame,output_path, performance_metric = ['Ben_Baseline','Ben','Mal', 'Ben&mal', 'Hallucination']): 
    """
        This function plots the results on a benchmark for a given parameter.
        For each parameter, it calls plot_parameter_performance
    """
    os.makedirs(output_path, exist_ok=True)

    df['Value'] = df.apply(lambda x: x['Value'].split('-')[0],axis=1)
    
    # Get a list of all the injection strategies to plot individually
    injection_strategies = [i for i in df['Injection strategy'].unique() if i]
    
    # If no injection strategy is present, add None to the list
    if len(injection_strategies) == 0:
        injection_strategies = [None]
    
    # For each injection strategy, plot the performance of the parameter
    for injection_strategy in injection_strategies:
        
        injection_strategy_plt_folder = f'{output_path}/{injection_strategy}'
        if not os.path.exists(injection_strategy_plt_folder):
            os.makedirs(injection_strategy_plt_folder)        

        # Filter the dataframe to get only the rows related to the injection strategy (or for optimization having no injection strategy at all)
        df_injection_strategy = df[(df['Injection strategy'] == injection_strategy) | (df['Injection strategy'].isnull()) | (df['Injection strategy'] == 'nan')]
        
        plot_parameters_performance(df_injection_strategy, injection_strategy_plt_folder, performance_metric)
        
def plot_parameters_performance_grouping_optimizations(df:pd.DataFrame,output_path, families):
    """
    this function plots benchmark results for each parameter, grouping the optimizations in families.

    Args:
        df (pd.DataFrame): The dataframe containing the results
        output_path (str): The path where the plots will be saved
        families (dict)): dict containing the families of optimizations and their members
    """
    os.makedirs(output_path, exist_ok=True)
    assert families is not None, 'families must be specified'
    
    df['Optimization Family'] = 'unoptimized'
    for family, optimizations in families.items():
        df.loc[df['Optimization'].isin(optimizations),'Optimization Family'] = family

    # Step 1: Group by the specified columns and aggregate
    df = df.groupby(['Parameter', 'Value', 'Optimization Family','Baseline'])[ ['Ben', 'Mal', 'Ben&mal', 'Hallucination', 'Avg # mal docs in prompt']].agg({                                                                                                                                                                                            'Ben':['min','mean','max'],
        'Mal':['min','mean','max'],
        'Ben&mal':['min','mean','max'],
        'Hallucination':['min','mean','max']
        })

    # Step 2: Reshape the DataFrame to introduce the 'aggregation' column
    df = df.stack(level=-1).reset_index()

    # Step 3: Rename the stacked level to 'aggregation'
    df.rename(columns={'level_4': 'aggregation'}, inplace=True)
    
    # Drops records where aggregation is 'min' or 'max' and the baseline is 'unoptimized'
    df = df[~((df['aggregation'].isin(['min','max'])) & (df['Optimization Family'] == 'unoptimized'))]
    df.loc[(df['aggregation'] == 'mean') & (df['Optimization Family']=='unoptimized'),'aggregation'] = ''
    
    # concat aggregation with optimization family
    df['Optimization'] = df['Optimization Family'] + ' ' + df['aggregation']
    
    colors_list = list(mcolors.TABLEAU_COLORS.keys())
    
    colors = {}
    styles = {}
    markers = {}
    for i,_family in enumerate(df['Optimization Family'].unique()):
        for aggregation in ['min','mean','max','']:
            colors[_family + ' ' + aggregation] = colors_list[i]
            if aggregation == '' or aggregation == 'mean':
                styles[_family + ' ' + aggregation] = '-'
                markers[_family + ' ' + aggregation] = 'o'
            else:
                styles[_family + ' ' + aggregation] = ':'
                markers[_family + ' ' + aggregation] = 'x'
                
        
    # Loop over each Parameter to create individual plots
    for param in df['Parameter'].unique():
        param_data = df[df['Parameter'] == param]

        are_numeric = [str(v).isnumeric() for v in param_data['Value'].tolist()]
        if all(are_numeric):
            param_data['Value'] = param_data['Value'].astype(float)


        plot_parameter_performance(param_data,f'{output_path}/{param}.jpeg',colors,styles,markers,performance_metric=['Ben', 'Mal', 'Ben&mal', 'Hallucination'])

def plot_bias_graphs(df:pd.DataFrame,output_path):
    
    os.makedirs(output_path, exist_ok=True)

    df['Optimization'] = df['Parameter']
    
    df1 = df[["Value","Optimization","Ben"]]
    df1['Metric'] = 'Benign'
    df1['Marker'] = 'x'
    df1['LineStyle'] = '-'
    df1.rename(columns={'Ben':'Score'},inplace=True)
    
    df2 = df[["Value","Optimization","Hallucination"]] 
    df2['Metric']= 'Hallucination'
    df2['Marker'] = '*'
    df2['LineStyle'] = '--'
    df2.rename(columns={'Hallucination':'Score'},inplace=True)
            
    df = pd.concat([df1,df2]) 

    allowed_colors = {
        'Original queries':'tab:red',
        'Mutated queries':'tab:green'
    }
    df['Color'] = df['Optimization'].apply(lambda x: allowed_colors[x])
    
    df['Optimization'] = df['Optimization'] + ' ' + df['Metric']
    colors,styles,markers = {},{},{}
    metadata = {}
    for i,record in df.iterrows():
        colors[record['Optimization']] = record['Color']
        styles[record['Optimization']] = record['LineStyle']
        markers[record['Optimization']] = record['Marker']
        metadata[record['Optimization']] = {'color':record['Color'],'style':'-','marker':record['Marker']}

    json.dump(metadata,open(f'{output_path}/styles.json','w',encoding='utf-8'),indent=4)
    
    df.to_csv(f'./tmp.csv',index=False)
    print(f'{output_path}/bias.jpeg')
    plot_parameter_performance(df[['Optimization','Value','Score']],f'{output_path}/bias.jpeg',colors,styles,markers,['Score'])
    
def plot_num_docs_graph(df:pd.DataFrame,output_path, performance_metric = ['Ben','Mal', 'Ben&mal', 'Hallucination']): 
    """
        This function plots the desired performance at changing K,bum_benign and num_adv docs.
    Args:
        df (pd.DataFrame): The dataframe containing the results
        output_path (_type_): The path where the plots will be saved
        performance_metric (list, optional): The list of performance metrics to plot. Defaults to ['Ben','Mal', 'Ben&mal', 'Hallucination'].
    """
    os.makedirs(output_path, exist_ok=True)
    # We plot a graph for each value of K 
    for K in df['K'].unique().tolist(): 
        
        # Filter the data
        df_k = df[df['K'] == K]
        
        df_k['Optimization'] = df_k['NumBenDocs']
        df_k['Value'] = df_k['NumAdvDocs']
        
        colors = {}
        styles = {}
        markers = {}
        
        metadata = {}
        for i,optimization in enumerate(df_k['Optimization'].unique()):
            colors[optimization] = colors_optimized[i]
            styles[optimization] = styles_optimized
            markers[optimization] = marker_optimized
            metadata[optimization] = {'color':colors_optimized[i],'style':styles_optimized,'marker':marker_optimized}
            
        json.dump(metadata,open(f'{output_path}/styles.json','w',encoding='utf-8'),indent=4)
        # Plot 
        plot_parameter_performance(df_k,f'{output_path}/K={K}.jpeg',colors,styles,markers,performance_metric)