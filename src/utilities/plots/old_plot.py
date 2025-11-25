import json
import matplotlib.pyplot as plt
import os
import pandas as pd
import textwrap

def character_trunc_function(x):
    return textwrap.fill(x.replace('-', ' '), 20)

def plot_bias_graph(df,output_path,legend=False,stat_names= ['Ben', 'Hallucination'],show=True): 
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    
    attacks = [f for f  in df['Attack'].unique().tolist() if f.lower() != 'unoptimized']
    colors = ['g','c','m','y','b','k']
    line_types = ['o-','^--','--']
    styles = {'unoptimized': 'r^--'}
    
    # Replace terms in the parameter names to make them more readable
    replace_terms = {
        'MsMarsco': '',
        'MsMarco':'',
        'Openai': '',
        'Claude3-': '',
        '-8B': '',
        '-13B': '',
    }
    for term, replacement in replace_terms.items():
        df['Value'] = df['Value'].str.replace(term, replacement)
        
    df['Value'] = df.apply(lambda x: x['Value'].strip(), axis=1)
    
    for attack in attacks: 
        for i, stat_name in enumerate(stat_names):
            styles[(attack,stat_name)] = colors[attacks.index(attack)] + line_types[i]
    # Dump styles to json so later on we can re-identify the lines
    #with open(f'{output_path}/styles_benchmark.json', 'w') as fp:
        #json.dump(styles, fp)
    
    parameters = df['Parameter'].unique()
    
    y_min, y_max = 1,0
    for stat_name in ['Ben','Mal', 'Ben&mal', 'Hallucination']:
        y_min = min(y_min,df[stat_name].min())
        y_max = max(y_max,df[stat_name].max())
    y_max = min(1,y_max+0.015)
    
    for parameter in parameters: 
                
        parameter_df = df[df['Parameter'] == parameter]
                
        parameter_df =parameter_df.sort_values('Value',ascending=True)
        
        fig, axes = plt.subplots(nrows=1, ncols=1,sharey="row")
        
        for i,stat_name in enumerate(stat_names): 
        
            parameter_df['stat'] = stat_name
            
            # Add here incex = ['Value','Injection strategy'] if we are testing multiple injection strategies concurrently
            pivot_df = parameter_df.pivot(index='Value', columns=['Attack','stat'], values=stat_name)
            pivot_df = pivot_df.sort_values('Value')
            ax = pivot_df.plot(kind='line',ylim=[y_min,y_max],legend=legend, style=styles, figsize=(5, 3), xlabel='',ax=axes)
            
            # Set x-ticks if the index is numeric
            if pd.api.types.is_numeric_dtype(pivot_df.index):
                ax.set_xticks(pivot_df.index)  # This assumes that pivot_df.index has the correct numeric x-values
                ax.set_xticklabels(pivot_df.index)
            else:
                
                x_ticks = [character_trunc_function(f) for f in pivot_df.index]
                x_ticks = [s.replace('map_','') for s  in x_ticks]
                ax.set_xticks(range(len(pivot_df.index)))
                ax.set_xticklabels(x_ticks)
            
            ax.grid(axis="y")
            ax.margins(x=0.15)
        
        #background_ax.tick_params(labelleft=False, labelbottom=False, left=False, right=False, top=False,bottom=False )
        #background_ax.get_shared_y_axes().join(background_ax,axes[0])
        #background_ax.grid(axis="y")
        
        plt.tight_layout()
        plt.savefig(f'{output_path}/{parameter}.png',dpi=200,bbox_inches = 'tight')
        plt.show()
        plt.close()

def plot_individual_parameters_graphs(df,output_path,legend=False): 
    
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    
    attacks = df['Attack'].unique().tolist()
    injection_strategies = df['Injection strategy'].unique().tolist()
    colors = ['b','g','r','c','m','y','k']
    line_types = ['o-','^-','*-']
    styles = {}
    
    if len(injection_strategies) > 2:
        for i,row in df[['Attack','Injection strategy']].drop_duplicates().iterrows():
            styles[f'{row["Attack"]}-{row["Injection strategy"]}'] = colors[attacks.index(row["Attack"])] + line_types[injection_strategies.index(row["Injection strategy"])]
        df['Attack'] = df[['Attack', 'Injection strategy']].agg('-'.join, axis=1)
    else: 
        for attack in attacks: 
            styles[f'{attack}'] = colors[attacks.index(attack)] + line_types[0]
        
    # Dump styles to json so later on we can re-identify the lines
    with open(f'{output_path}/styles.json', 'w') as fp:
        json.dump(styles, fp)
    
    parameters = df['Parameter'].unique()    
    for parameter in parameters: 
                
        parameter_df = df[df['Parameter'] == parameter]

        if parameter in ['K','chunk_size','overlap_size','temperature']: 
            parameter_df['Value']= pd.to_numeric(parameter_df['Value'], errors='coerce')
        
        parameter_df =parameter_df.sort_values('Value',ascending=True)
        for result in ['Ben','Mal', 'Ben&mal', 'Hallucination']: 
            
            # Add here incex = ['Value','Injection strategy'] if we are testing multiple injection strategies concurrently
            pivot_df = parameter_df.pivot(index='Value', columns='Attack', values=result)
            pivot_df = pivot_df.sort_values('Value')
                        
            ax = pivot_df.plot(kind='line',ylim=[0,1],legend=legend, style=styles, figsize=(5, 3), xlabel='')
            
            # Set x-ticks if the index is numeric
            if pd.api.types.is_numeric_dtype(pivot_df.index):
                ax.set_xticks(pivot_df.index)  # This assumes that pivot_df.index has the correct numeric x-values
                ax.set_xticklabels(pivot_df.index)
            else:
                ax.set_xticks(range(len(pivot_df.index)))
                ax.set_xticklabels([character_trunc_function(f) for f in pivot_df.index])
            
            ax.margins(x=0.1)
                
            plt.tight_layout()
            plt.savefig(f'{output_path}/{parameter}_{result}.png',dpi=100)
            plt.close()

def plot_parameters_graphs(df,output_path, legend, stat_names= ['Ben_baseline','Ben','Mal', 'Ben&mal', 'Hallucination']): 
    
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    
    attacks = [f for f  in df['Attack'].unique().tolist() if f.lower() != 'unoptimized']
    injection_strategies = df['Injection strategy'].unique().tolist()
    colors = ['g','c','m','y','b','k']
    line_types = ['o-','*-']
    styles = {'unoptimized': 'r^--'}
    
    # Replace terms in the parameter names to make them more readable
    replace_terms = {
        'MsMarsco': '',
        'MsMarco':'',
        'Openai': '',
        'Claude3': '',
        '8B': '',
        '13B': '',
    }
    for term, replacement in replace_terms.items():
        df['Value'] = df['Value'].str.replace(term, replacement)
    df['Value'] = df.apply(lambda x: x['Value'].strip(), axis=1)
    
    if len(injection_strategies) > 2:
        for i,row in df[['Attack','Injection strategy']].drop_duplicates().iterrows():
            if row["Attack"] == 'unoptimized':
                continue
            styles[f'{row["Attack"]}-{row["Injection strategy"]}'] = colors[attacks.index(row["Attack"])] + line_types[injection_strategies.index(row["Injection strategy"])]
        df['Attack'] = df[['Attack', 'Injection strategy']].agg('-'.join, axis=1)
    else: 
        for attack in attacks: 
            styles[f'{attack}'] = colors[attacks.index(attack)] + line_types[0]
        
    # Dump styles to json so later on we can re-identify the lines
    with open(f'{output_path}/styles_benchmark.json', 'w') as fp:
        json.dump(styles, fp)
    
    parameters = df['Parameter'].unique()
    
    y_min, y_max = 1,0
    for stat_name in stat_names:
        y_min = min(y_min,df[stat_name].min())
        y_max = max(y_max,df[stat_name].max())
    y_max = min(1,y_max+0.015)
    
    for c, parameter in enumerate(parameters): 
                
        parameter_df = df[df['Parameter'] == parameter]

        if parameter in ['K','chunk_size','overlap_size','temperature']: 
            parameter_df['Value']= pd.to_numeric(parameter_df['Value'], errors='coerce')
        
        parameter_df =parameter_df.sort_values('Value',ascending=True)
        
        fig, axes = plt.subplots(nrows=1, ncols=len(stat_names),sharey="row")
        
        for i,result in enumerate(stat_names): 
            
            # Add here incex = ['Value','Injection strategy'] if we are testing multiple injection strategies concurrently
            pivot_df = parameter_df.pivot(index='Value', columns='Attack', values=result)
            pivot_df = pivot_df.sort_values('Value')            
            
            ax = pivot_df.plot(kind='line',ylim=[y_min,y_max],legend=legend, style=styles, figsize=(5*len(stat_names), 3), xlabel='',ax=axes[i])
            
            # Set x-ticks if the index is numeric
            if pd.api.types.is_numeric_dtype(pivot_df.index):
                ax.set_xticks(pivot_df.index)  # This assumes that pivot_df.index has the correct numeric x-values
                ax.set_xticklabels(pivot_df.index)
            else:
                
                x_ticks = [character_trunc_function(f) for f in pivot_df.index]
                x_ticks = [s.replace('map_','') for s  in x_ticks]
                ax.set_xticks(range(len(pivot_df.index)))
                ax.set_xticklabels(x_ticks)
            
            ax.grid(axis="y")
            ax.margins(x=0.15)
        
        #background_ax.tick_params(labelleft=False, labelbottom=False, left=False, right=False, top=False,bottom=False )
        #background_ax.get_shared_y_axes().join(background_ax,axes[0])
        #background_ax.grid(axis="y")
        
        plt.tight_layout()
        plt.savefig(f'{output_path}/{parameter}.png',dpi=200,bbox_inches = 'tight')
        plt.show()
        plt.close()

def plot_parameters_graphs_old(df,output_path, legend, stat_names= ['Ben','Mal', 'Ben&mal', 'Hallucination']): 
    
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    
    attacks = [f for f  in df['Attack'].unique().tolist() if f.lower() != 'unoptimized']
    injection_strategies = df['Injection strategy'].unique().tolist()
    colors = ['g','c','m','y','b','k']
    line_types = ['o-','*-']
    styles = {'unoptimized': 'r^--'}
    
    # Replace terms in the parameter names to make them more readable
    replace_terms = {
        'MsMarsco': '',
        'MsMarco':'',
        'Openai': '',
        'Claude3': '',
        '8B': '',
        '13B': '',
    }
    for term, replacement in replace_terms.items():
        df['Value'] = df['Value'].str.replace(term, replacement)
    df['Value'] = df.apply(lambda x: x['Value'].strip(), axis=1)
    
    if len(injection_strategies) > 2:
        for i,row in df[['Attack','Injection strategy']].drop_duplicates().iterrows():
            if row["Attack"] == 'unoptimized':
                continue
            styles[f'{row["Attack"]}-{row["Injection strategy"]}'] = colors[attacks.index(row["Attack"])] + line_types[injection_strategies.index(row["Injection strategy"])]
        df['Attack'] = df[['Attack', 'Injection strategy']].agg('-'.join, axis=1)
    else: 
        for attack in attacks: 
            styles[f'{attack}'] = colors[attacks.index(attack)] + line_types[0]
        
    # Dump styles to json so later on we can re-identify the lines
    with open(f'{output_path}/styles_benchmark.json', 'w') as fp:
        json.dump(styles, fp)
    
    parameters = df['Parameter'].unique()
    
    y_min, y_max = 1,0
    for stat_name in ['Ben','Mal', 'Ben&mal', 'Hallucination']:
        y_min = min(y_min,df[stat_name].min())
        y_max = max(y_max,df[stat_name].max())
    y_max = min(1,y_max+0.015)
    
    for parameter in parameters: 
                
        parameter_df = df[df['Parameter'] == parameter]

        if parameter in ['K','chunk_size','overlap_size','temperature']: 
            parameter_df['Value']= pd.to_numeric(parameter_df['Value'], errors='coerce')
        
        parameter_df =parameter_df.sort_values('Value',ascending=True)
        
        fig, axes = plt.subplots(nrows=1, ncols=len(stat_names),sharey="row")
        
        for i,result in enumerate(stat_names): 
            
            # Add here incex = ['Value','Injection strategy'] if we are testing multiple injection strategies concurrently
            pivot_df = parameter_df.pivot(index='Value', columns='Attack', values=result)
            pivot_df = pivot_df.sort_values('Value')
                        
            ax = pivot_df.plot(kind='line',ylim=[y_min,y_max],legend=legend, style=styles, figsize=(5*len(stat_names), 3), xlabel='',ax=axes[i])
            
            # Set x-ticks if the index is numeric
            if pd.api.types.is_numeric_dtype(pivot_df.index):
                ax.set_xticks(pivot_df.index)  # This assumes that pivot_df.index has the correct numeric x-values
                ax.set_xticklabels(pivot_df.index)
            else:
                
                x_ticks = [character_trunc_function(f) for f in pivot_df.index]
                x_ticks = [s.replace('map_','') for s  in x_ticks]
                ax.set_xticks(range(len(pivot_df.index)))
                ax.set_xticklabels(x_ticks)
            
            ax.grid(axis="y")
            ax.margins(x=0.15)
        
        #background_ax.tick_params(labelleft=False, labelbottom=False, left=False, right=False, top=False,bottom=False )
        #background_ax.get_shared_y_axes().join(background_ax,axes[0])
        #background_ax.grid(axis="y")
        
        plt.tight_layout()
        plt.savefig(f'{output_path}/{parameter}.png',dpi=200,bbox_inches = 'tight')
        plt.show()
        plt.close()

def plot_num_docs_graphs(df_orig, output_path, legend):
           
    for K in df_orig['Attack'].unique():
        
        df = df_orig[df_orig['Attack'] == K]    
        
        if not os.path.exists(output_path):
            os.makedirs(output_path)
        
        df['benign_doc_count'] = df.apply(lambda x: x['Value'].split('-')[0], axis=1)
        df['mal_doc_count'] = df.apply(lambda x: x['Value'].split('-')[1], axis=1)

        styles = {}
        colors = ['g','c','m','y','b','k']

        for i, n in enumerate(df['benign_doc_count'].unique()):
            styles[n] = colors[i] + 'o-'
            
        # Dump styles to json so later on we can re-identify the lines
        with open(f'{output_path}/styles_num_docs.json', 'w') as fp:
            json.dump(styles, fp)
        
        fig, axes = plt.subplots(nrows=1, ncols=4,sharey="row")


        for i,result in enumerate(['Ben','Mal', 'Ben&mal', 'Hallucination']): 
                
                pivot_df = df.pivot(index=['mal_doc_count'], columns='benign_doc_count', values=result)
                pivot_df = pivot_df.sort_values('mal_doc_count')
                
                ax = pivot_df.plot(kind='line',ylim=[0,1],legend=legend, style=styles, figsize=(20, 3), xlabel='',ax=axes[i])
                
                # Set x-ticks if the index is numeric
                if pd.api.types.is_numeric_dtype(pivot_df.index):
                    ax.set_xticks(pivot_df.index)  # This assumes that pivot_df.index has the correct numeric x-values
                    ax.set_xticklabels(pivot_df.index)
                
                ax.grid(axis="y")
                ax.margins(x=0.1)

        plt.tight_layout()
        plt.savefig(f'{output_path}/benchmark-num-docs-{K}.png',dpi=200,bbox_inches = 'tight')
        plt.show()
        plt.close()

def plot_adv_docs_distribution(responses,graph_output_path, consider_only_first_mal = False): 
    
    ranks = []
    for response in responses: 
        for rank, relevant_doc in enumerate(response['results']['relevant_documents']):
            if relevant_doc['is_adversarial']:
                ranks.append(int(rank))
                if consider_only_first_mal: 
                    break
                
    plt.hist(ranks, bins=25)
    plt.ylabel('Count')
    plt.xlabel('Rank')
    plt.savefig(graph_output_path, bbox_inches='tight')
    plt.close()
    