"""
    This file reads the results of a group of experiments and computes the statistics of the results in a single CSV for good readability.
"""
import os
from argparse import ArgumentParser
import shutil
from statistics import mean
import pandas as pd
import logging
from src.pipelines.evaluate import compute_grouped_stat
from src.utilities.plots.plots import plot_bias_graphs, plot_num_docs_graph, plot_parameters_by_injectionstr, plot_parameters_performance, plot_parameters_performance_grouping_optimizations

logger = logging.getLogger(__name__)

def format_experiment_data(experiment_results_folder,dest_folder=None):
    """_summary_

    Args:
        experiment_results_folder (_type_): _description_
    """
    
    # The folder where the results of the experiments will be stored
    root_data_folder = f'{experiment_results_folder}/data' if not dest_folder else dest_folder

    if not os.path.exists(root_data_folder):
        os.makedirs(root_data_folder)
        
        for folder1 in os.listdir(experiment_results_folder): # Iterate over timestamped folders
                        
            if folder1 in ['data','plots','baseline'] or not os.path.isdir(f'{experiment_results_folder}/{folder1}'):
                continue

            for folder2 in os.listdir(f'{experiment_results_folder}/{folder1}'): # Iterate over parameter folders
                
                os.makedirs(f'{root_data_folder}/{folder2}',exist_ok=True)
                
                if not os.path.isdir(f'{experiment_results_folder}/{folder1}/{folder2}/'):
                        continue
                                
                for folder3 in os.listdir(f'{experiment_results_folder}/{folder1}/{folder2}'): # Iterate over attack folders
                    
                    if not os.path.isdir(f'{experiment_results_folder}/{folder1}/{folder2}/{folder3}'):
                        continue
                    
                    for folder4 in os.listdir(f'{experiment_results_folder}/{folder1}/{folder2}/{folder3}'): # Iterate over parameter value folders
                        
                        # Compute paths of source and destination folders
                        path1 = f'{experiment_results_folder}/{folder1}/{folder2}/{folder3}/{folder4}'
                        path2 = f'{root_data_folder}/{folder2}/{folder3}/{folder4}'
                        
                        if os.path.exists(path2):
                            logging.warning(f'Folder {path2} already exists. Skipping copy.')
                            continue
                        
                        # if os.path.isdir(path1):
                        shutil.copytree(path1,path2)
                        # else: 
                        #     os.makedirs(f'{root_data_folder}/{folder2}/{folder3}',exist_ok=True)
                        #     shutil.copy(path1,path2)

def read_parameter_results(parameter_experiment_folder): 
    
    results_df = []
    
    result_file = os.path.join(parameter_experiment_folder, 'statistics.csv')
    if os.path.exists(result_file):
        
        # Read statistics .csv
        seed_results_df = pd.read_csv(result_file)
        
        # Compute the statistics of this seed
        seed_stats = extract_statistics(seed_results_df)

        results_df.append(pd.DataFrame(seed_stats, index=[0]))
        
    else: 
        # Go throught the seed folder
        for seed in os.listdir(parameter_experiment_folder):
            
            seed_folder_path = os.path.join(parameter_experiment_folder, seed)

            if not os.path.isdir(seed_folder_path) or seed.startswith('.'):
                continue
                        
            # Make sure that statistics.csv exists
            run_results_file = os.path.join(seed_folder_path, 'statistics.csv') 

            if not os.path.exists(run_results_file):
                logging.warning(f'Could not find result file for experiment: {seed_folder_path} with seed:{seed}')
                continue
            
            # Read statistics .csv
            seed_results_df = pd.read_csv(run_results_file)
            
            # Compute the statistics of this seed
            seed_stats = extract_statistics(seed_results_df)

            results_df.append(pd.DataFrame(seed_stats, index=[0]))
        
        if not results_df:
            logging.warning(f'Could not find any result file for experiment: {parameter_experiment_folder}')
            return None
    
    df = pd.concat(results_df)

    return df.mean().to_dict()


def group_statistics(results_folder):
    records = []
    for parameter_name in os.listdir(results_folder):
        parameter_results_folder = f'{results_folder}/{parameter_name}'
        if not os.path.isdir(parameter_results_folder):
            continue
            
        # Foreach attack considered in the evaluation
        for attack_name in os.listdir(parameter_results_folder): 
            attack_results_folder = f'{parameter_results_folder}/{attack_name}'
            if not os.path.isdir(attack_results_folder):
                continue
                        
            # Foreach value of the parameter
            for parameter_value in os.listdir(attack_results_folder): 
                parameter_value_results_folder = f"{attack_results_folder}/{parameter_value}"
                
                if not os.path.isdir(parameter_value_results_folder):
                    continue
                                    
                # Record containing details and results of a specific experiment          
                record = {}
                record['Parameter'] = parameter_name
                record['Value'] = parameter_value
                record['Optimization'] = attack_name.split('_')[0] if '_' in attack_name else attack_name
                record['Injection strategy']= attack_name.split('_')[1] if '_' in attack_name else ''
                                    
                # Collect the results computer under these settings
                statistics = read_parameter_results(parameter_value_results_folder)
                
                if statistics is None:
                    continue
                
                for k,v in statistics.items():
                    record[k] = v
                records.append(record)
                    
    return pd.DataFrame(records)

def compute_statistics(root_data_folder):
    """This function loads the statistics of the results of a group of experiments computing them from the files where necessary.

    Args:
        root_data_folder (str): Path to the root folder containing the results of the experiments

    Returns:
        pd.Dataframe: dataframe containing the statistics of the results of the experiments
    """
    df_grouped_statistics = None
    if not os.path.exists(f'{root_data_folder}/grouped_statistics.csv'):
        # Group all statistics in a single CSV and save it
        df_grouped_statistics = group_statistics(root_data_folder)
        df_grouped_statistics.to_csv(f'{root_data_folder}/grouped_statistics.csv',index=False)
        
    else:
        df_grouped_statistics = pd.read_csv(f'{root_data_folder}/grouped_statistics.csv')
    
    for k in df_grouped_statistics.columns:
        df_grouped_statistics[k] = df_grouped_statistics[k].astype(str)
    
    # cast Ben, Mal, Ben&mal and Hallucination to float
    df_grouped_statistics['Ben'] = df_grouped_statistics['Ben'].astype(float)
    df_grouped_statistics['Mal'] = df_grouped_statistics['Mal'].astype(float)
    df_grouped_statistics['Ben&mal'] = df_grouped_statistics['Ben&mal'].astype(float)
    df_grouped_statistics['Hallucination'] = df_grouped_statistics['Hallucination'].astype(float)
    
    return df_grouped_statistics

def load_statistics(root_results_folder):
    """This function loads the statistics of the results of a group of experiments computing them from the files where necessary.

    Args:
        root_data_folder (str): Path to the root folder containing the results of the experiments

    Returns:
        pd.Dataframe: dataframe containing the statistics of the results of the experiments
    """
    df_grouped_statistics = compute_statistics(f'{root_results_folder}/data')
        
    return df_grouped_statistics

def extract_statistics(df_result):
    ben_acc, mal_acc, both_acc, hall, avg_count_mal_docs_in_prompt = compute_grouped_stat(df_result)
        
    record = {
        'Ben': ben_acc,
        'Mal': mal_acc,
        'Ben&mal': both_acc,
        'Hallucination': hall,
        'Avg # mal docs in prompt': avg_count_mal_docs_in_prompt,
        #'Avg rank mal docs in prompt': avg_adv_doc_rank_in_prompt,
    }
    
    return record
    
if __name__ == '__main__': 
            
    
    parser = ArgumentParser(description="This script reads the results of a group of experiments and computes the statistics of the results in a single CSV for good readability.")
    parser.add_argument("--benchmark_root", type=str, help="Path to the root folder containing a set of experiments", required=True)
    parser.add_argument("--plot_type", type=str, help="Type of analysis to perform: [Parameters/Bias/NumDocs]", required=True)
    args = parser.parse_args()
    
    # Here is the list of the most successful attacks for each optimization strategy
    # We will use this list to select the attacks to plot
    most_successful_attacks =  [('unoptimized',''),('PAT','proximity'),('asc-natural-noreg','proximity'),('IDEM','proximity'),('query+','proximity'),('PoisonRAG-LM-targeted',''),('seo-documents-writer',''),('Phantom-corruption','')]
    
    # Theoretically, only one folder should be present in the root_data_folder and its name should be the timestamp of the experiment
    # However, it is possible that one may need to 'split' a benchmark into multiple runs with different timestamps. 
    # We create a new folder called 'data' and merge the results of all time-stemped benchmark(s) into this folder.
    # If the folder data already exists, we skip this step.
    format_experiment_data(args.benchmark_root)

    # Load all statistics in a single dataframe
    df_grouped_statistics = load_statistics(args.benchmark_root)
    df_grouped_statistics.sort_values(by=['Optimization','Injection strategy'],inplace=True)
    
    # Plot the results
    if args.plot_type.lower() == 'parameters' or args.plot_type.lower() == 'model_generations':
                
        # check if a baseline folder exists 
        baseline_folder = f'{args.benchmark_root}/baseline'
        if os.path.exists(baseline_folder):
            
            # Prepare the baseline data
            baseline_data_folder = format_experiment_data(baseline_folder,f'{baseline_folder}/data')
                
            # Load the baseline statistics
            df_baseline_statistics = compute_statistics(f'{baseline_folder}/data')
                        
            # Merge the baseline statistics with the grouped statistics
            df_grouped_statistics = df_grouped_statistics.merge(df_baseline_statistics[['Parameter', 'Value', 'Optimization', 'Injection strategy', 'Ben']], 
                                on=['Parameter', 'Value'], 
                                how='left', 
                                suffixes=('', '_Baseline'))
                                
            df_grouped_statistics['Ben_Baseline'] = df_grouped_statistics['Ben_Baseline'].astype(float)
            
        # Select the attacks to plot
        df_grouped_statistics.loc[df_grouped_statistics['Injection strategy']=='nan','Injection strategy'] = ''
        df_grouped_statistics = df_grouped_statistics.merge(pd.DataFrame(most_successful_attacks, columns=['Optimization','Injection strategy']))

        # Order by optimization strategy
        df_grouped_statistics.sort_values(by=['Optimization','Injection strategy'],inplace=True)
        base_record = df_grouped_statistics[(df_grouped_statistics['Parameter']=='model') & (df_grouped_statistics['Value']=='Llama3-8B')]
        
        # we don't have a reranker 'none' since the default is None, use the base record instead
        if args.plot_type.lower() == 'parameters' and 'reranker' in df_grouped_statistics['Parameter'].unique():
            reranker_none = base_record.copy()        
            reranker_none['Parameter'] = 'reranker'
            reranker_none['Value'] = 'None'
            df_grouped_statistics = pd.concat([df_grouped_statistics, reranker_none], ignore_index=True)
        
        # Plot the performance of the parameters
        plot_parameters_performance(df_grouped_statistics,f'{args.benchmark_root}/plots/parameters',performance_metric=['Ben_Baseline','Ben','Mal','Ben&mal','Hallucination'])
        df_grouped_statistics.to_csv(f'{args.benchmark_root}/statistics.csv',index=False)
        
    elif args.plot_type.lower() == 'bias':
        plot_bias_graphs(df_grouped_statistics,f'{args.benchmark_root}/plots')
        
    elif args.plot_type.lower() == 'numben':
        
        # select only the most successful attacks
        df_grouped_statistics.loc[df_grouped_statistics['Injection strategy']=='nan','Injection strategy'] = ''
        df_grouped_statistics = df_grouped_statistics.merge(pd.DataFrame(most_successful_attacks, columns=['Optimization','Injection strategy']))
        plot_parameters_performance(df_grouped_statistics,f'{args.benchmark_root}/plots')
        
    elif args.plot_type.lower() == 'numdocs':
        
        df_grouped_statistics[['NumBenDocs','NumAdvDocs']] = df_grouped_statistics['Value'].str.split('-',expand=True) 
        df_grouped_statistics.rename(columns={'Optimization':'K',},inplace=True)
        plot_num_docs_graph(df_grouped_statistics,f'{args.benchmark_root}/plots')
        
    else:
        raise ValueError(f"Analysis type: {args.analysis_type} is unknown.\n Supported analysis types are: ['Parameter','Bias','NumDocs']")
