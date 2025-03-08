import re
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sys

def parse_log(file_path):
    data = []
    current_iteration = None
    with open(file_path, 'r') as file:
        for line in file:
            if line.startswith("Evaluating iteration"):
                current_iteration = int(line.split()[-1][:-3])
            elif "/datasets/evo2" in line:
                if 'eukaryote_ncbi' in line:
                    kingdom = line.split('/')[5].split('_')[3]
                    dataset = f'eukaryote_ncbi_{kingdom}'
                else:
                    dataset = line.split('/')[3]
            elif 'results at the end of training for val data' in line:
                lm_ppl = float(line.split()[-2])
                data.append({'iteration': current_iteration, 'dataset': dataset, 'lm_ppl': lm_ppl})
    return pd.DataFrame(data)


def create_plot(df):
    df = df[df['iteration'] > 10]
    
    plt.figure(figsize=(14, 10))
    sns.set_style("whitegrid")
    sns.set_palette("husl", n_colors=len(df['dataset'].unique()))

    max_iter = max(df['iteration'])

    df_last = df[df['iteration'] == max_iter]
    order = sorted([
        (dataset, ppl) for (dataset, ppl) in zip(df_last['dataset'], df_last['lm_ppl'])
    ], key=lambda x: -x[1])
    order = [ dataset for dataset, _ in order ]

    ax = sns.lineplot(x='iteration', y='lm_ppl', hue='dataset', data=df, marker='o', hue_order=order)

    plt.title('Language Model Perplexity (lm_ppl) Across Iterations', fontsize=16)
    plt.xlabel('Iteration', fontsize=12)
    plt.ylabel('lm_ppl', fontsize=12)
    plt.legend(
        title=f'Dataset (ordered by ppl at iteration {max_iter})',
        bbox_to_anchor=(1.05, 1),
        loc='upper left',
    )
    plt.tight_layout()

    plt.savefig('evaluate_per_ds.svg', format='svg', dpi=300, bbox_inches='tight')
    plt.close()

if __name__ == "__main__":
    log_file_path = sys.argv[1]
    df = parse_log(log_file_path)
    create_plot(df)
    print("Plot saved as 'evaluate_per_ds.svg'")
