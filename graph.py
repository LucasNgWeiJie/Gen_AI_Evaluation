'''
This function provides custom graph functions
'''
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def plot_metrics_line(model_names, losses, perplexities, output_file):
    '''
    This function plots a beautified line graph
    '''
    # Apply a clean Seaborn theme
    sns.set_style("whitegrid")
    
    fig, ax1 = plt.subplots(figsize=(12, 7))
    
    # Colors and markers
    loss_color = 'royalblue'
    perplexity_color = 'darkgreen'
    base_marker = 'o'
    
    # Losses Line Plot
    ax1.set_xlabel('Models', fontsize=14, fontweight='bold')
    ax1.set_ylabel('Average Loss', color=loss_color, fontsize=14)
    ax1.plot(range(len(model_names)), losses, 
             color=loss_color, marker=base_marker, markersize=8, linewidth=2, label='Loss')
    
    ax1.tick_params(axis='y', labelcolor=loss_color)
    ax1.tick_params(axis='x', rotation=45, labelsize=12)

    # Perplexities Line Plot (Twin Axis)
    ax2 = ax1.twinx()
    ax2.set_ylabel('Perplexity', color=perplexity_color, fontsize=14)
    ax2.plot(range(len(model_names)), perplexities, 
             color=perplexity_color, linestyle='--', marker=base_marker, markersize=8, linewidth=2, label='Perplexity')

    ax2.tick_params(axis='y', labelcolor=perplexity_color)

    # Add value annotations
    for i, (loss, perplexity) in enumerate(zip(losses, perplexities)):
        ax1.text(i, loss, f'{loss:.2f}', fontsize=12, color=loss_color, ha='right', va='bottom')
        ax2.text(i, perplexity, f'{perplexity:.2f}', fontsize=12, color=perplexity_color, ha='left', va='top')

    # Title and Legends
    plt.title('Model Metrics: Loss and Perplexity', fontsize=16, fontweight='bold')
    ax1.legend(loc='upper left', fontsize=12)
    ax2.legend(loc='upper right', fontsize=12)

    # X-axis labels
    plt.xticks(range(len(model_names)), model_names, ha='right', fontsize=12)
    
    # Gridlines for better readability
    ax1.grid(True, linestyle='--', alpha=0.6)

    # Save and show
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.show()


def plot_metrics_single(model_names,
                        metric_type: str,
                        metric_values,
                        output_file,
                        color='cornflowerblue'):
    '''
    This function plots a single bar graph to visualize model evaluation metrics.
    '''
    
    # Ensure the input is valid
    if len(model_names) != len(metric_values):
        raise ValueError("The number of models and metric values must be the same.")
    
    # Set up the figure with dynamic size
    fig, ax1 = plt.subplots(figsize=(max(12, len(model_names) * 1.5), 7))  # Dynamic width

    # Create a bar plot using Seaborn
    sns.barplot(x=model_names, y=metric_values, ax=ax1, color=color, edgecolor='darkblue')
    
    # Customize the x-axis and y-axis labels
    ax1.set_xlabel('Models', fontsize=14)
    ax1.set_ylabel(f'{metric_type} Score', fontsize=14, color=color)
    ax1.tick_params(axis='x', labelrotation=45, labelsize=12)
    ax1.tick_params(axis='y', labelcolor=color, labelsize=12)

    # Set title with improved styling
    plt.title(f'Model Comparison: {metric_type} Scores', fontsize=16, fontweight='bold')

    # Add gridlines for better readability
    ax1.grid(axis='y', linestyle='--', alpha=0.6)

    # Set the font size for the legend
    ax1.legend([f'{metric_type} Scores'], loc='upper left', bbox_to_anchor=(0.1, -0.15), fontsize=12, title=metric_type, title_fontsize=13)

    # Tighten layout to ensure nothing is cut off
    fig.tight_layout()

    # Save and show the plot
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.show()


def plot_metrics_bar(model_names, losses, perplexities, output_file):
    '''
    This function plots a grouped bar chart using Seaborn with two separate categories: Loss and Perplexity.
    '''
    # Set up figure and axis
    fig, ax = plt.subplots(figsize=(12, 7))
    
    # Create grouped x-axis positions
    x = np.arange(len(model_names))  # Base positions for each model
    width = 0.35  # Width for each bar

    # Define color palette
    colors = sns.color_palette("coolwarm", 2)  # Blue & Red for contrast

    # Plot bars
    bars1 = ax.bar(x - width/2, losses, width, color=colors[0], edgecolor='black', label='Loss')
    bars2 = ax.bar(x + width/2, perplexities, width, color=colors[1], edgecolor='black', label='Perplexity')

    # Add labels
    ax.set_xlabel('Models', fontsize=14, fontweight='bold')
    ax.set_ylabel('Value', fontsize=14, fontweight='bold')
    ax.set_title('Model Metrics: Loss vs Perplexity', fontsize=16, fontweight='bold', pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels(model_names, rotation=45, ha="right", fontsize=12)

    # Add values on top of bars
    for bars in [bars1, bars2]:
        ax.bar_label(bars, fmt='%.2f', fontsize=12, fontweight='bold')

    # Improve aesthetics
    sns.despine()  # Remove unnecessary borders
    ax.legend(title='Metric', fontsize=12, loc='upper right')

    # Save and show the plot
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.show()



