import matplotlib.pyplot as plt
from scipy.stats import norm
import matplotlib.mlab as mlab
import seaborn as sns
import pickle

plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["axes.labelsize"] = 12

num_subplots = 4
colours = ['royalblue', 'seagreen', 'orange']
edge_colours = ['royalblue', 'seagreen', 'orange']
titles = ['Exit Time Distributions of Ornstein–Uhlenbeck Process',
          '', '']
annotations = [r'$\alpha=0.525$   $\sigma=0.688$   $x_0=9.6$',
               r'$\alpha=0.0105$   $\sigma=0.011$   $x_0=1.06$',
               r'$\alpha=0.235$   $\sigma=1.4$   $x_0=8.5$']
ann_loc = [[8, 0.27],[8, 0.12],[8, 0.23]]
          
n_bins = [30, 100, 50]
f, axes = plt.subplots(num_subplots, 1, gridspec_kw={'height_ratios': [1,0.5,1,0.5]})
#f, axes = plt.subplots(num_subplots, 1, gridspec_kw={'height_ratios': [1,0.5,1,0.5,1,0.5]})


# Data for mean/var exit time plots
x_data = [[2.163,5.633], [1.389,9.393], [-0.167,12.169]]   # Mean +- var range for params 1,2,3
y1 = [1.0, 1.0]
means = [3.898, 5.391, 6.001]

for plt_idx_base in range(2):
    # Plot histograms
    data = pickle.load(open('OU_process/p_safety/safer_p_values/seed100_params'+str(plt_idx_base+1)+'_exit_times_50k.pkl', 'rb'))
    exit_times = data[0]

    plt_idx = int(plt_idx_base*2)
    colour = colours[plt_idx_base]
    edge_colour = edge_colours[plt_idx_base]
    p = sns.distplot(exit_times, hist=True, kde=True, 
                    bins=n_bins[plt_idx_base], color = colour,
                    hist_kws={'edgecolor':edge_colour},
                    kde_kws={'linewidth': 1.3}, ax=axes[plt_idx])
    axes[plt_idx].axis(xmin=-1.0, xmax=18.0)
    axes[plt_idx].grid()
    axes[plt_idx].set_title(titles[plt_idx_base], fontsize=17)
    #axes[plt_idx].set_ylabel('')


    if plt_idx == 4:
        axes[plt_idx+1].set_xlabel('Time')
    axes[plt_idx].text(ann_loc[plt_idx_base][0], ann_loc[plt_idx_base][1], annotations[plt_idx_base], horizontalalignment='left', size='medium', color='black', weight='semibold')

    # Plot means/vars
    axes[plt_idx+1].plot(x_data[plt_idx_base], y1, c=colours[plt_idx_base], marker='|', mew=1.5, ms=8)
    line, = axes[plt_idx+1].plot(means[plt_idx_base], 1.0, c=colours[plt_idx_base], marker='o', label='Mean Exit Time', linestyle='None')
    axes[plt_idx+1].axis(xmin=-1.0, xmax=18.0)
    axes[plt_idx+1].set_yticks([])
    axes[plt_idx+1].legend()

# for plt_idx in range(3,num_subplots):
#     axes[plt_idx].plot(x_data[int(plt_idx-3)], y1)
#     axes[plt_idx].axis(xmin=-3.0, xmax=10.0)
plt.tight_layout()
#plt.gcf().subplots_adjust(bottom=0.05, top=0.95)
plt.show()