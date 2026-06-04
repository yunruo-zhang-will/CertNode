from main import main
import argparse
import matplotlib.pyplot as plt
import math
import numpy as np
import os


def plot(k_list_all, acc_list_all, x_max, use_marker, args):

    print("Plotting results...")
    
    # 设置字体为Arial
    plt.rcParams['font.family'] = ['Arial', 'sans-serif']
    plt.rcParams['font.monospace'] = ['Arial', 'sans-serif']

    # 创建图形，指定精确尺寸 1.51 x 0.935 英寸
    plt.figure(figsize=(1.51, 1.1))

    # 读取数据
    x = np.array(k_list_all[0])
    y_array_all = [np.array(acc_list) for acc_list in acc_list_all]

    # set y_max
    max_acc = max([max(y_array) for y_array in y_array_all])
    y_max = math.floor(max_acc * 10) / 10 + 0.1

    # 定义自定义RGB颜色
    color1 = '#9271B1'  # 紫色 #B696B6
    color2 = '#4C9AC9'  # 蓝色  
    color3 = '#66C1A4'  # 绿色
    color4 = '#FCBB44'  # 黄色
    color5 = '#C25759'  # 红色
    color6 = '#7C9895'
    color7 = '#FCB2AF'

    if args.exp == 1:
        colors = [color1, color2, color3, color4, color5]
        labels = ['p = 0.9', 'p = 0.95', 'p = 0.98', 'p = 0.99', 'p = 0.995']
        xticks = [0, 25, 50, 75, 100, 125, 150]
        yticks = np.arange(0, y_max + 0.1, 0.1).tolist()
        figure_dir = f'./results/exp1_plots/'
    elif args.exp == 2:
        colors = [color1, color2, color3, color4, color5]
        labels = ['p = 0.9', 'p = 0.95', 'p = 0.98', 'p = 0.99', 'p = 0.995']
        xticks = [0, 25, 50, 75, 100, 125, 150]
        yticks = np.arange(0, y_max + 0.1, 0.1).tolist()
        figure_dir = f'./results/exp2_plots/'
    else:
        raise Exception(f"unknown exp: {args.exp}")

    # 绘制多条曲线
    for i, y in enumerate(y_array_all):
        if use_marker:
            plt.plot(x, y, 
                    color=colors[i],           # 指定RGB颜色
                    linewidth=1.0,          # 线宽
                    linestyle='-',          # 实线
                    marker='o',             # 圆形数据点
                    markersize=2,           # 标记大小
                    markerfacecolor=colors[i], # 标记填充颜色
                    markeredgecolor=colors[i], # 标记边缘颜色
                    markevery=10,           # 每20个点显示一个标记
                    label=labels[i])  # 图例标签
        else:
            plt.plot(x, y, 
                    color=colors[i],           # 指定RGB颜色
                    linewidth=1.0,          # 线宽
                    linestyle='-',          # 实线
                    label=labels[i])  # 图例标签

    # 设置坐标轴范围
    plt.xlim(0, x_max)  # 横坐标范围 
    plt.ylim(0, y_max)  # 纵坐标范围

    # 设置刻度
    plt.xticks(xticks)
    plt.yticks(yticks)

    # 隐藏坐标轴刻度线
    plt.tick_params(axis='both', which='both', length=0)

    # 设置坐标轴标签字体大小（使用Arial字体）
    plt.xlabel('Perturbation Size', fontsize=5, fontname='Arial')
    plt.ylabel('Certifeid Accuracy', fontsize=5, fontname='Arial')

    # 设置刻度字体大小（使用Arial字体）
    plt.tick_params(axis='both', which='major', labelsize=5)

    # 添加网格（细网格，适合小图）
    plt.grid(True, linestyle=':', alpha=0.3, linewidth=0.5)

    # 添加图例（使用Arial字体，小字体，紧凑布局）
    plt.legend(fontsize=4, 
            loc='lower left', 
            frameon=True,
            fancybox=True, 
            framealpha=0.8,
            handlelength=1.5,
            prop={'family': 'Arial', 'size': 4})

    # 调整布局，确保所有元素都在小图范围内
    plt.tight_layout(pad=0.1)

    # 保存为PDF - 适合LaTeX使用
    if not os.path.exists(figure_dir):
        os.makedirs(figure_dir)
    pdf_filename = f'{figure_dir}/{args.dataset}_{args.model}_{args.n_smoothing}.pdf'
    plt.savefig(pdf_filename, 
                format='pdf', 
                dpi=600,                    # 高分辨率确保小图清晰
                bbox_inches='tight', 
                pad_inches=0.01,
                transparent=False)
    
    print(f'Save figure to {pdf_filename}')


if __name__ == "__main__":
    
    # Model Settings=======================================
    parser = argparse.ArgumentParser(description='certify GNN node injecttion')

    parser.add_argument('-gpuID', type=int, default=0)
    parser.add_argument('-seed', type=int, default=2020)
    parser.add_argument('-lr', type=float, default=0.001, help='learning rate')
    parser.add_argument('-clip_max', type=float, default=2.0, help='gradient clipping max norm')
    parser.add_argument('-criterion', type=str, default='nll', choices=['nll', 'ce'], help='loss function')
    parser.add_argument('-patience', type=int, default=30, help='patience for early stopping')
    parser.add_argument('-epochs', type=int, default=1000, help='training epoch')
    parser.add_argument('-save_model', action='store_true', default=True, help="save model")
    parser.add_argument('-task', type=str, default='None', choices=['Node', 'Graph'], 
                        help='GNN tasks, Node for node classification, Graph for graph classification')
    parser.add_argument('-dataset', type=str, default='CiteSeer', 
                        choices=['PubMed', 'CiteSeer', 'computers', 'Cora-ML', 'Amazon', # for node classification, computers is Amazon-C, Amazon is Amazon2M
                                'Mutagenicity', 'PROTEINS', 'DD', 'AIDS'])    # for graph classification
    parser.add_argument('-model', type=str, default='GAT', choices=['GCN', 'GAT', 'GSAGE', 'GIN', 'APPNP'], help='GNN model')
    parser.add_argument('-n_hidden', type=int, default=64, help='size of hidden layer')
    parser.add_argument('-drop', type=float, default=0.5, help='dropout rate')
    parser.add_argument('-weight_decay', type=float, default=0, help='weight_decay rate') # 1e-3
    parser.add_argument('-n_per_class', type=int, default=50, help='sample numebr per class')
    parser.add_argument('-force_training', action='store_true', default=False, 
                        help="force training even if pretrained model exist")

    # Certify setting----------------------
    #parser.add_argument('-certify_mode', type=str, default='evasion', choices=['evasion', 'poisoning'])
    parser.add_argument('-p_n', type=float, default=0.9, help='probability of deleting nodes')
    parser.add_argument('-n_smoothing', type=int, default=10000, help='number of smoothing samples evalute (N)')
    parser.add_argument('-conf_alpha', type=float, default=0.01, help='confident alpha for statistic testing')

    # Dir setting-------------------------
    parser.add_argument('-output_dir', type=str, default='./results/', help='output directory')

    # Experiment setting------------------
    parser.add_argument('-exp', type=int, default=1, help='experiment id')

    args = parser.parse_args()

    if args.exp == 1:
        # this exp eval certified accuracy under different p
        x_max = 150
        k_list_all = []
        acc_list_all = []
        for p in [0.9, 0.95, 0.98, 0.99, 0.995]:
            args.p_n = p
            print(f"================= Experiment 1: certify under p={p} =================")
            k_list, acc_list = main(args)
            k_list_all.append(k_list)
            acc_list_all.append(acc_list)
    else:
        raise Exception(f"unknown exp: {args.exp}")

    plot(k_list_all, acc_list_all, x_max, use_marker=False, args=args)



# For node classification
# python plot.py -epochs 200 -lr 0.002 -criterion ce -task Node -dataset CiteSeer -model GCN -n_smoothing 10000 -gpuID 5


