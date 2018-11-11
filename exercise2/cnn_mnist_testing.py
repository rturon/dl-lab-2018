import argparse
import matplotlib.pyplot as plt
import numpy as np

from cnn_mnist import mnist, train_and_validate, test

NUM_EPOCHS = 3
def plot_learning_curves(learning_curves_lst, legend):
    # legend: dictionary of strings
    # graph_labels: legend of each curve
    # title: title of the plot

    plt.figure(figsize=(12, 6))

    plt.title(legend['title'], size=22)
    plt.xlabel('Epoch',size=18)
    plt.ylabel('Validation error',size=18)
    # plt.xlim(0,len(learning_curves_lst[0]))
    for i in range(len(learning_curves_lst)):
        plt.plot(learning_curves_lst[i], label=legend['graph_labels'][i])

    plt.legend(fontsize=16)

    plt.savefig('%s.pdf' % legend['title'])

    # plt.show()

if __name__ == "__main__":

    # get data
    train_x, train_y, val_x, val_y, test_x, test_y = mnist('../exercise1/data')

    # use smaller data set for debugging
    train_x = train_x[:1000]
    train_y = train_y[:1000]
    val_x = val_x[:100]
    val_y = val_y[:100]


    # 2: test the effect of the learning rate
    learning_rates = [0.1, 0.01, 0.001, 0.0001]

    learning_curves_lr = []
    for lr in learning_rates:
        learning_curve, model = train_and_validate(train_x, train_y,
                                                   val_x, val_y,
                                                   num_epochs=NUM_EPOCHS,
                                                   lr=lr,
                                                   num_filters=16,
                                                   kernel_size=3,
                                                   batch_size=64)
        learning_curves_lr.append(learning_curve)

        # compute the network's test error
        # test_err = test(test_x, test_y, model)
        # print('Final test error: %.4f' % test_err)
    # plot all learning curves in one figure (validation performance after
    # each epoch)
    plot_learning_curves(learning_curves_lr, {'graph_labels': learning_rates,
                                              'title':
                                              'Validation Performance for Different Learning Rates'})
    lc_lr_np = np.array(learning_curves_lr)
    np.save('learning_rates', lc_lr_np)

    # 3: test different filter sizes
    filter_sizes = [1, 3, 5, 7]
    learning_curves_fs = []
    for fs in filter_sizes:
        learning_curve, model = train_and_validate(train_x, train_y,
                                                   val_x, val_y,
                                                   num_epochs=NUM_EPOCHS,
                                                   lr=0.1,
                                                   num_filters=16,
                                                   kernel_size=fs,
                                                   batch_size=64)
        learning_curves_fs.append(learning_curve)

    plot_learning_curves(learning_curves_fs, {'graph_labels': filter_sizes,
                                              'title':
                                              'Validation Performance for Different Filter Sizes'})
    lc_fs_np = np.array(learning_curves_fs)
    np.save('filter_sizes', lc_fs_np)
