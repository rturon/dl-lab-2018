import argparse

from cnn_mnist import mnist, train_and_validate, test

parser = argparse.ArgumentParser()
parser.add_argument("--gpu", help="perform computations on GPU")
args = parser.parse_args()
if args.gpu == '1' or args.gpu == 'true' or args.gpu == 'True':
    DEVICE = 'gpu'
else:
    DEVICE = 'cpu'

# get data
train_x, train_y, val_x, val_y, test_x, test_y = mnist('../exercise1/data')

# use smaller data set for debugging
train_x = train_x[:10000]
train_y = train_y[:10000]
val_x = val_x[:1000]
val_y = val_y[:1000]

# build and train a network
learning_curve, model = train_and_validate(train_x, train_y, val_x, val_y,
                                           num_epochs=2,
                                           lr=0.01,
                                           num_filters=16,
                                           kernel_size=3,
                                           batch_size=64,
                                           device=DEVICE)

# compute the network's test error
test_err = test(test_x, test_y, model)
print('Final test error: %.4f' % test_err)
# 2: test the effect of the learning rate
learning_rates = [0.1, 0.01, 0.001, 0.0001]

# plot all learning curves in one figure (validation performance after
# each epoch)

# 3: test different filter sizes
filter_sizes = [1, 3, 5, 7]
