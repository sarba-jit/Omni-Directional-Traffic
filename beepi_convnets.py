##############################################################
# module: beepi_convnets.py
#
# description: All ConvNet models trained, tested, and validated
# in the article: ConvNetGS1, ConvNetGS2, ConvNetGS3, ConvNetGS4,
# ConvNetGS5, ConvNet1, ConvNet2, ConvNet3, ConvNet4, ConvNet5,
# ConvNet6, ConvNet7, ConvNet8, ConvNet9, ConvNet10, VGG16, and
# ResNet-32 for CIFAR10.
#
# authors: Vladimir Kulyukin, Sarbajit Mukherjee, Prateek Vats
###############################################################


import tflearn
from tflearn import input_data, conv_2d, max_pool_2d, fully_connected, regression, dropout, batch_normalization
import tensorflow as tf

class beepi_convnets:

    ##############################################################################################
    # ConvNetGS models are models auto-designed through multi-generational greedy grid search (GS)
    # The number following the abbreviation ConvNetGS stands for the number of hidden
    # layers where a hidden layer consists of a convolutional layer followed by a maxpool layer
    ##############################################################################################

    @staticmethod
    def ConvNetGS1(tensorWidth, tensorHeight, tensorDepth, tb_verb):
        aug = tflearn.ImageAugmentation()

        conv_net = input_data(shape=[None, tensorWidth, tensorHeight, tensorDepth], data_augmentation=aug)

        # hidden layer 1
        conv_net = conv_2d(conv_net,
                           nb_filter=29,
                           filter_size=9,
                           strides=1,
                           activation='relu', name='conv_layer_1')
        conv_net = max_pool_2d(conv_net, 2)

        conv_net = fully_connected(conv_net, 2, activation='softmax', name='output')
        conv_net = regression(conv_net, optimizer='adam', learning_rate=0.001, loss='categorical_crossentropy',
                              name='targets')

        model = tflearn.DNN(conv_net, tensorboard_verbose=tb_verb)

        return model

    @staticmethod
    def ConvNetGS2(tensorWidth, tensorHeight, tensorDepth, tb_verb=0):
        aug = tflearn.ImageAugmentation()

        conv_net = input_data(shape=[None, tensorWidth, tensorHeight, tensorDepth], data_augmentation=aug)

        # hidden layer 1
        conv_net = conv_2d(conv_net,
                           nb_filter=31,
                           filter_size=12,
                           strides=1,
                           activation='relu', name='conv_layer_1')
        conv_net = max_pool_2d(conv_net, 2)

        # hidden layer 2
        conv_net = conv_2d(conv_net,
                           nb_filter=40,
                           filter_size=10,
                           strides=1,
                           activation='relu', name='conv_layer_2')
        conv_net = max_pool_2d(conv_net, 2)

        conv_net = fully_connected(conv_net, 2, activation='softmax', name='output')
        conv_net = regression(conv_net, optimizer='adam', learning_rate=0.001, loss='categorical_crossentropy',
                              name='targets')

        model = tflearn.DNN(conv_net, tensorboard_verbose=tb_verb)

        return model

    @staticmethod
    def ConvNetGS3(tensorWidth, tensorHeight, tensorDepth, tb_verb=0):
        aug = tflearn.ImageAugmentation()

        conv_net = input_data(shape=[None, tensorWidth, tensorHeight, tensorDepth], data_augmentation=aug)

        # hidden layer 1
        conv_net = conv_2d(conv_net,
                           nb_filter=27,
                           filter_size=9,
                           strides=1,
                           activation='relu', name='conv_layer_1')
        conv_net = max_pool_2d(conv_net, 2)

        # hidden layer 2
        conv_net = conv_2d(conv_net,
                           nb_filter=42,
                           filter_size=11,
                           strides=1,
                           activation='relu', name='conv_layer_2')
        conv_net = max_pool_2d(conv_net, 2)

        # hidden layer 3
        conv_net = conv_2d(conv_net,
                           nb_filter=105,
                           filter_size=11,
                           strides=1,
                           activation='relu', name='conv_layer_3')
        conv_net = max_pool_2d(conv_net, 2)

        conv_net = fully_connected(conv_net, 2, activation='softmax', name='output')
        conv_net = regression(conv_net, optimizer='adam', learning_rate=0.001, loss='categorical_crossentropy',
                              name='targets')

        model = tflearn.DNN(conv_net, tensorboard_verbose=tb_verb)

        return model

    @staticmethod
    def ConvNetGS4(tensorWidth, tensorHeight, tensorDepth, tb_verb=0):
        aug = tflearn.ImageAugmentation()

        conv_net = input_data(shape=[None, tensorWidth, tensorHeight, tensorDepth], data_augmentation=aug)

        # hidden layer 1
        conv_net = conv_2d(conv_net,
                           nb_filter=27,
                           filter_size=9,
                           strides=1,
                           activation='relu', name='conv_layer_1')
        conv_net = max_pool_2d(conv_net, 2)

        # hidden layer 2
        conv_net = conv_2d(conv_net,
                           nb_filter=38,
                           filter_size=9,
                           strides=1,
                           activation='relu', name='conv_layer_2')
        conv_net = max_pool_2d(conv_net, 2)

        # hidden layer 3
        conv_net = conv_2d(conv_net,
                           nb_filter=85,
                           filter_size=9,
                           strides=1,
                           activation='relu', name='conv_layer_3')
        conv_net = max_pool_2d(conv_net, 2)

        # hidden layer 4
        conv_net = conv_2d(conv_net,
                           nb_filter=220,
                           filter_size=10,
                           strides=1,
                           activation='relu', name='conv_layer_4')
        conv_net = max_pool_2d(conv_net, 2)

        conv_net = fully_connected(conv_net, 2, activation='softmax', name='output')
        conv_net = regression(conv_net, optimizer='adam', learning_rate=0.00038, loss='categorical_crossentropy',
                              name='targets')

        model = tflearn.DNN(conv_net, tensorboard_verbose=tb_verb)

        return model

    @staticmethod
    def ConvNetGS5(tensorWidth, tensorHeight, tensorDepth, tb_verb=0):
        aug = tflearn.ImageAugmentation()

        conv_net = input_data(shape=[None, tensorWidth, tensorHeight, tensorDepth], data_augmentation=aug)

        # hidden layer 1
        conv_net = conv_2d(conv_net,
                           nb_filter=27,
                           filter_size=9,
                           strides=1,
                           activation='relu', name='conv_layer_1')
        conv_net = max_pool_2d(conv_net, 2)

        # hidden layer 2
        conv_net = conv_2d(conv_net,
                           nb_filter=38,
                           filter_size=9,
                           strides=1,
                           activation='relu', name='conv_layer_2')
        conv_net = max_pool_2d(conv_net, 2)

        # hidden layer 3
        conv_net = conv_2d(conv_net,
                           nb_filter=85,
                           filter_size=9,
                           strides=1,
                           activation='relu', name='conv_layer_3')
        conv_net = max_pool_2d(conv_net, 2)

        # hidden layer 4
        conv_net = conv_2d(conv_net,
                           nb_filter=260,
                           filter_size=8,
                           strides=1,
                           activation='relu', name='conv_layer_4')
        conv_net = max_pool_2d(conv_net, 2)

        # hidden layer 5
        conv_net = conv_2d(conv_net,
                           nb_filter=240,
                           filter_size=9,
                           strides=1,
                           activation='relu', name='conv_layer_5')
        conv_net = max_pool_2d(conv_net, 2)

        conv_net = fully_connected(conv_net, 2, activation='softmax', name='output')
        conv_net = regression(conv_net, optimizer='adam', learning_rate=0.001, loss='categorical_crossentropy',
                              name='targets')

        model = tflearn.DNN(conv_net, tensorboard_verbose=tb_verb)

        return model

    # The version of the VGG network tested in the article
    # it is available at https://github.com/tflearn/tflearn/blob/master/examples/images/vgg_network.py.
    @staticmethod
    def VGG16(tensorWidth, tensorHeight, tensorDepth, tb_verb=0):
        network = input_data(shape=[None, tensorWidth, tensorHeight, tensorDepth])

        network = conv_2d(network, 64, 3, activation='relu')
        network = conv_2d(network, 64, 3, activation='relu')
        network = max_pool_2d(network, 2, strides=2)

        network = conv_2d(network, 128, 3, activation='relu')
        network = conv_2d(network, 128, 3, activation='relu')
        network = max_pool_2d(network, 2, strides=2)

        network = conv_2d(network, 256, 3, activation='relu')
        network = conv_2d(network, 256, 3, activation='relu')
        network = conv_2d(network, 256, 3, activation='relu')
        network = max_pool_2d(network, 2, strides=2)

        network = conv_2d(network, 512, 3, activation='relu')
        network = conv_2d(network, 512, 3, activation='relu')
        network = conv_2d(network, 512, 3, activation='relu')
        network = max_pool_2d(network, 2, strides=2)

        network = conv_2d(network, 512, 3, activation='relu')
        network = conv_2d(network, 512, 3, activation='relu')
        network = conv_2d(network, 512, 3, activation='relu')
        network = max_pool_2d(network, 2, strides=2)

        network = fully_connected(network, 4096, activation='relu')
        network = dropout(network, 0.5)
        network = fully_connected(network, 4096, activation='relu')
        network = dropout(network, 0.5)
        network = fully_connected(network, 2, activation='softmax')

        network = regression(network, optimizer='rmsprop',
                             loss='categorical_crossentropy',
                             learning_rate=0.0001)

        # Training
        model = tflearn.DNN(network, checkpoint_path='model_vgg',
                            max_checkpoints=1, tensorboard_verbose=tb_verb)
        return model

    # the version of the ResNet network tested in the article. It is ResNet-32 for CIFAR10.
    # https://github.com/tflearn/tflearn/blob/master/examples/images/residual_network_cifar10.py
    @staticmethod
    def ResNet32(tensorWidth, tensorHeight, tensorDepth, tb_verb=0):
        n = 5
        net = tflearn.input_data(shape=[None, tensorWidth, tensorHeight, tensorDepth])
        net = tflearn.conv_2d(net, 16, 3, regularizer='L2', weight_decay=0.0001)
        net = tflearn.residual_block(net, n, 16)
        net = tflearn.residual_block(net, 1, 32, downsample=True)
        net = tflearn.residual_block(net, n - 1, 32)
        net = tflearn.residual_block(net, 1, 64, downsample=True)
        net = tflearn.residual_block(net, n - 1, 64)
        net = tflearn.batch_normalization(net)
        net = tflearn.activation(net, 'relu')
        net = tflearn.global_avg_pool(net)
        # Regression
        net = tflearn.fully_connected(net, 2, activation='softmax')
        mom = tflearn.Momentum(0.1, lr_decay=0.1, decay_step=32000, staircase=True)
        net = tflearn.regression(net, optimizer=mom,
                                 loss='categorical_crossentropy')
        # Training
        model = tflearn.DNN(net, checkpoint_path='model_resnet_bee1',
                            max_checkpoints=10, tensorboard_verbose=tb_verb,
                            clip_gradients=0.)
        return model


    ########################################################################
    # These are hand-designed ConvNets tested in the article.
    ########################################################################

    # P.V.
    @staticmethod
    def ConvNet1(tensorWidth, tensorHeight, tensorDepth, tb_verb=0):
        # g = tf.Graph()
        # with g.as_default():
        aug = tflearn.ImageAugmentation()
        # aug.add_random_blur(3)
        conv_net = input_data(shape=[None, tensorWidth, tensorHeight, tensorDepth],data_augmentation=aug)

        conv_net = conv_2d(conv_net,
                           nb_filter=64,
                           filter_size=3,
                           activation='relu', name='conv_layer_1')
        conv_net = max_pool_2d(conv_net, 2)
        conv_net = batch_normalization(conv_net)
        conv_net = conv_2d(conv_net,
                           nb_filter=128,
                           filter_size=3,
                           activation='relu', name='conv_layer_2')
        conv_net = max_pool_2d(conv_net, 2)
        conv_net = batch_normalization(conv_net)
        conv_net = conv_2d(conv_net,
                           nb_filter=256,
                           filter_size=3,
                           activation='relu', name='conv_layer_3')
        conv_net = max_pool_2d(conv_net, 2)

        conv_net = dropout(conv_net, keep_prob=0.5)
        conv_net = fully_connected(conv_net, 512, activation='relu', name='fc_layer_1')

        conv_net = dropout(conv_net, keep_prob=0.5)
        conv_net = fully_connected(conv_net, 256, activation='relu', name='fc_layer_2')
        #
        conv_net = dropout(conv_net, keep_prob=0.5)
        conv_net = fully_connected(conv_net, 128, activation='relu', name='fc_layer_3')

        conv_net = dropout(conv_net, keep_prob=0.5)
        conv_net = fully_connected(conv_net, 64, activation='relu', name='fc_layer_4')

        conv_net = fully_connected(conv_net, 2, activation='softmax', name='output')
        conv_net = regression(conv_net, optimizer='adam', learning_rate=0.0001, loss='categorical_crossentropy',
                              name='targets')

        model = tflearn.DNN(conv_net, tensorboard_verbose=tb_verb)

        return model


    # D.S.
    @staticmethod
    def ConvNet2(tensorWidth, tensorHeight, tensorDepth, tb_verb=0):
        conv_net = input_data(shape=[None, tensorWidth, tensorHeight, tensorDepth])
        conv_net = conv_2d(conv_net,nb_filter=32,filter_size=5, activation='relu', bias=True)
        conv_net = batch_normalization(conv_net)
        conv_net = max_pool_2d(conv_net, 4)
        conv_net = dropout(conv_net, 0.5)
        conv_net = fully_connected(conv_net, 100, activation='relu')
        conv_net = dropout(conv_net, 0.5)


        conv_net = fully_connected(conv_net, 2, activation='softmax')
        conv_net = regression(conv_net, optimizer='sgd', learning_rate=0.01, loss='categorical_crossentropy',
                              )

        model = tflearn.DNN(conv_net, tensorboard_verbose=tb_verb)

        return model

    # M.G.
    @staticmethod
    def ConvNet3(tensorWidth, tensorHeight, tensorDepth, tb_verb=0):
        conv_net = input_data(shape=[None, tensorWidth, tensorHeight, tensorDepth])
        conv_net = conv_2d(conv_net,50,4,activation='tanh')
        conv_net = max_pool_2d(conv_net,5)
        conv_net = conv_2d(conv_net,50,4,activation='tanh')
        conv_net = max_pool_2d(conv_net,5)
        conv_net = conv_2d(conv_net,100,4,activation='tanh')
        conv_net = max_pool_2d(conv_net,5)
        conv_net = fully_connected(conv_net, 50, activation='tanh')

        conv_net = fully_connected(conv_net, 2, activation='softmax')

        conv_net = regression(conv_net, optimizer='sgd', learning_rate=0.01, loss='categorical_crossentropy',
                              )

        model = tflearn.DNN(conv_net, tensorboard_verbose=tb_verb)

        return model

    # A.J.
    @staticmethod
    def ConvNet4(tensorWidth, tensorHeight, tensorDepth, tb_verb=0):
        network = input_data(shape=[None, tensorWidth, tensorHeight, tensorDepth])
        network = conv_2d(network, 32, 3, activation='relu', regularizer="L2")
        network = max_pool_2d(network, 2)
        network = conv_2d(network, 64, 3, activation='relu', regularizer="L2")
        network = conv_2d(network, 64, 3, activation='relu', regularizer="L2")
        network = max_pool_2d(network, 2)
        network = fully_connected(network, 512, activation='relu')
        network = dropout(network, 0.5)
        network = fully_connected(network, 2, activation='softmax')
        network = regression(network,
                             optimizer='adam',
                             loss='categorical_crossentropy',
                             learning_rate=0.01)
        model = tflearn.DNN(network, tensorboard_verbose=tb_verb)
        return model

    # M.M.
    @staticmethod
    def ConvNet5(tensorWidth, tensorHeight, tensorDepth, tb_verb=0):
        conv_net = input_data(shape=[None, tensorWidth, tensorHeight, tensorDepth])
        conv_net = conv_2d(conv_net,
                          nb_filter=20,
                          filter_size=5,
                          activation='relu')
        conv_net = conv_2d(conv_net,
                               nb_filter=25,
                               filter_size=5,
                               activation='relu')

        conv_net = max_pool_2d(conv_net, 2)
        conv_net = fully_connected(conv_net, 100,
                                     activation='sigmoid')
        conv_net = fully_connected(conv_net, 2,
                                     activation='softmax')


        conv_net = fully_connected(conv_net, 2, activation='softmax')

        conv_net = regression(conv_net, optimizer='sgd', learning_rate=0.01, loss='categorical_crossentropy')

        model = tflearn.DNN(conv_net, tensorboard_verbose=tb_verb)

        return model

    # A.K.
    @staticmethod
    def ConvNet6(tensorWidth, tensorHeight, tensorDepth, tb_verb=0):
        conv_net = input_data(shape=[None, tensorWidth, tensorHeight, tensorDepth])

        conv_net = conv_2d(conv_net, nb_filter=10, filter_size=10,
                               activation='tanh')
        conv_net = max_pool_2d(conv_net, 4)
        conv_net = conv_2d(conv_net, nb_filter=30, filter_size=3,
                               activation='tanh')
        conv_net = max_pool_2d(conv_net, 2)
        conv_net = fully_connected(conv_net, 100, activation='sigmoid')

        conv_net = fully_connected(conv_net, 2, activation='softmax')

        conv_net = regression(conv_net, optimizer='sgd', learning_rate=0.01, loss='categorical_crossentropy')

        model = tflearn.DNN(conv_net, tensorboard_verbose=tb_verb)

        return model

    # A.G.
    @staticmethod
    def ConvNet7(tensorWidth, tensorHeight, tensorDepth, tb_verb=0):
        conv_net = input_data(shape=[None, tensorWidth, tensorHeight, tensorDepth])

        conv_net = conv_2d(conv_net, nb_filter=20, filter_size=5, activation='relu')
        conv_net = max_pool_2d(conv_net, 2)
        conv_net = conv_2d(conv_net, nb_filter=15, filter_size=5, activation='relu')
        conv_net = max_pool_2d(conv_net, 2)
        conv_net = fully_connected(conv_net, 15, activation='relu')

        conv_net = fully_connected(conv_net, 2, activation='sigmoid')
        conv_net = regression(conv_net, optimizer='sgd', learning_rate=0.01, loss='categorical_crossentropy',
                              )

        model = tflearn.DNN(conv_net, tensorboard_verbose=tb_verb)

        return model

    # R.W.
    @staticmethod
    def ConvNet8(tensorWidth, tensorHeight, tensorDepth, tb_verb=0):
        conv_net = input_data(shape=[None, tensorWidth, tensorHeight, tensorDepth])

        conv_net = conv_2d(conv_net, nb_filter=20,
                             filter_size=7, activation='relu',
                             name='conv_layer_1')
        conv_net = max_pool_2d(conv_net, 3)
        conv_net = conv_2d(conv_net, nb_filter=20,
                             filter_size=5, activation='relu')
        conv_net = max_pool_2d(conv_net, 2)
        conv_net = fully_connected(conv_net, 100,
                                     activation='sigmoid')


        conv_net = fully_connected(conv_net, 2, activation='sigmoid')
        conv_net = regression(conv_net, optimizer='sgd', learning_rate=0.01, loss='categorical_crossentropy',
                              )

        model = tflearn.DNN(conv_net, tensorboard_verbose=tb_verb)

        return model

    # C.K.
    @staticmethod
    def ConvNet9(tensorWidth, tensorHeight, tensorDepth, tb_verb=0):
        conv_net = input_data(shape=[None, tensorWidth, tensorHeight, tensorDepth])

        conv_net = conv_2d(conv_net, nb_filter=20, filter_size=5, activation="relu")
        conv_net = max_pool_2d(conv_net, 2)
        conv_net = conv_2d(conv_net, nb_filter=50, filter_size=5, activation="relu")
        conv_net = max_pool_2d(conv_net, 2)
        conv_net = conv_2d(conv_net, nb_filter=30, filter_size=5, activation="relu")
        conv_net = max_pool_2d(conv_net, 2)
        conv_net = fully_connected(conv_net, 140, activation="relu")
        conv_net = fully_connected(conv_net, 80, activation="sigmoid")
        conv_net = fully_connected(conv_net, 2, activation="sigmoid")

        conv_net = regression(conv_net, optimizer='adam', learning_rate=0.01, loss='categorical_crossentropy',
                              )

        model = tflearn.DNN(conv_net, tensorboard_verbose=tb_verb)

        return model

    # V.S.
    @staticmethod
    def ConvNet10(tensorWidth, tensorHeight, tensorDepth, tb_verb=0):
        conv_net = input_data(shape=[None, tensorWidth, tensorHeight, tensorDepth])
        conv_net = conv_2d(conv_net,
                               nb_filter=32,
                               filter_size=3,
                               activation='relu')
        conv_net = conv_2d(conv_net,
                               nb_filter=64,
                               filter_size=3,
                               activation='relu')
        conv_net = max_pool_2d(conv_net, 2)
        conv_net = fully_connected(conv_net, 256, activation='sigmoid')
        conv_net = fully_connected(conv_net, 2, activation='sigmoid')
        conv_net = regression(conv_net, optimizer='sgd', learning_rate=0.01, loss='categorical_crossentropy',
                              )
        model = tflearn.DNN(conv_net, tensorboard_verbose=tb_verb)

        return model
