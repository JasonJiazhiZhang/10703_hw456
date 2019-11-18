import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
from keras.layers import Dense, Flatten, Input, Concatenate, Lambda, Activation
from keras.models import Model
from keras.regularizers import l2
import tensorflow.compat.v1.keras.backend as K
import numpy as np
from util import ZFilter

HIDDEN1_UNITS = 400
HIDDEN2_UNITS = 400
HIDDEN3_UNITS = 400

class PENN:
    """
    (P)robabilistic (E)nsemble of (N)eural (N)etworks
    """

    def __init__(self, num_nets, state_dim, action_dim, learning_rate):
        """
        :param num_nets: number of networks in the ensemble
        :param state_dim: state dimension
        :param action_dim: action dimension
        :param learning_rate:
        """
        self.sess = tf.Session()
        self.num_nets = num_nets
        self.state_dim = state_dim
        self.action_dim = action_dim
        K.set_session(self.sess)

        self.learning_rate = learning_rate
        self.models = []
        self.inputs_placeholder = []
        self.targets_placeholder = []
        self.optimizers = []
        self.means = []
        self.logvars = []
        self.losses = []
        self.rmses = []
        # Log variance bounds
        self.max_logvar = tf.Variable(-3 * np.ones([1, self.state_dim]), dtype=tf.float32)
        self.min_logvar = tf.Variable(-7 * np.ones([1, self.state_dim]), dtype=tf.float32)

        # TODO write your code here
        # Create and initialize your model
        for i in range(self.num_nets):
            model= self.create_network()
            self.models.append(model)

            out_mean, out_logvar = self.get_output(self.models[i].output)
            self.means.append(out_mean)
            self.logvars.append(out_logvar)

            self.targets_placeholder.append(tf.placeholder(tf.float32, shape=[None, self.state_dim]))
            loss, rmse = self.compile_loss(self.targets_placeholder[i], self.means[i], self.logvars[i])
            self.losses.append(loss)
            self.rmses.append(rmse)
            
            # gradients = tf.gradients(self.losses[i], self.model[i].trainable_weights)
            # optimizer = tf.train.AdamOptimizer(self.learning_rate).apply_gradients(zip(gradients, self.model[i].trainable_weights))
            optimizer = tf.train.AdamOptimizer(self.learning_rate).minimize(self.losses[i])
            self.optimizers.append(optimizer)
        
        self.sess.run(tf.initialize_all_variables())
   

    def get_output(self, output):
        """
        Argument:
          output: tf variable representing the output of the keras models, i.e., model.output
        Return:
          mean and log variance tf tensors
        Note that you will still have to call sess.run on these tensors in order to get the
        actual output.
        """
        mean = output[:, 0:self.state_dim]
        raw_v = output[:, self.state_dim:]
        logvar = self.max_logvar - tf.nn.softplus(self.max_logvar - raw_v)
        logvar = self.min_logvar + tf.nn.softplus(logvar - self.min_logvar)
        return mean, logvar

    def create_network(self):
        I = Input(shape=[self.state_dim + self.action_dim], name='input')
        h1 = Dense(HIDDEN1_UNITS, activation='relu', kernel_regularizer=l2(0.0001))(I)
        h2 = Dense(HIDDEN2_UNITS, activation='relu', kernel_regularizer=l2(0.0001))(h1)
        h3 = Dense(HIDDEN3_UNITS, activation='relu', kernel_regularizer=l2(0.0001))(h2)
        O = Dense(2 * self.state_dim, activation='linear', kernel_regularizer=l2(0.0001))(h3)
        model = Model(input=I, output=O)
        return model 

    def compile_loss(self, target_state, out_mean, out_logvar):
        inv_var = tf.exp(-out_logvar)
        mean_loss = tf.reduce_mean(tf.reduce_sum(tf.square(out_mean - target_state) * inv_var, axis=-1))
        var_loss = tf.reduce_mean(tf.reduce_sum(out_logvar, axis=-1))
        rmse = tf.sqrt(tf.reduce_sum(tf.reduce_mean(tf.square(out_mean - target_state), axis=0)))
        #rmse = tf.sqrt(tf.reduce_sum(tf.square(out_mean - target_state)))
        return mean_loss + var_loss, rmse
        
    def train(self, inputs, targets, batch_size=128, epochs=5):
        """
        Arguments:
          inputs: state and action inputs.  Assumes that inputs are standardized.
          targets: resulting states
        """
        # TODO: write your code here
        # losses = []
        # rmses = []
        # print(len(inputs))
        # #print(targets)
        # for i in range(self.num_nets):
        #     _, loss, rmse = self.sess.run(
        #         [self.optimizers[i],
        #          self.losses, self.rmses],
        #         feed_dict={
        #             self.true_state: targets[i],
        #             self.I: inputs[i]
        #         }
        #     )
        #     losses.append(loss)
        #     rmses.append(rmse)
        # return losses, rmses

        losses = [[] for i in range(self.num_nets)]
        rmses = [[] for i in range(self.num_nets)] 
		
        for e in range(epochs):
            for i in range(self.num_nets):
                index = np.random.choice(len(inputs), size=len(inputs), replace=True)
                batch_losses = []
                batch_rmses = []
                counter = 0
                for batch_id in range(0, len(inputs), batch_size):
                    batch_index = index[batch_id : min(batch_id + batch_size, len(inputs))]
                    batch_inputs = inputs[batch_index]
                    batch_targets = targets[batch_index]

                    _, loss, rmse, mean, logvar = self.sess.run(
                        [self.optimizers[i], self.losses[i], self.rmses[i], self.means[i], self.logvars[i]],
                        feed_dict = {self.models[i].input: batch_inputs, self.targets_placeholder[i]: batch_targets}
                    )
                    # print("mean", mean)
                    # print("logvar", logvar)
                    # print("loss", loss)
                    # print("rmse", rmse)
                    batch_losses.append(loss)
                    batch_rmses.append(rmse)

                losses[i].append(np.mean(batch_losses))
                rmses[i].append(np.mean(batch_rmses))

                print("epoch {}, loss {}, rmse {}".format(e, losses[i][-1], rmses[i][-1]))
        return np.array(losses), np.array(rmses)


    # TODO: Write any helper functions that you need
