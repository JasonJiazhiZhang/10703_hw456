import tensorflow as tf
from tensorflow.keras.layers import Dense, Input

from tensorflow.keras import Model

HIDDEN1_UNITS = 400
HIDDEN2_UNITS = 400


def create_actor_network(state_size, action_size):
    """Creates an actor network.

    Args:
        state_size: (int) size of the input.
        action_size: (int) size of the action.
    Returns:
        model: an instance of tf.keras.Model.
        state_input: a tf.placeholder for the batched state.
    """
    state_input = Input(shape=[state_size])
    h0 = Dense(HIDDEN1_UNITS, activation='relu', kernel_initializer='he_normal',)(state_input)
    h1 = Dense(HIDDEN2_UNITS, activation='relu', kernel_initializer='he_normal',)(h0)
    h2 = Dense(action_size, activation='tanh', kernel_initializer='he_normal',)(h1)
    model = Model(input=state_input, output=h2)
    return model, state_input


class ActorNetwork(object):
    def __init__(self, sess, state_size, action_size, batch_size,
                 tau, learning_rate):
        """Initialize the ActorNetwork.
        This class internally stores both the actor and the target actor nets.
        It also handles training the actor and updating the target net.

        Args:
            sess: A Tensorflow session to use.
            state_size: (int) size of the input.
            action_size: (int) size of the action.
            batch_size: (int) the number of elements in each batch.
            tau: (float) the target net update rate.
            learning_rate: (float) learning rate for the critic.
        """
        self.sess = sess
        self.state_size = state_size
        self.action_size = action_size
        self.batch_size = batch_size
        self.tau = tau
        self.learning_rate = learning_rate

        self.model, self.states = create_actor_network(state_size, action_size)
        self.model_target, _ = create_actor_network(state_size, action_size)
        self.action_grads = tf.placeholder(tf.float32, shape=(None, action_size))

        self.param_grads = tf.gradients(self.model.output, self.model.trainnable_weights, self.action_grads)
        grads = zip(-self.param_grads, self.model.trainnable_weights)
        self.optimize_actor = tf.train.AdamOptimizer(self.learning_rate).apply_gradients(grads)


        self.sess.run(tf.initialize_all_variables())

        # sync initial weights between model and target
        model_weights = self.model.get_weights()
        self.model_target.set_weights(model_weights)


    def train(self, states, action_grads):
        """Updates the actor by applying dQ(s, a) / da.

        Args:
            states: a batched numpy array storing the state.
            actions: a batched numpy array storing the actions.
            action_grads: a batched numpy array storing the
                gradients dQ(s, a) / da.
        """
        self.sess.run(
            self.optimize_actor,
            feed_dict={
                self.states: states,
                self.action_grads:action_grads,
            }
        )

    def update_target(self):
        """Updates the target net using an update rate of tau."""
        model_weights = self.model.get_weights()
        target_weights = self.model_target.get_weights()
        target_weights = [self.tau * w + (1 - self.tau) * w_t for w, w_t in zip(model_weights, target_weights)]
        self.model_target.set_weights(target_weights)

