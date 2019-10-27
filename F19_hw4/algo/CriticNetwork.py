import tensorflow as tf
from tensorflow.keras.layers import Dense, Input, concatenate
from tensorflow.keras.optimizers import Adam

HIDDEN1_UNITS = 400
HIDDEN2_UNITS = 400


def create_critic_network(state_size, action_size, learning_rate):
    """Creates a critic network.

    Args:
        state_size: (int) size of the input.
        action_size: (int) size of the action.
        learning_rate: (float) learning rate for the critic.
    Returns:
        model: an instance of tf.keras.Model.
        state_input: a tf.placeholder for the batched state.
        action_input: a tf.placeholder for the batched action.
    """
    state_input = Input(shape=[state_size])
    action_input = Input(shape=[action_size])
    cat_input = concatenate([state_input, action_input])
    h0 = Dense(HIDDEN1_UNITS, activation='relu')(cat_input)
    h1 = Dense(HIDDEN2_UNITS, activation='relu')(h0)
    value = Dense(1)(h1)
    model = tf.keras.Model(inputs=[state_input, action_input], outputs=value)
    model.compile(loss="mse", optimizer=Adam(lr=learning_rate))
    return model, state_input, action_input


class CriticNetwork(object):
    def __init__(self, sess, state_size, action_size, batch_size,
                 tau, learning_rate):
        """Initialize the CriticNetwork.
        This class internally stores both the critic and the target critic
        nets. It also handles computation of the gradients and target updates.

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
        tf.keras.backend.set_session(self.sess)

        self.model, self.state_input, self.action_input = create_critic_network(state_size, action_size, learning_rate)
        self.model_target, _, _ = create_critic_network(state_size, action_size, learning_rate)
        self.get_action_gradients = tf.gradients(self.model.output, self.action_input)

        self.sess.run(tf.initialize_all_variables())

        # sync initial weights between model and target
        model_weights = self.model.get_weights()
        self.model_target.set_weights(model_weights)

    def gradients(self, states, actions):
        """Computes dQ(s, a) / da.
        Note that tf.gradients returns a list storing a single gradient tensor,
        so we return that gradient, rather than the singleton list.

        Args:
            states: a batched numpy array storing the state.
            actions: a batched numpy array storing the actions.
        Returns:
            grads: a batched numpy array storing the gradients.
        """
        return self.sess.run(
            self.get_action_gradients,
            feed_dict={
                self.state_input: states,
                self.action_input: actions
            }
        )

    def update_target(self):
        """Updates the target net using an update rate of tau."""
        model_weights = self.model.get_weights()
        target_weights = self.model_target.get_weights()
        target_weights = [self.tau * w + (1 - self.tau) * w_t for w, w_t in zip(model_weights, target_weights)]
        self.model_target.set_weights(target_weights)
