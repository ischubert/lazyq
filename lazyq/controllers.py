"""
Definition of the controller class for Q learning with
lazy action model.
"""
import pickle
import time
from os import path, mkdir
import tqdm
import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras
from cvxopt.solvers import qp
from cvxopt import matrix as cvxopt_matrix


class DnnRegressor2DPlus1D():
    """
    DNN regression model using 2D state input from
    [-floor_size,floor_size]x[-floor_size,floor_size] and 1D goal input (angle)
    """

    def __init__(
            self,
            floor_size,
            target_update_interval,
            n_epochs,
            keras_model=None
    ):
        self.floor_size = floor_size
        self.target_update_interval = target_update_interval
        self.n_epochs = n_epochs

        self.norm_factors = [
            1/floor_size,    # state: x-position
            1/floor_size,    # state: y-position
            1,               # goal: intended direction x
            1                # goal: intended direction y
        ]

        if keras_model is None:
            self.model = self.build_model()
            self.target_model = self.build_model()
        else:
            self.model = keras_model
            self.target_model = keras_model.copy()

        self.fit_counter = 0

    def build_model(self):
        """
        Build underlying Keras model
        """
        x_in = keras.layers.Input((4,))

        # Normalize stuff
        f_x = keras.layers.Lambda(
            lambda x: x * self.norm_factors
        )(x_in)

        f_x = keras.layers.Dense(20, activation='relu')(f_x)
#         f_x = keras.layers.Dropout(0.4)(f_x)
        f_x = keras.layers.BatchNormalization()(f_x)
        f_x = keras.layers.Dense(30, activation='relu')(f_x)
#         f_x = keras.layers.Dropout(0.4)(f_x)
        f_x = keras.layers.BatchNormalization()(f_x)
        f_x = keras.layers.Dense(50, activation='relu')(f_x)
        f_x = keras.layers.BatchNormalization()(f_x)
        f_x = keras.layers.Dense(40, activation='relu')(f_x)
        f_x = keras.layers.BatchNormalization()(f_x)
        f_x = keras.layers.Dense(50, activation='relu')(f_x)
        f_x = keras.layers.BatchNormalization()(f_x)
        f_x = keras.layers.Dense(40, activation='relu')(f_x)
        f_x = keras.layers.BatchNormalization()(f_x)
        f_x = keras.layers.Dense(30, activation='relu')(f_x)
#         f_x = keras.layers.BatchNormalization()(f_x)
        f_x = keras.layers.Dense(20, activation='relu')(f_x)
#         f_x = keras.layers.BatchNormalization()(f_x)
        y_out = keras.layers.Dense(1)(f_x)

        new_model = keras.models.Model(
            inputs=x_in,
            outputs=y_out
        )
        new_model.compile(
            optimizer='Adam',
            loss='mse'
        )

        return new_model

    def fit(self, x_data, y_data, verbose=0):
        """
        Fit the value function to data
        """
        # fit the model
        self.fit_counter += 1
        self.model.fit(x_data, y_data, verbose=verbose, epochs=self.n_epochs)

        # and possibly update the target_model
        if self.fit_counter % self.target_update_interval == 0:
            self.target_model.set_weights(
                self.model.get_weights()
            )
            # if target network has been updated, set flag so that the
            # controller updates the data
            return True
        return False

    def predict(self, x_data):
        """
        Predict instances of the value function
        """
        # predict using the target model
        return np.array(self.target_model.predict(x_data))

    def visualize(self, goal, state_grid):
        """utility function to plot the V function under goal 'goal'"""
        plt.imshow(
            self.predict(
                np.concatenate(
                    (
                        state_grid,
                        np.repeat(
                            goal.reshape((1, 2)),
                            len(state_grid),
                            axis=0
                        ),
                    ),
                    axis=-1
                )
            ).reshape(int(np.sqrt(len(state_grid))), int(np.sqrt(len(state_grid))))[:, ::-1].T
        )
        plt.colorbar()
        plt.xticks(
            np.linspace(
                0,
                int(np.sqrt(len(state_grid))),
                3
            ).astype(int),
            np.linspace(
                min(state_grid[:, 0]),
                max(state_grid[:, 0]),
                3
            )
        )
        plt.yticks(
            np.linspace(
                0,
                int(np.sqrt(len(state_grid))),
                3
            ).astype(int)[::-1],
            np.linspace(
                min(state_grid[:, 1]),
                max(state_grid[:, 1]),
                3
            )
        )
        plt.xlabel('x1')
        plt.ylabel('x2')

        plt.show()


class Controller():
    """
    The controller maintains the data set and the value function estimator,
    and contains methods for performing training iterations as well as
    predicting optimal actions using the value function and the lazy
    action model
    """

    def __init__(
            self,
            v_functions,
            local_model_regularization,
            discount,
            k_nearest,
            boundaries,
            n_v_function_update_iterations,
            environment,
            scaleup_factor,
            # how many of the elements with too many neighbours should actually be thrown out
            prune_ratio=0.,
            init_length=3,
            scratch_dir=None,
            record_plots=False,
            scale_logger=False
    ):
        self.v_functions = v_functions
        self.local_model_regularization = local_model_regularization
        self.discount = discount
        self.k_nearest = k_nearest
        self.boundaries = boundaries
        self.n_v_function_update_iterations = n_v_function_update_iterations
        self.environment = environment
        self.scaleup_factor = scaleup_factor
        self.prune_ratio = prune_ratio
        self.init_length = init_length
        self.scratch_dir = scratch_dir
        self.record_plots = record_plots
        self.scale_logger = scale_logger

        self.mean_kept = []
        if self.scale_logger:
            self.scales = []

        # Initialize raw data
        angles = 2*np.pi*np.random.rand(self.init_length*self.k_nearest+1)
        self.goals_all = [
            np.stack((
                np.cos(angles),
                np.sin(angles)
            ), axis=-1)
            for _ in range(len(self.v_functions))
        ]
        self.states_all = [
            np.random.rand(self.init_length*self.k_nearest+1, 2)
            for v_function in range(len(self.v_functions))
        ]
        self.actions_all = [
            np.random.rand(self.init_length*self.k_nearest+1, 2)
            for _ in range(len(self.v_functions))
        ]
        self.rewards_all = [
            np.zeros(self.init_length*self.k_nearest+1)
            for _ in range(len(self.v_functions))
        ]
        self.next_states_all = [
            np.random.rand(self.init_length*self.k_nearest+1, 2)
            for _ in range(len(self.v_functions))
        ]

        # initialize processed data
        # This only depends on the current data set
        self.k_smallest_inds_all = [
            None
            for _ in range(len(self.v_functions))
        ]
        # This only depends on my current estimate of the V function
        self.r_plus_gamma_v_all = [
            None
            for _ in range(len(self.v_functions))
        ]
        # This depends both on the current data set
        # and on the current estimate of the V function
        self.targets_all = [
            None
            for _ in range(len(self.v_functions))
        ]

        # Flags that show if the initial data has been kicked out already
        self.already_pruned = [
            False
            for _ in range(len(self.v_functions))
        ]

        for train_signal in range(len(self.v_functions)):
            self.update(train_signal)

    def get_lazy_policy_action(
            self,
            r_plus_gamma_v_nexts,
            actions,
    ):
        """
        Get optimal action using the lazy action model
        """
        actions_augmented = np.concatenate((
            np.ones((len(actions), 1)),
            actions
        ), axis=-1)

        sol = qp(
            cvxopt_matrix(
                np.sum(
                    actions_augmented[
                        :, :, None
                    ] * actions_augmented[
                        :, None, :
                    ],
                    axis=0
                ) + self.local_model_regularization * np.eye(3)
            ),
            cvxopt_matrix(np.sum(
                -(r_plus_gamma_v_nexts[:, None] * actions_augmented[:, :]),
                axis=0
            ))
        )

        if 'optimal' not in sol['status']:
            raise Exception('Optimal solution not found')

        alphabeta = np.array(sol['x']).reshape(-1)

        return self.environment.action_length/np.linalg.norm(alphabeta[1:]) * alphabeta[1:]

    def get_lazy_model_value(
            self,
            r_plus_gamma_v_nexts,
            actions,
    ):
        """
        Predict value of V-function, performing the maximum over the available actions using
        the lazy action model
        """
        actions_augmented = np.concatenate((
            np.ones((len(actions), 1)),
            actions
        ), axis=-1)

        sol = qp(
            cvxopt_matrix(
                np.sum(
                    actions_augmented[
                        :, :, None
                    ] * actions_augmented[
                        :, None, :
                    ],
                    axis=0
                ) + self.local_model_regularization * np.eye(3)
            ),
            cvxopt_matrix(np.sum(
                -(r_plus_gamma_v_nexts[:, None] * actions_augmented[:, :]),
                axis=0
            ))
        )

        if 'optimal' not in sol['status']:
            raise Exception('Optimal solution not found')

        alphabeta = np.array(sol['x']).reshape(-1)

        # Instability countermeasure: The target cant be higher than the maximum
        # r_plus_gamma_V_next in the vicinity
        return min([
            alphabeta[0] + self.environment.action_length *
            np.linalg.norm(alphabeta[1:]),
            np.max(r_plus_gamma_v_nexts)
        ])

    def append_new_data_point(
            self,
            rollout_policy_inds,
            goal,
            state,
            action,
            reward,
            next_state
    ):
        """
        Add new data point to the value functions
        """
        for train_signal in rollout_policy_inds:
            # Append raw data ONLY
            self.goals_all[train_signal] = np.append(
                self.goals_all[train_signal],
                goal.reshape((1, len(goal))),
                axis=0
            )
            self.states_all[train_signal] = np.append(
                self.states_all[train_signal],
                state.reshape((1, len(state))),
                axis=0
            )
            self.actions_all[train_signal] = np.append(
                self.actions_all[train_signal],
                action.reshape((1, len(action))),
                axis=0
            )
            self.rewards_all[train_signal] = np.append(
                self.rewards_all[train_signal],
                np.array([reward]),
                axis=0
            )
            self.next_states_all[train_signal] = np.append(
                self.next_states_all[train_signal],
                next_state.reshape((1, len(next_state))),
                axis=0
            )

            if not self.already_pruned[train_signal]:
                if len(self.states_all[train_signal]) > self.init_length*self.k_nearest + 3:
                    # throw our placeholders from raw data
                    print(
                        'throw out initial placeholder raw data for no.', train_signal)
                    self.goals_all[train_signal] = self.goals_all[train_signal][
                        self.init_length*self.k_nearest + 2:
                    ]
                    self.states_all[train_signal] = self.states_all[train_signal][
                        self.init_length*self.k_nearest + 2:
                    ]
                    self.actions_all[train_signal] = self.actions_all[train_signal][
                        self.init_length*self.k_nearest + 2:
                    ]
                    self.rewards_all[train_signal] = self.rewards_all[train_signal][
                        self.init_length*self.k_nearest + 2:
                    ]
                    self.next_states_all[train_signal] = self.next_states_all[train_signal][
                        self.init_length*self.k_nearest + 2:
                    ]
                    self.already_pruned[train_signal] = True

    def rapid_near_neighbours_scale_up(
            self,
            rollout_policy_ind,
            state,
            goal
    ):
        """
        Find neighbors in a neighborhood of gradually increased size
        """
        scale = 1
        for _ in range(20):
            neighbours = self.environment.find_near_neighbours(
                self.states_all[rollout_policy_ind],
                self.goals_all[rollout_policy_ind],
                state,
                goal,
                scale
            )
#             print(neighbours.shape)
            if len(neighbours) > self.k_nearest:
                # If there is a sufficient number
                # of points that has been found,
                # choose randomly without replace
                # from them.
                if self.scale_logger:
                    self.scales.append(scale)
                return [
                    np.random.choice(
                        neighbours,
                        size=self.k_nearest,
                        replace=False
                    ),
                    scale
                ]
            # else: scale up and repeat
            scale = scale*self.scaleup_factor
        # Raise exception if not enough found
        raise Exception(
            'Not enough near neighbours found after scaling to ' + str(scale))

    def get_individual_action(self, rollout_policy_ind, state, goal):
        """get action from policy no. rollout_policy_ind"""

        k_neighbours_inds = self.rapid_near_neighbours_scale_up(
            rollout_policy_ind,
            state,
            goal
        )[0]

        return self.get_lazy_policy_action(
            self.r_plus_gamma_v_all[rollout_policy_ind][k_neighbours_inds].reshape(
                -1),
            self.actions_all[rollout_policy_ind][k_neighbours_inds]
        )

    def get_action(self, state, goal):
        """Get averaged action from all policies"""
        average = np.mean(
            np.stack(
                [
                    self.get_individual_action(pol_ind, state, goal)
                    for pol_ind in range(len(self.v_functions))
                ],
                axis=0
            ),
            axis=0
        )

        return self.environment.action_length*average/np.linalg.norm(average)

    def update_targets_only(
            self,
            train_signal
    ):
        """
        Update all targets. This function assumes
        up-to-date nearest neighbours and up-to-date V function values"""

        self.targets_all[train_signal] = np.array([
            self.get_lazy_model_value(
                # Here I assume that the V function values are up-to-date
                self.r_plus_gamma_v_all[train_signal][k_smallest_inds],
                self.actions_all[train_signal][k_smallest_inds]
            )
            # Here I assume that the indices of all neighbours are up-to-date
            for k_smallest_inds in self.k_smallest_inds_all[train_signal]
        ])

        # data augmentation: set targets to 0 outside of the interesting domain
        self.environment.get_augmented_targets(
            self.states_all[train_signal],
            self.targets_all[train_signal]
        )

        # The rollout ends once a reward is given. Thus, the value of the value function is
        # the reward itself whereever rewards are given
        reward_was_given_mask = (
            np.abs(self.rewards_all[train_signal]) > 1e-10)
        self.targets_all[train_signal][
            reward_was_given_mask
        ] = self.rewards_all[train_signal][
            reward_was_given_mask
        ]
        # There are user-defined lower and upper bounds on the value function as well
        self.targets_all[train_signal][
            self.targets_all[train_signal] < self.boundaries[0]
        ] = self.boundaries[0]
        self.targets_all[train_signal][
            self.targets_all[train_signal] > self.boundaries[1]
        ] = self.boundaries[1]

        _ = plt.hist(self.targets_all[train_signal], bins=100, log=True)

        if self.record_plots:
            if not path.exists(self.scratch_dir + '/plots/'):
                mkdir(self.scratch_dir + '/plots/')
            plt.savefig(self.scratch_dir + '/plots/' + str(
                time.time()
            ) + '.png')

    def update_r_plus_gamma_v(self, train_signal):
        """Update the V values of all data points. These are only
        dependent on the weights of the target model"""
        self.r_plus_gamma_v_all[train_signal] = self.rewards_all[
            train_signal
        ] + self.discount * self.v_functions[
            train_signal
        ].predict(
            np.concatenate((
                self.next_states_all[train_signal],
                self.goals_all[train_signal]
            ), axis=-1)
        ).reshape(-1)

    def update_k_smallest_inds_and_calculate_pruning(self, train_signal):
        """Update the indices of the nearest neighbours
        this depends on the data set only (in the sense that
        for every point in the set, its nearest neighbours
        depend on the entire data set). After that delete
        some points in areas with high overlap"""

        self.k_smallest_inds_all[train_signal] = []

        keep_in = []
        for state, goal in zip(
                self.states_all[train_signal],
                self.goals_all[train_signal]
        ):
            inds, scale = self.rapid_near_neighbours_scale_up(
                train_signal,
                state,
                goal
            )
            self.k_smallest_inds_all[train_signal].append(
                inds
            )
            keep_in.append(
                scale != 1
            )

        keep_in = np.array(keep_in)

        # only throw out self.prune_ratio of all elements with too many neighbours
        keep_in = np.logical_or(
            keep_in,
            np.random.rand(len(keep_in)) < 1-self.prune_ratio
        )

        self.mean_kept.append(np.mean(keep_in))

        return keep_in

    def prune(
            self,
            train_signal,
            keep_in
    ):
        """
        Thin out dataset in areas of high density
        """
        self.goals_all[train_signal] = self.goals_all[train_signal][keep_in]
        self.states_all[train_signal] = self.states_all[train_signal][keep_in]
        self.actions_all[train_signal] = self.actions_all[train_signal][keep_in]
        self.rewards_all[train_signal] = self.rewards_all[train_signal][keep_in]
        self.next_states_all[train_signal] = self.next_states_all[train_signal][keep_in]

        print(
            len(self.goals_all[train_signal]),
            len(self.states_all[train_signal]),
            len(self.actions_all[train_signal]),
            len(self.rewards_all[train_signal]),
            len(self.next_states_all[train_signal])
        )

    def update(self, train_signal):
        """This function is supposed to be a full from-scratch
        extraction of a value function from a static data set"""
        print('TRAIN_SIGNAL NO.', train_signal)

        # Before the self-consistent V function iteration starts,
        # all the stuff that only depends on the static raw data set is pre-calcluated
        # 1. The X values
        print('Update X values')
        x_values = np.concatenate(
            (
                self.states_all[train_signal],
                self.goals_all[train_signal]
            ),
            axis=-1
        )
        # 2. k_smallest_inds_all also only depends on static raw data
        print('Update k_smallest_inds_all and calculate pruning mask for later...')
        keep_in = self.update_k_smallest_inds_and_calculate_pruning(
            train_signal)

        # Once the static data is up-to-date, we learn the value function
        # self-consistently on the static data set.
        print('Self-consistent V-function iterations...')
        # Start loop with initializing
        target_weights_updated = True
        for _ in tqdm.tqdm(range(self.n_v_function_update_iterations)):
            if target_weights_updated:
                # If the weights of the target network have been updated,
                # self.r_plus_gamma_v_all has to be updated as well.
                print('Update r_plus_gamma_V_all')
                self.update_r_plus_gamma_v(train_signal)

                # Since self.targets_all depend on self.r_plus_gamma_v_all,
                # self.targets_all has to be updated as well
                print('Update targets_all')
                self.update_targets_only(
                    train_signal
                )

            # Fit network on X and Y
            print('Fit network...')
            target_weights_updated = self.v_functions[train_signal].fit(
                x_values,
                self.targets_all[train_signal]
            )
        # Apply pruning mask

        print('Prune data set')
        self.prune(train_signal, keep_in)
        print('Kept', self.mean_kept[-1], 'of original data')

    def save(self, folder_name):
        """
        Save the controller object
        """
        # make sure not to overwrite anything
        assert not path.exists(folder_name)
        mkdir(folder_name)
        # save v functions
        for ind, v_function in enumerate(self.v_functions):
            v_function.target_model.save_weights(
                folder_name + '/v_function_' + str(ind) + '.hd5')

        # save the controller's collected data
        data = {
            'goals_all': self.goals_all,
            'states_all': self.states_all,
            'actions_all': self.actions_all,
            'rewards_all': self.rewards_all,
            'next_states_all': self.next_states_all
        }
        with open(folder_name + '/data.pickle', 'wb') as file:
            pickle.dump(data, file)

    def load(self, folder_name, light=False, only_values=False):
        """
        Load the controller object
        """

        # make sure data exists
        if isinstance(folder_name, str):
            assert path.exists(folder_name)
            v_func_paths = [folder_name + '/v_function_' + str(
                ind
            ) + '.hd5' for ind in range(len(self.v_functions))]
            data_dirs = folder_name + '/data.pickle'
            data_inds = range(len(self.v_functions))
        else:
            for name in folder_name:
                assert path.exists(name)
            v_func_paths = [name + '/v_function_' + str(
                0
            ) + '.hd5' for name in folder_name]
            data_dirs = [name + '/data.pickle' for name in folder_name]
            data_inds = [0 for name in folder_name]

        # load V functions
        for v_function, v_func_path in zip(self.v_functions, v_func_paths):
            v_function.target_model.load_weights(v_func_path)

        # load the controller's collected data. Not efficient, but this is
        # not significant usually
        for ind, [data_dir, data_ind] in enumerate(zip(data_dirs, data_inds)):
            with open(data_dir, 'rb') as file:
                data = pickle.load(file)
                self.goals_all[ind] = data['goals_all'][data_ind]
                self.states_all[ind] = data['states_all'][data_ind]
                self.actions_all[ind] = data['actions_all'][data_ind]
                self.rewards_all[ind] = data['rewards_all'][data_ind]
                self.next_states_all[ind] = data['next_states_all'][data_ind]

        if not light:
            for train_signal in tqdm.tqdm(range(len(self.v_functions))):
                self.update_r_plus_gamma_v(train_signal)
                if not only_values:
                    self.update_k_smallest_inds_and_calculate_pruning(train_signal)
                    self.update_targets_only(train_signal)
