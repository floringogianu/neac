""" RL-Routines
"""


class Episode:
    """ An iterator accepting an environment and a policy, that returns
    experience tuples.
    """

    def __init__(self, env, policy, _state=None):
        self.env = env
        self.policy = policy
        self.__R = 0
        self.__step_cnt = -1
        if _state is None:
            self.__state, self.__done = self.env.reset(), False
        else:
            self.__state, self.__done = _state, False

    def __iter__(self):
        return self

    def __next__(self):
        if self.__done:
            raise StopIteration

        _pi = self.policy.act(self.__state)
        _state = self.__state.clone()
        self.__state, reward, self.__done, _ = self.env.step(_pi.action)

        self.__R += reward
        self.__step_cnt += 1
        return (_state, _pi, reward, self.__state, self.__done)

    def __enter__(self):
        return self

    def __exit__(self, *args):
        print("Episode done")

    @property
    def Rt(self):
        """ Return the expected return.
        """
        return self.__R

    @property
    def steps(self):
        """ Return steps taken in the environment.
        """
        return self.__step_cnt
