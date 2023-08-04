import argparse
import numpy as np

def load_q_table(q_table_file):
    return np.load(q_table_file)

def get_action(state, Q):
    state_index = state[0] * 8 * 3 + state[1] * 3 + state[2]
    return np.argmax(Q[state_index, :])

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Get the optimal action for a given state.')
    parser.add_argument('-s', '--state', type=int, nargs=3, required=True, help='The current state as three integers.')
    args = parser.parse_args()

    state = tuple(args.state)
    Q = load_q_table('q_table.npy')
    action = get_action(state, Q)
    print(action)
