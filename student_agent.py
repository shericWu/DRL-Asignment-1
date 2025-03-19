# Remember to adjust your student ID in meta.xml
import numpy as np
import pickle
import random
import gym

with open ("q_table.pkl", "rb") as f:
    q_table = pickle.load(f)

def get_state(obs, prev_action, target_pos, got_passenger):
    taxi_pos = (obs[0], obs[1])
    clear = [obs[11], obs[10], obs[12], obs[13]]
    opposite_dir = [1, 0, 3, 2]
    if 0 <= prev_action and prev_action <= 3:
        clear[opposite_dir[prev_action]] += 1
    on_target = 1 if (target_pos == taxi_pos) and (obs[14 + got_passenger] == 1) else 0
    return (int(target_pos[0] < taxi_pos[0]), int(target_pos[1] < taxi_pos[1]), clear[0], clear[1], clear[2], clear[3], got_passenger, on_target)

first_time = True
prev_action = -1
target = 0
target_pos = None
dest_pos = None
got_passenger = 0
in_station = True

def get_action(obs):
    
    # TODO: Train your own agent
    # HINT: If you're using a Q-table, consider designing a custom key based on `obs` to store useful information.
    # NOTE: Keep in mind that your Q-table may not cover all possible states in the testing environment.
    #       To prevent crashes, implement a fallback strategy for missing keys. 
    #       Otherwise, even if your agent performs well in training, it may fail during testing.
    global first_time
    global prev_action
    global target
    global target_pos
    global dest_pos
    global got_passenger
    global in_station

    
    if first_time:
        target_pos = (obs[2 + target * 2], obs[3 + target * 2])
        first_time = False
    print("target: ", target, target_pos)
    state = get_state(obs, prev_action, target_pos, got_passenger)
    action = np.argmax(q_table[state])
    
    # update global variables
    prev_action = action
    taxi_pos = (obs[0], obs[1])
    station_pos = [(obs[2], obs[3]), (obs[4], obs[5]), (obs[6], obs[7]), (obs[8], obs[9])]
    if ((taxi_pos in station_pos and in_station) or taxi_pos == target_pos) and obs[14] == 1 and got_passenger == 0 and action == 4:  # success pick up
        got_passenger = 1
        target += 1
        target_pos = dest_pos if dest_pos != None else (obs[2 + target * 2], obs[3 + target * 2])
        prev_action = -1
    elif got_passenger == 1 and action == 5:  # fail drop off
        got_passenger = 0
        in_station = False
        target -= 1
        target_pos = (obs[0], obs[1])
        prev_action = -1
    elif (obs[0], obs[1]) == target_pos:
        if obs[15] == 1:
            dest_pos = (obs[0], obs[1])
        if (obs[14] == 0 and got_passenger == 0) or (obs[15] == 0 and got_passenger == 1):
            target = (target + 1) % 4
            target_pos = (obs[2 + target * 2], obs[3 + target * 2])
            prev_action = -1
        
    return action

    return random.choice([0, 1, 2, 3, 4, 5]) # Choose a random action
    # You can submit this random agent to evaluate the performance of a purely random strategy.

