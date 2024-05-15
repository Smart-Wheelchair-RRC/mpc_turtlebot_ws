from casadi_code import Agent
import os
import numpy as np
import time
import casadi as ca
import matplotlib.pyplot as plt
import copy
import utils
np.set_printoptions(suppress=True)

if __name__ == "__main__":
    rec_video = False

    exp_name = "Dynamic Obstacle Avoidance"
    repo_path = os.path.abspath(os.path.dirname(__file__))
    results_path = repo_path + "/results"
    exp_path = results_path + "/" + exp_name
    plt_dir = exp_path + "/tmp/"
    os.makedirs(exp_path + "/tmp/", exist_ok=True)

    timeout = 100

    agent_v_ub = 12
    agent_v_lb = 0

    y_lane = np.arange(-1000, 1000)
    x1_l_lane = 1.5 * np.ones(y_lane.shape)
    x1_r_lane = 4.5 * np.ones(y_lane.shape)
    x2_l_lane = -1.5 * np.ones(y_lane.shape)
    x3_l_lane = -4.5 * np.ones(y_lane.shape)

    draw_list = []

    y_l_lim = -10
    y_u_lim = 40

    # Fixed target state
    x_target = 5  # Replace with actual x-coordinate of the target
    y_target = 50  # Replace with actual y-coordinate of the target
    theta_target = np.deg2rad(90)  # Replace with actual orientation in radians

    agent1 = Agent(1, [3, -4, np.deg2rad(90)], [x_target, y_target, theta_target], 30)
    draw_list.append(agent1)
    agent1.v_ub = agent_v_ub
    agent1.v_lb = agent_v_lb
    agent1.vl = 5

    # Initialize first obstacle
    #obs = Agent(2, [3, 18, np.deg2rad(90)], [3, 18 + 30, np.deg2rad(90)], 30)
    obs = Agent(2, [-6, 5, np.deg2rad(90)], [30, 40, np.deg2rad(90)], 30)
    obs.v_ub = 8
    obs.v_lb = 0
    obs.vl = 6
    draw_list.append(obs)

    # Initialize new obstacle moving from goal position to agent's starting position
    obs2 = Agent(3, [7, 20, np.deg2rad(90)], [3, 20, np.deg2rad(90)], 30)
    obs2.v_ub = 8
    obs2.v_lb = 0
    obs2.vl = 6
    draw_list.append(obs2)

    # Initialize new obstacle moving from goal position to agent's starting position
    obs3 = Agent(4, [9, 5, np.deg2rad(90)], [5, 18 + 20, np.deg2rad(90)], 30)
    obs3.v_ub = 8
    obs3.v_lb = 0
    obs3.vl = 6
    draw_list.append(obs3)

    # Initialize new obstacle moving from goal position to agent's starting position
    obs4 = Agent(5, [9, 15, np.deg2rad(90)], [5+7, 15, np.deg2rad(90)], 30)
    obs4.v_ub = 8
    obs4.v_lb = 0
    obs4.vl = 6
    draw_list.append(obs4)

    # Initialize new obstacle moving from goal position to agent's starting position
    obs5 = Agent(6, [-5, 20, np.deg2rad(90)], [9, 50, np.deg2rad(90)], 30)
    obs5.v_ub = 7
    obs5.v_lb = 0
    obs5.vl = 2
    draw_list.append(obs5)


    

    # Add obstacles to agent1
    agent1.obstacles.append(obs)
    agent1.obstacles.append(obs2)
    agent1.obstacles.append(obs3)
    agent1.obstacles.append(obs4)
    agent1.obstacles.append(obs5)
    agent1.avoid_obs = True

    # Initial predictions for controls
    agent1.pred_controls()
    obs.pred_controls()
    obs2.pred_controls()
    #obs3.pred_controls()
    #obs4.pred_controls()
    obs5.pred_controls()


    if rec_video:
        plt_sv_dir = plt_dir
        p = 0

    while ca.norm_2(agent1.state_init - agent1.state_target) >= 1 and timeout > 0:
        timeout -= agent1.dt
        t1 = time.time()

        # Predict controls for all agents and obstacles
        agent1.pred_controls()
        obs.pred_controls()
        obs2.pred_controls()
        obs3.pred_controls()
        obs4.pred_controls()
        obs5.pred_controls()  # Control prediction for the new obstacle

        print(time.time() - t1)
        print("#################")

        # Update states and velocities for all agents and obstacles
        agent1.vl += agent1.u0[0, 0] * agent1.dt
        agent1.wl += agent1.u0[1, 0] * agent1.dt
        agent1.state_init = agent1.X0[:, 1]
        agent1.i_state = np.array(agent1.state_init.full()).reshape(3)

        obs.vl += obs.u0[0, 0] * obs.dt
        obs.wl += obs.u0[1, 0] * obs.dt
        obs.state_init = copy.deepcopy(obs.X0[:, 1])
        obs.i_state = np.array(obs.state_init.full()).reshape(3)

        obs2.vl += obs2.u0[0, 0] * obs2.dt
        obs2.wl += obs2.u0[1, 0] * obs2.dt
        obs2.state_init = copy.deepcopy(obs2.X0[:, 1])
        obs2.i_state = np.array(obs2.state_init.full()).reshape(3)

        obs3.vl += obs3.u0[0, 0] * obs3.dt
        obs3.wl += obs3.u0[1, 0] * obs3.dt
        obs3.state_init = copy.deepcopy(obs3.X0[:, 1])
        obs3.i_state = np.array(obs3.state_init.full()).reshape(3)

        obs4.vl += obs4.u0[0, 0] * obs4.dt
        obs4.wl += obs4.u0[1, 0] * obs4.dt
        obs4.state_init = copy.deepcopy(obs4.X0[:, 1])
        obs4.i_state = np.array(obs4.state_init.full()).reshape(3)

        obs5.vl += obs5.u0[0, 0] * obs5.dt
        obs5.wl += obs5.u0[1, 0] * obs5.dt
        obs5.state_init = copy.deepcopy(obs5.X0[:, 1])
        obs5.i_state = np.array(obs5.state_init.full()).reshape(3)

        utils.draw(draw_list)

        
        # Commented out lane plotting
        # plt.plot(x1_r_lane, y_lane, 'k', linewidth=1)
        # plt.plot(x1_l_lane, y_lane, 'k', linewidth=1)
        # plt.plot(x2_l_lane, y_lane, 'k', linewidth=1)
        # plt.plot(x3_l_lane, y_lane, 'k', linewidth=1)

        #Plot size
        plt.xlim([-40, 40])
        plt.ylim([y_l_lim, y_u_lim])

        if rec_video:
            plt.savefig(plt_sv_dir + str(p) + ".png", dpi=500, bbox_inches='tight')
            p += 1
            plt.clf()
        else:
            plt.pause(1e-10)
            plt.clf()

        y_l_lim = agent1.i_state[1] - 10
        y_u_lim = agent1.i_state[1] + 40

    plt.close()

    # if rec_video:
    #      os.system('ffmpeg -r 10 -f image2 -i ' + plt_dir + '/%d.png -s 1000x1000 -pix_fmt yuv420p -y ' + exp_path + '/' + exp_name + '.mp4')
