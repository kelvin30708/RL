# for policy gradient
import torch.nn.functional as F
import os
import torch
from torch.distributions import Categorical
from torch.autograd import Variable
from itertools import count
import matplotlib.pyplot as plt
from testpolicy import PolicyNet
import tqdm
# for environment
from pickle import FALSE
from distutils.log import error
import numpy as np
import time
import sys
import random
import sys
if sys.version_info.major == 2:
    import Tkinter as tk
else:
    import tkinter as tk




# 6 * 9
# basket_pos = np.array([3, 8])
# ball_pos = np.array([3, 0])
# opp_pos = [np.array([0, 5]), np.array([2, 8]), np.array([0, 1]), np.array([3, 6]), np.array([0, 8])]
# agent_pos = np.array([1, 2])
# done = False
# UNIT = 40   # pixels    
# MAZE_H, MAZE_W = 6, 9

# 10 * 12
# basket_pos = np.array([5, 11])
# ball_pos = np.array([5, 0])
# opp_pos = [np.array([4, 1]), np.array([9, 3]), np.array([8, 9]), np.array([4, 3]), np.array([6, 8]), np.array([3, 5]), np.array([4, 5]), np.array([2, 9]), np.array([8, 7]), np.array([5, 7])]
# agent_pos = np.array([7, 2])
# done = False
# UNIT = 40   # pixels    
# MAZE_H, MAZE_W = 10, 12
# 
# 12* 15
basket_pos = np.array([6, 14])
ball_pos = np.array([6, 0])
opp_pos = [np.array([ 7, 10]), np.array([10,  0]), np.array([10, 14]), np.array([11, 10]), np.array([9, 0]), np.array([ 2, 12]), np.array([11,  4]), np.array([ 5, 10]), np.array([4, 3]), np.array([4, 4]), np.array([6, 4]), np.array([6, 6]), np.array([8, 2]), np.array([11, 13]), np.array([4, 8]), np.array([9, 1]), np.array([9, 9]), np.array([7, 5]), np.array([ 1, 14]), np.array([11,  0]), np.array([7, 9]), np.array([6, 3]), np.array([3, 8]), np.array([0, 5]), np.array([3, 6]), np.array([8, 5]), np.array([ 5, 13]), np.array([2, 6]), np.array([1, 1]), np.array([1, 2]), np.array([3, 4]), np.array([2, 5]), np.array([ 2, 13]), np.array([ 3, 10]), np.array([9, 2]), np.array([11, 14]), np.array([9, 7]), np.array([5, 4]), np.array([0, 2]), np.array([11,  7]), np.array([7, 4]), np.array([5, 8]), np.array([ 5, 12]), np.array([1, 8]), np.array([11, 12]), np.array([0, 8]), np.array([11,  6]), np.array([ 3, 11]), np.array([3, 3]), np.array([6, 7])]
agent_pos = np.array([1, 5])
done = False
UNIT = 40   # pixels    
MAZE_H, MAZE_W = 12, 15 

class Maze(tk.Tk, object):
    def __init__(self):
        super(Maze, self).__init__()
        self.title('basketball')
        self.canvas = tk.Canvas(self, bg='white',
                           height=MAZE_H * UNIT,
                           width=MAZE_W * UNIT)               
        
        #create ball
        self.ball = self.create_ball(ball_pos)      
        self.opp = []

        #create basket
        self.basket = self.create_block(basket_pos, type=2)
        
        #create opp
        for x in opp_pos:
            obj = self.create_block(x, type=3)
            self.opp.append(obj)
        
        #create agent
        self.agent = self.create_block(agent_pos, type=4)
        self.with_ball = False

        #
        self.n_step = 0
        self._build_maze()
    def create_block(self, pos, type=3):
        if type == 0:
            return
        block_color= {
            2: 'green', # basket
            3: 'black', # opp
            4: 'red',   # agent
            5: 'yellow' # agent with ball
        }
        origin = np.array([20, 20])
        pos = pos[::-1]
        block_center = origin + np.array([UNIT * pos[0], UNIT * pos[1]])
        return self.canvas.create_rectangle(
            block_center[0] - 15, block_center[1] - 15,
            block_center[0] + 15, block_center[1] + 15,
            fill=block_color[type])
        
    def create_ball(self, pos):
        origin = np.array([20, 20])
        pos = pos[::-1]
        oval_center = origin + np.array([UNIT * pos[0], UNIT * pos[1]])
        return self.canvas.create_oval(
            oval_center[0] - 15, oval_center[1] - 15,
            oval_center[0] + 15, oval_center[1] + 15,
            fill='orange')
    
    def _build_maze(self):
        # create grids
        for c in range(0, MAZE_W * UNIT, UNIT):
            x0, y0, x1, y1 = c, 0, c, MAZE_H * UNIT
            self.canvas.create_line(x0, y0, x1, y1)
        for r in range(0, MAZE_H * UNIT, UNIT):
            x0, y0, x1, y1 = 0, r, MAZE_W * UNIT, r
            self.canvas.create_line(x0, y0, x1, y1)
        # pack all
        self.canvas.pack()
    
    def getPos(self, obj):
        S = self.canvas.coords(obj)
        try:
            pos = np.array([S[1], S[0]])
            pos -= 5
            pos /= 40
            pos = pos.astype(int)
        except:
            pos = np.array([-1, -1])
        
        return pos

    def reset(self):
        # time.sleep(0.5)
        reward = 0
        self.canvas.delete(self.agent)
        self.agent = self.create_block(agent_pos, type=4)
        self.canvas.delete(self.ball)
        self.ball = self.create_ball(ball_pos)
        self.with_ball = False
        self.n_step = 0
        return self.getPos(self.agent), self.getPos(self.ball), self.with_ball, reward, done
    
    def action_fun(self, action):
        self.n_step += 1
        action_dict= {
            0: np.array([-1, 0]), # up
            1: np.array([1, 0]),  # down
            2: np.array([0, 1]),  # right
            3: np.array([0, -1]), # left
            4: np.array([0, 0])   # shoot
        }
        move = np.array([0, 0])        
        pb = random.random()
        if  pb <= 0.7:
            scale = 1
        else:
            scale = 2         
        move = scale * action_dict[action]
        self.canvas.move(self.agent, move[1]*40, move[0]*40)  # move agent
        
    def step(self, action):             
        done=False
        reward = 0
        
        # state
        s = self.getPos(self.agent)                
        
        # act fun.
        self.action_fun(action)
        s_ = self.getPos(self.agent)
                
        # rwd func.
        reward, done = self.reward_fun(pre_state=s, state=s_,  action=action)
        # rwd func. changes the agent pos
        s_ = self.getPos(self.agent)
        reward += self.n_step * 1
        return s_, self.getPos(self.ball), self.with_ball, reward, done
        
    def reward_fun(self, pre_state, state, action):
        reward = 0
        if state[0] < 0 or state[0] > (MAZE_H - 1) or state[1] < 0 or state[1] > (MAZE_W - 1):
            self.with_ball = False
            self.canvas.delete(self.agent)
            reward = - 100
            done = True
            return reward, done
        
        if list(state) == list(self.getPos(self.ball)):
            self.with_ball = True
            reward += 200 # could be deleted
            pos = self.getPos(self.agent)
            self.canvas.delete(self.ball)
            self.canvas.delete(self.agent)            
            self.agent = self.create_block(pos, type=5)
            done = False
        else:
            for i in range(len(self.opp)):
                if list(state) == list(self.getPos(self.opp[i])):                
                    self.canvas.delete(self.agent)            
                    self.agent = self.create_block(pre_state, type=4)
                    self.canvas.delete(self.ball)
                    self.ball = self.create_ball(ball_pos)
                    self.with_ball = False
                    reward -= 50
                    done = False
                    break                       
        # return reward, done
        if self.with_ball:
            # shoot
            distance = np.sqrt((basket_pos[0]-pre_state[0])**2 + (basket_pos[1]-pre_state[1])**2)
            distance = np.round(distance, 2)
            if action == 4:
                
                pb = 0
                reward = 0
                if distance == 1:
                    pb = 0.9
                    reward += 2

                elif distance<=3 and distance > 1:
                    pb = 0.66
                    reward += 10

                elif distance > 3 and distance <= 4:
                    pb = 0.1
                    reward += 30
                rdn = random.random()   
                # print(f'rdn{a} pb{pb}')
                if  rdn < pb:
                    done=True                
                    self.ball = self.create_ball(self.getPos(self.basket))
                else:
                    done=False
                    reward += 0
                    self.with_ball = False
                    pos = np.array([(MAZE_H//2), round(MAZE_W*0.8)])
                    self.ball = self.create_ball(pos)
                    self.canvas.delete(self.agent)
                    self.agent = self.create_block(state, type=4)
            
            else:
                if distance == 1:
                    reward += 25*3
                elif distance<=3 and distance > 1:
                    reward += 4*2
                elif distance > 3 and distance <= 4:
                    reward += 1
                else:
                    reward+=0
                # distance = np.sqrt((basket_pos[0] - pre_state[0])**2 + (basket_pos[1] - pre_state[1])**2)
                # distance = np.round(distance, 2)

                # distance2 = np.sqrt((basket_pos[0] - state[0])**2 + (basket_pos[1] - state[1])**2)
                # distance2 = np.round(distance, 2)
                # if distance2 > distance:
                #     reward = 100000
                # else:
                #     reward = -np.sum(np.power((distance2 - distance), 2))
                # else:
                # reward = (distance2 - distance)*10
                
                done = False
            return reward, done
        
        else:
            if action == 4:
                reward -= 100
                done=False
            else:
                eps = 0.1
                ballps = self.getPos(self.ball)
                # print(ballps)
                distance = np.sqrt((ballps[0] - state[0])**2 + (ballps[1] - state[1])**2)
                # distance = np.round(distance, 2)

                # distance2 = np.sqrt((ballps[0] - state[0])**2 + (ballps[1] - state[1])**2)
                # distance2 = np.round(distance, 2)
                reward = 2*(distance + eps)
                # if distance <= 1:
                #     reward = 10
                # elif distance<=3 and distance > 1:
                #     reward = 5
                # elif distance > 3 and distance <= 4:
                #     reward = 1
                # else:
                #     reward=0
                done=False    
        return reward, done

        
        
    def render(self):
        time.sleep(0.05)
        self.update()
    
def main():
    # Parameters
    num_episode = 5000
    batch_size = 2
    learning_rate = 1e-6
    gamma = 0.99

    env = Maze()
    policy_net = PolicyNet()
    policy_net.__init__(input_feat=MAZE_H * MAZE_W + 1)
    policy_net.train()
    optimizer = torch.optim.Adam(policy_net.parameters(), lr=learning_rate)

    # Batch History
    state_pool = []
    action_pool = []
    reward_pool = []
    steps = 0
    quick_obs = 4000
    QUICK = 100
    WATCH = 3
    each_obs = 1500

    rwd_list=[]
    loss_list=[]
    def plot_rewards():
        plt.figure(1)
        plt.clf()
        
        # rwd_t = torch.FloatTensor(rwd_list)
        plt.title('Training...')
        plt.xlabel('Episode')
        plt.ylabel('Rewards')
        
        plt.plot(rwd_list)
        
        # plt.plot(loss_list)
        
        plt.pause(1e-9)  # pause a bit so that plots are updated
    
    def plot_losses():
        plt.figure(2)
        plt.clf()
        x = [batch_size*i for i in range(1, len(loss_list)+1)]
        # x = lis[]
        
        plt.title('Training...')
        plt.xlabel(f'Each {batch_size} Episode')
        plt.ylabel('Losses')
        # plt.plot(rwd_t.numpy())
        plt.plot(x, loss_list, 'g')
        
        plt.pause(0.001)  # pause a bit so that plots are updated

    
    for e in range(num_episode):

        state, ballpos, withball, reward, done = env.reset()
        
        # input the whole grid
        #===========================================
        test_input = np.zeros((MAZE_H, MAZE_W))
        if(ballpos[0]==-1 and ballpos[1]==-1):
            test_input[test_input == 1] = 0
        else:
            test_input[ballpos[0]][ballpos[1]]= 1
        test_input[basket_pos[0]][basket_pos[1]]= 2
        for x in opp_pos:
            test_input[x[0]][x[1]]=3
        test_input[state[0]][state[1]] = 4
                
        test_input = test_input.flatten()
        test_input = np.append(test_input, withball)
        state = test_input
        state = torch.from_numpy(state).float()
        state = Variable(state)
        #=====================================
        
        
        if e > quick_obs:
            env.render()
        rwd = 0
        for t in count():
            
            probs = policy_net(state)
            m = Categorical(probs)
            action = m.sample() 

            next_state, ballpos, withball, reward, done = env.step(int(action))
            if e > quick_obs:
                env.render()

            # To mark boundarys between episodes
            # if done:
            #     reward = 0

            state_pool.append(state)
            action_pool.append(float(action))
            reward_pool.append(reward)

            state = next_state
            
            # input the whole grid
            #===========================================
            test_input = np.zeros((MAZE_H, MAZE_W))
            
            if(ballpos[0]==-1 and ballpos[1]==-1):
                test_input[test_input == 1] = 0
            else:
                test_input[ballpos[0]][ballpos[1]]= 1

             # fixed component
            test_input[basket_pos[0]][basket_pos[1]]= 2
            for x in opp_pos:
                test_input[x[0]][x[1]]=3
            
            test_input[state[0]][state[1]] = 4
            test_input = np.append(test_input, withball)
            state = test_input
            state = torch.from_numpy(state).float()
            state = Variable(state)
            #===========================================

            steps += 1
            rwd += reward
            if done or t > 50:
                # episode_durations.append(t + 1)
                rwd_list.append(rwd)
                plot_rewards()
                break

        # Update policy
        if e > 0 and e % batch_size == 0:

            # Discount reward
            running_add = 0
            for i in reversed(range(steps)):
                if reward_pool[i] == 0:
                    running_add = 0
                else:
                    running_add = running_add * gamma + reward_pool[i]
                    reward_pool[i] = running_add
            
            # Normalize reward
            reward_mean = np.mean(reward_pool)
            reward_std = np.std(reward_pool)
            for i in range(steps):
                reward_pool[i] = (reward_pool[i] - reward_mean) / reward_std

            # Gradient Desent
            optimizer.zero_grad()
            
            losses = []
            for i in range(steps):
                
                
                state = state_pool[i]
                action = Variable(torch.FloatTensor([action_pool[i]]))
                reward = reward_pool[i]

                probs = policy_net(state)
                m = Categorical(probs)
                loss = -m.log_prob(action) * reward  # Negtive score function x reward
                
                losses.append(loss.item())                        
                loss.backward()
                torch.nn.utils.clip_grad_norm_(policy_net.parameters(), 10)
            # print(np.mean(losses))
            # for i in range(batch_size):
            loss_list.append(np.mean(losses))
            # plot_rewards()
            plot_losses()
            optimizer.step()

            state_pool = []
            action_pool = []
            reward_pool = []
            steps = 0

if __name__ == '__main__':
    main()
    os.system("pause")