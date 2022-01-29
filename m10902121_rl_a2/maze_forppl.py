
from testpolicy import PolicyNet
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



# field = np.array(
# [[0, 3, 0, 0, 0, 3, 0, 0, 3],
#  [0, 0, 0, 0, 0, 0, 0, 0, 0],
#  [0, 0, 0, 0, 0, 0, 0, 0, 3],
#  [0, 0, 0, 0, 0, 0, 3, 0, 2],
#  [0, 0, 0, 0, 0, 0, 0, 0, 0],
#  [0, 0, 0, 0, 0, 0, 0, 0, 0]])
basket_pos = np.array([3, 8])
ball_pos = np.array([3, 0])
opp_pos = [np.array([0, 5]), np.array([2, 8]), np.array([0, 1]), np.array([3, 6]), np.array([0, 8])]
agent_pos = np.array([1, 2])
UNIT = 40   # pixels    
done = False
MAZE_H, MAZE_W = 6, 9
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
        self.canvas.delete(self.agent)
        self.agent = self.create_block(agent_pos, type=4)
        self.canvas.delete(self.ball)
        self.ball = self.create_ball(ball_pos)
        self.with_ball = False
        return self.getPos(self.agent)

    def step(self, action):
        action_dict= {
            0: np.array([-1, 0]), #up
            1: np.array([1, 0]),  #down
            2: np.array([0, 1]),  #right
            3: np.array([0, -1])
        }
        if self.with_ball:
            action_dict= {
                0: np.array([-1, 0]), #up
                1: np.array([1, 0]),  #down
                2: np.array([0, 1]),  #right
                3: np.array([0, -1]),
                4: np.array([0, 0])
            }
        move = np.array([0, 0])        
        pb = random.random()
        if  pb <= 0.7:
            scale = 1
        else:
            scale = 2
        try:
            
            move = scale * action_dict[action]
        except:
            pass
        
        agentpos = self.getPos(self.agent)
        tmppos = agentpos        
        self.canvas.move(self.agent, move[1]*40, move[0]*40)  # move agent

        # s_ = self.canvas.coords(self.agent)  # next state
        s_ = self.getPos(self.agent)
        agentpos = self.getPos(self.agent)
        # rewards function
        done=False
        reward=-1
        if action == 4:

            distance = np.sqrt((basket_pos[0]-agentpos[0])**2 + (basket_pos[1]-agentpos[1])**2)
            distance = np.round(distance, 2)
            pb = 0
            reward = -1
            if distance == 1:
                pb = 0.9
                reward = 2
                
            elif distance<=3 and distance > 1:
                pb = 0.66
                reward = 10
                
            elif distance > 3 and distance <= 4:
                pb = 0.1
                reward = 30
            a = random.random()   
            # print(f'rdn{a} pb{pb}')
            if  a < pb:
                done=True
                self.ball = self.create_ball(self.getPos(self.basket))
            else:
                done=False
                reward = -1
                self.with_ball = False
                self.canvas.delete(self.agent)
                self.agent = self.create_block(agentpos, 4)
                pos = np.array([(MAZE_H//2), round(MAZE_W*0.8)])
                self.ball = self.create_ball(pos)
                # print()
                # print(self.getPos(self.ball))
            return s_, reward, done

        
        if agentpos[0] < 0 or agentpos[0] > (MAZE_H - 1):
            return s_, -100, True
        elif agentpos[1] < 0 or agentpos[1] > (MAZE_W - 1):
            return s_, -100, True
        
        if list(s_) == list(self.getPos(self.ball)):
            self.with_ball = True
            reward = 20
            pos = self.getPos(self.agent)
            self.canvas.delete(self.ball)
            self.canvas.delete(self.agent)            
            self.agent = self.create_block(pos, type=5)
            done = False
            
        else:
            reward = -1
            done = False
        
        for i in range(len(self.opp)):
            if list(s_) == list(self.getPos(self.opp[i])):
                reward = -5
                pos = self.getPos(self.agent)
                self.canvas.delete(self.agent)            
                self.agent = self.create_block(tmppos, type=4)
                self.canvas.delete(self.ball)
                self.ball = self.create_ball(ball_pos)
                break                        
        return s_, reward, done

    def render(self):
        time.sleep(0.1)
        self.update()

def update():
    for t in range(3):
        s = env.reset()
        done = False
        while not done:
            env.render()
            # for ppl to test
            a = str(input())
            action_change ={
                'w':0,
                's':1,
                'd':2,
                'a':3,
                'q':4
            }
            action = action_change[a]
            #------------------------
            s, r, done = env.step(action)
            print(f's:{s} r:{r} done:{done}')
    sys.exit()







            

if __name__ == '__main__':
    env = Maze()
    update() 
    # env.after(100, update)
    # env.mainloop()