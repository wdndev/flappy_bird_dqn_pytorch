import pdb
import cv2
import sys
import os
sys.path.append("game/")
import game.wrapped_flappy_bird as game
import random
import numpy as np
from collections import deque
import torch
from torch.autograd import Variable
import torch.nn as nn

GAME = 'bird' # the name of the game being played for log files
ACTIONS = 2 # number of valid actions
GAMMA = 1 # decay rate of past observations
# 前OBSERVE轮次，不对网络进行训练，只是收集数据，存到记忆库中
# 第OBSERVE到OBSERVE+EXPLORE轮次中，对网络进行训练，且对epsilon进行退火，逐渐减小epsilon至FINAL_EPSILON
# 当到达EXPLORE轮次时，epsilon达到最终值FINAL_EPSILON，不再对其进行更新
OBSERVE = 2000. 
EXPLORE = 200000. 
FINAL_EPSILON = 0.0001 
INITIAL_EPSILON = 0.0001 
REPLAY_MEMORY = 50000 
BATCH_SIZE = 64 
# 每隔FRAME_PER_ACTION轮次，就会有epsilon的概率进行探索
FRAME_PER_ACTION = 1
# 每隔UPDATE_TIME轮次，对target网络的参数进行更新
UPDATE_TIME = 500000000
width = 80
height = 80

def preprocess(observation):
    """ 将一帧彩色图像处理成黑白的二值图像
    """
    observation = cv2.cvtColor(cv2.resize(observation, (80, 80)), cv2.COLOR_BGR2GRAY)
    ret, observation = cv2.threshold(observation,1,255,cv2.THRESH_BINARY)
    return np.reshape(observation, (1,80,80))

class DeepNetWork(nn.Module):
    """  神经网络结构
    """
    def __init__(self,):
        super(DeepNetWork,self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=4, out_channels=32, kernel_size=8, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True)
        )
        self.fc1 = nn.Sequential(
            nn.Linear(1600,256),
            nn.ReLU()
        )
        self.out = nn.Linear(256,2)

    def forward(self, x):
        """ 前向传播
        """
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = x.view(x.size(0),-1)
        x = self.fc1(x)
        return self.out(x)

class BirdDQN(object):
    def save(self):
        print("save model param")
        torch.save(self.Q_net.state_dict(), 'params3.pth')

    def load(self):
        if os.path.exists("params3.pth"):
            print("load model param")
            self.Q_net.load_state_dict(torch.load('params3.pth'))
            self.Q_netT.load_state_dict(torch.load('params3.pth'))

    def __init__(self,actions):
        # 在每个timestep下agent与环境交互得到的转移样本 (st,at,rt,st+1) 储存到回放记忆库，
        # 要训练时就随机拿出一些（minibatch）数据来训练，打乱其中的相关性
        self.replayMemory = deque() # init some parameters
        self.timeStep = 0
        # 有epsilon的概率，随机选择一个动作，1-epsilon的概率通过网络输出的Q（max）值选择动作
        self.epsilon = INITIAL_EPSILON
        # 初始化动作
        self.actions = actions
        # 当前值网络, 目标网络
        self.Q_net=DeepNetWork()
        self.Q_netT=DeepNetWork()
        # 加载训练好的模型，在训练的模型基础上继续训练
        self.load()
        # 使用均方误差作为损失函数
        self.loss_func=nn.MSELoss()
        LR=1e-6
        self.optimizer = torch.optim.Adam(self.Q_net.parameters(), lr=LR)

    def train(self): # Step 1: obtain random minibatch from replay memory
        """ 使用minibatch训练网络
        """
        # 从记忆库中随机获得BATCH_SIZE个数据进行训练
        minibatch = random.sample(self.replayMemory, BATCH_SIZE)
        state_batch = [data[0] for data in minibatch]
        action_batch = [data[1] for data in minibatch]
        reward_batch = [data[2] for data in minibatch]
        nextState_batch = [data[3] for data in minibatch] # Step 2: calculate y
        # y_batch用来存储reward
        y_batch = np.zeros([BATCH_SIZE,1])
        nextState_batch=np.array(nextState_batch) #print("train next state shape")
        #print(nextState_batch.shape)
        nextState_batch=torch.Tensor(nextState_batch)
        action_batch=np.array(action_batch)
        # 每个action包含两个元素的数组，数组必定是一个1，一个0，最大值的下标也就是该action的下标
        index=action_batch.argmax(axis=1)
        # print("action "+str(index))
        index=np.reshape(index,[BATCH_SIZE,1])
        # 预测的动作的下标
        action_batch_tensor=torch.LongTensor(index)
        # 使用target网络，预测nextState_batch的动作
        QValue_batch = self.Q_netT(nextState_batch)
        QValue_batch=QValue_batch.detach().numpy()
        # 计算每个state的reward
        for i in range(0, BATCH_SIZE):
            terminal = minibatch[i][4]
            if terminal:
                y_batch[i][0]=reward_batch[i]
            else:
                # 这里的QValue_batch[i]为数组，大小为所有动作集合大小，QValue_batch[i],代表
                # 做所有动作的Q值数组，y计算为如果游戏停止，y=rewaerd[i],如果没停止，则y=reward[i]+gamma*np.max(Qvalue[i])
                # 代表当前y值为当前reward+未来预期最大值*gamma(gamma:经验系数)
                y_batch[i][0]=reward_batch[i] + GAMMA * np.max(QValue_batch[i])

        y_batch=np.array(y_batch)
        y_batch=np.reshape(y_batch,[BATCH_SIZE,1])
        state_batch_tensor=Variable(torch.Tensor(state_batch))
        y_batch_tensor=Variable(torch.Tensor(y_batch))
        y_predict=self.Q_net(state_batch_tensor).gather(1,action_batch_tensor)
        loss=self.loss_func(y_predict,y_batch_tensor)
        # print("loss is "+str(loss))
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        # 每隔UPDATE_TIME轮次，用训练的网络的参数来更新target网络的参数
        if self.timeStep % UPDATE_TIME == 0:
            self.Q_netT.load_state_dict(self.Q_net.state_dict())
            self.save()

    def setPerception(self,nextObservation,action,reward,terminal, acc_reward): #print(nextObservation.shape)
        """ 更新记忆库，若轮次达到一定要求则对网络进行训练
        """
        # 每个state由4帧图像组成
        # nextObservation是新的一帧图像,记做5。currentState包含4帧图像[1,2,3,4]，则newState将变成[2,3,4,5]
        newState = np.append(self.currentState[1:,:,:],nextObservation,axis = 0) # newState = np.append(nextObservation,self.currentState[:,:,1:],axis = 2)
        # 将当前状态存入记忆库
        self.replayMemory.append((self.currentState,action,reward,newState,terminal))
        # 若记忆库已满，替换出最早进入记忆库的数据
        if len(self.replayMemory) > REPLAY_MEMORY:
            self.replayMemory.popleft()
        # 在训练之前，需要先观察OBSERVE轮次的数据，经过收集OBSERVE轮次的数据之后，开始训练网络
        if self.timeStep > OBSERVE: # Train the network
            self.train()

        # print info
        state = ""
        # 在前OBSERVE轮中，不对网络进行训练，相当于对记忆库replayMemory进行填充数据
        if self.timeStep <= OBSERVE:
            state = "observe"
        elif self.timeStep > OBSERVE and self.timeStep <= OBSERVE + EXPLORE:
            state = "explore"
        else:
            state = "train"
        print ("step:", self.timeStep, "; state:", state,  "; epsilon: ", self.epsilon, "; acc reward: %.2f" % acc_reward)
        # if self.timeStep % 500 == 0:
            # print ("step: ", self.timeStep, " ; state: ", state, "; epsilon: ", self.epsilon)
            # print ("step:", self.timeStep, "; state:", state, "; reward:", reward)
        self.currentState = newState
        self.timeStep += 1

    def getAction(self):
        """ 获得下一步要执行的动作
        """
        currentState = torch.Tensor([self.currentState])
        # QValue为网络预测的动作
        QValue = self.Q_net(currentState)[0]
        action = np.zeros(self.actions)
        # FRAME_PER_ACTION=1表示每一步都有可能进行探索
        if self.timeStep % FRAME_PER_ACTION == 0:
            if random.random() <= self.epsilon:
                action_index = random.randrange(self.actions)
                # print("choose random action " + str(action_index))
                action[action_index] = 1
            else: # 1-epsilon的概率通过神经网络选取下一个动作
                action_index = np.argmax(QValue.detach().numpy())
                # print("choose qnet value action " + str(action_index))
                action[action_index] = 1
        else:
            action[0] = 1  # do nothing

        # 随着迭代次数增加，逐渐减小episilon
        if self.epsilon > FINAL_EPSILON and self.timeStep > OBSERVE:
            self.epsilon -= (INITIAL_EPSILON - FINAL_EPSILON) / EXPLORE
        return action

    def setInitState(self, observation):
        """ 初始化状态
        """
        # 增加一个维度，observation的维度是80x80，讲过stack()操作之后，变成4x80x80
        self.currentState = np.stack((observation, observation, observation, observation),axis=0)
        print(self.currentState.shape)

if __name__ == '__main__': 
    actions = 2 # 动作个数
    brain = BirdDQN(actions) 
    flappyBird = game.GameState() 
    action0 = np.array([1,0]) # 一个随机动作
    # 执行一个动作，获得执行动作后的下一帧图像、reward、游戏是否终止的标志
    observation0, reward0, terminal = flappyBird.frame_step(action0)
    observation0 = cv2.cvtColor(cv2.resize(observation0, (80, 80)), cv2.COLOR_BGR2GRAY)
    ret, observation0 = cv2.threshold(observation0,1,255,cv2.THRESH_BINARY)
    # 将一帧图片重复4次，每一张图片为一个通道，变成通道为4的输入，即初始输入是4帧相同的图片
    brain.setInitState(observation0)
    print(brain.currentState.shape) 

    acc_reward = 0

    while 1!= 0:
        # 获取下一个动作
        action = brain.getAction()
        # 执行动作，获得执行动作后的下一帧图像、reward、游戏是否终止的标志
        nextObservation,reward,terminal = flappyBird.frame_step(action)
        # 将一帧彩色图像处理成黑白的二值图像
        nextObservation = preprocess(nextObservation)
        #print(nextObservation.shape)
        acc_reward += reward
        brain.setPerception(nextObservation,action,reward,terminal, acc_reward)
        if terminal :
            acc_reward = 0
