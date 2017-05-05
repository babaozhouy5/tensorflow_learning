
# coding: utf-8

# ## 3.基于Deep Q-Network构建的Game AI

# ### 3.1.首先需要对原游戏做改动

# In[1]:

import pygame
from pygame.locals import *

# import sys
import numpy as np
import PIL.Image as im
import PIL.ImageOps as imop

####################
# pygame的游戏坐标系 #
# 0------>x        #
# |                #
# |                #
# |                #
# y                #
####################

BLACK = (0  ,0  ,0  )
WHITE = (255,255,255)
 
SCREEN_SIZE = [320,400]
BAR_SIZE    = [20,   5]
BALL_SIZE   = [15,  15]
 
MOVE_LEFT  = [1, 0, 0]
MOVE_STAY  = [0, 1, 0]
MOVE_RIGHT = [0, 0, 1]

import time
def convert2gray(arr3d):

    image = im.fromarray(arr3d, 'RGB')
    image = image.resize((100, 80))
    image = imop.grayscale(image)
    result_array = np.asarray(image)
    image.close()
    return result_array
    
class Game(object):
    
    def __init__(self):
        
        # 初始化游戏界面
        pygame.init()
        self.clock = pygame.time.Clock()
        self.screen = pygame.display.set_mode(SCREEN_SIZE)
        pygame.display.set_caption('Simple Game')
        
        # 初始分数
        self.score = 0
        
        # ball移动方向
        self.ball_dir_x = -1 # -1 = left 1 = right  
        self.ball_dir_y = -1 # -1 = up   1 = down
        
        # 初始化球
        self.ball_pos_x = SCREEN_SIZE[0]//2 - BALL_SIZE[0]/2
        self.ball_pos_y = SCREEN_SIZE[1]//2 - BALL_SIZE[1]/2
        self.ball = pygame.Rect(self.ball_pos_x, self.ball_pos_y, BALL_SIZE[0], BALL_SIZE[1])
        
        # 初始化挡板
        self.bar_pos_x = SCREEN_SIZE[0]//2-BAR_SIZE[0]//2
        self.bar = pygame.Rect(self.bar_pos_x, SCREEN_SIZE[1]-BAR_SIZE[1], BAR_SIZE[0], BAR_SIZE[1])
    
    # 挡板移动函数
    def bar_move_left(self):
        self.bar_pos_x = self.bar_pos_x - 2
        # 边界修复
        if self.bar_pos_x < 0:
            self.bar_pos_x = 0
       
    def bar_move_right(self):
        self.bar_pos_x = self.bar_pos_x + 2
        # 边界修复
        if self.bar_pos_x > SCREEN_SIZE[0] - BAR_SIZE[0]:
            self.bar_pos_x = SCREEN_SIZE[0] - BAR_SIZE[0]
            
    # 主事件
    def one_step(self, action):
        
        # 更新挡板位置
        if action == MOVE_LEFT:
            self.bar_move_left()
        elif action == MOVE_RIGHT:
            self.bar_move_right()
        
            
        # 刷新游戏中挡板位置
        self.screen.fill(BLACK)
        self.bar.left = self.bar_pos_x
        pygame.draw.rect(self.screen, WHITE, self.bar)

        # 刷新游戏中球的位置
        self.ball.left += self.ball_dir_x * 2
        self.ball.bottom += self.ball_dir_y * 3
        pygame.draw.rect(self.screen, WHITE, self.ball)

        # 球边缘碰撞检测及移动方向调整
        if self.ball.top <= 0 or self.ball.bottom >= (SCREEN_SIZE[1] - BAR_SIZE[1]+1):
            self.ball_dir_y = self.ball_dir_y * -1
        if self.ball.left <= 0 or self.ball.right >= (SCREEN_SIZE[0]):
            self.ball_dir_x = self.ball_dir_x * -1
        
        # 采取动作a_t(在s_t)后获得地奖励r_(t+1)
        reward = 0
        # 挡板与球的碰撞检测
        if self.bar.top <= self.ball.bottom and (self.bar.left < self.ball.right and self.bar.right > self.ball.left):
            reward = 1  # AI击中球的奖励
        elif self.bar.top <= self.ball.bottom and (self.bar.left > self.ball.right or self.bar.right < self.ball.left):
            reward = -1 # AI没有击中球的惩罚
        
        # 注意：这句必须放在update()之前，对应的是当前的动作a_t和状态s_t
        screen_pixel = pygame.surfarray.array3d(pygame.display.get_surface())
        screen_image = convert2gray(screen_pixel)
        pygame.display.update()
        self.clock.tick(60)
        
        return screen_image, reward


# ### 3.2.构建Deep Q-Network模型

# #### 3.2.1.引入依赖包

# In[2]:

from collections import deque
import tensorflow as tf
import random

sess = tf.InteractiveSession()


# #### 3.2.2.定义网络参数

# In[3]:

width, height = 80, 100
num_actions   = 3

X = tf.placeholder(tf.float32, [None, width, height, 4])
Y = tf.placeholder(tf.float32, [None, num_actions])

# 测试观测次数
EXPLORE = 500000
OBSERVE = 50000
# 存储过往经验大小
REPLAY_MEMORY = 500000

initial_epsilon = 1.0
final_epsilon   = 0.01

batch_size = 100
learning_rate = 0.9

# [(size, size, in_channel, output_channel)]
filter_shapes = [(6, 6, 4, 32), (4, 4, 32, 64), (3, 3, 64, 64)]
stride_shapes = [(1, 2, 2, 1), (1, 2, 2, 1), (1, 1, 1, 1)]


# #### 3.2.3.定义Deep Q-Network模型

# In[4]:

def weight_variable(shape):
    return tf.Variable(tf.zeros(shape))

def bias_variable(shape):
    return tf.Variable(tf.zeros(shape))

def conv2d(x, W, bias, strides):
    conv = tf.nn.conv2d(x, W, strides=strides, padding='VALID')
    return tf.nn.relu(conv + bias)

def DeepQNetwork(x):
    
    # 1.convolution layer
    # conv_layer = tf.reshape(-1, width, height, 4)
    conv_layer = x
    for idx, filter_shape in enumerate(filter_shapes):
        with tf.name_scope('conv_%d' % idx):
            W = weight_variable(filter_shape)
            b = bias_variable(filter_shape[-1])
            conv_layer = conv2d(conv_layer, W, b, stride_shapes[idx])
    
    # 2.fully-connected layer
    W_fc1 = weight_variable([16*21*64, 512])
    b_fc1 = bias_variable([512])
    conv_flat = tf.reshape(conv_layer, [-1, 16*21*64])
    fc1_output = tf.nn.relu(tf.matmul(conv_flat, W_fc1) + b_fc1)
    
    # 3.readout layer
    W_fc2 = weight_variable([512, 3])
    b_fc2 = bias_variable([3])
    output = tf.matmul(fc1_output, W_fc2) + b_fc2
    
    return output


# #### 3.2.4.训练网络

# In[5]:

def train_neural_network(x):
    predict_action = DeepQNetwork(x) # give state x, and output Q-values for the three actions

    argmax = tf.placeholder("float", [None, num_actions])
    gt = tf.placeholder("float", [None])
    action = tf.reduce_sum(tf.multiply(predict_action, argmax), reduction_indices = 1)
    cost = tf.reduce_mean(tf.square(action - gt))
    optimizer = tf.train.AdamOptimizer(1e-6).minimize(cost)

    game = Game()
    D = deque()

    image, _ = game.one_step(MOVE_STAY)
    input_image_data = np.stack((image, image, image, image), axis = 2)
    sess.run(tf.global_variables_initializer())

    saver = tf.train.Saver()

    n = 0
    epsilon = initial_epsilon
    while True:
        action_t = predict_action.eval(feed_dict = {X : [input_image_data]})[0]

        argmax_t = np.zeros([num_actions], dtype=np.int)
        if(random.random() <= initial_epsilon):
            maxIndex = random.randrange(num_actions)
        else:
            maxIndex = np.argmax(action_t)
        argmax_t[maxIndex] = 1
        if epsilon > final_epsilon:
            epsilon -= (initial_epsilon - final_epsilon) / EXPLORE

        #for event in pygame.event.get():  macOS需要事件循环，否则白屏
        #    if event.type == QUIT:
        #        pygame.quit()
        #        sys.exit()
        image, reward = game.one_step(list(argmax_t))
        image = np.reshape(image, (80, 100, 1))
        input_image_data1 = np.append(image, input_image_data[:, :, 0:3], axis = 2)

        D.append((input_image_data, argmax_t, reward, input_image_data1))

        if len(D) > REPLAY_MEMORY:
            D.popleft()

        if n > OBSERVE:
            minibatch = random.sample(D, batch_size)
            input_image_data_batch = [d[0] for d in minibatch]
            argmax_batch = [d[1] for d in minibatch]
            reward_batch = [d[2] for d in minibatch]
            input_image_data1_batch = [d[3] for d in minibatch]

            gt_batch = []

            out_batch = predict_action.eval(feed_dict = {X : input_image_data1_batch})

            for i in range(0, len(minibatch)):
                gt_batch.append(reward_batch[i] + learning_rate * np.max(out_batch[i]))

            optimizer.run(feed_dict = {gt : gt_batch, argmax : argmax_batch, X : input_image_data_batch})

        input_image_data = input_image_data1
        n = n+1

        if n % 10000 == 0:
            saver.save(sess, 'game.cpk', global_step = n)  # 保存模型

        print(n, "epsilon:", epsilon, " " ,"action:", maxIndex, " " ,"reward:", reward)


train_neural_network(X)

# In[ ]:



