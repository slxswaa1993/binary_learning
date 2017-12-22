#-*- coding=utf-8 -*-
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
import tensorflow as tf
import pickle
import gc
from sklearn.model_selection import train_test_split

#读取数据集，一共49768个样本，104个类别。每个样本shape=[19470,64]也就是对于RNN来说，num_steps=19470,input_size=64
#data_lenth是每个样本实际的num_steps，去掉padding的影响。
#mnist = input_data.read_data_sets('mnist',one_hot=True)
print 'Loading Dataset ......'
data_file = open('./data/train_data.pkl', mode='r')
label_file = open('./data/train_label.pkl', mode='r')
lenth_file = open('./data/data_lenth.pkl', mode='r')
data_set = pickle.load(data_file)
label_set = pickle.load(label_file)
data_lenth = pickle.load(lenth_file)
print 'successfully load dataset !'
train_x,test_x,train_y,test_y,train_lenth,test_lenth = train_test_split(data_set,label_set,data_lenth,test_size=0.3,shuffle=True)
#回收不用的内存
del data_set,label_set,data_lenth
gc.collect()
#
#设置batch_size=64,则每次放入128张图片，每个时间
input_size = 8
num_steps = 19470
learning_rate = 0.001
batch_size =128
n_classes =104
num_states = [32,32,64]


#build graph
sess = tf.Session()

#首先传入数据，数据传入后要进行何种变化之后再说，先按照28*28的数据输入
with tf.name_scope('input'):
    x =tf.placeholder(dtype=tf.float32,shape=[None,num_steps,input_size])
    y =tf.placeholder(dtype=tf.int32,shape=[None])
    seq_lenth = tf.placeholder(dtype=tf.int32,shape=[None])
    onehot_y = tf.one_hot(y,n_classes)
    tf.summary.histogram('x',x)
    tf.summary.histogram('y',y)
    tf.summary.histogram('lenth',seq_lenth)
#传入x后，要将其送到rnn中，RNN的计算过程:放入batch里所有样本的第一个xt,计算loss,更新梯度,放入batch里所有样本的xt+1。
#过去我们的x的shape是[batch_size,num_steps,input_size]，现在要以时间序列为第一维,[num_steps,batch_size,input_size]

def get_rnnCell(num_states):
    return tf.contrib.rnn.BasicLSTMCell(num_states)

#创建rnn cell,num_states是RNN神经元个数
with tf.name_scope('rnn'):
    #cell = tf.contrib.rnn.BasicLSTMCell(num_states)
    cell = tf.contrib.rnn.MultiRNNCell([get_rnnCell(num_states[i]) for i in range(3)])
    state = cell.zero_state(batch_size,tf.float32)  #state就是RNN的状态ht,ht的shape应该为[batch_size,num_state]如果LSTM，因为有C,H两个状态，所以state是[batch_size,num_states]组成的tuple
    # outputs = []  #每一个时刻都有输出，append到output
    # states =[]   #每一个时刻都有ht,append到states         在basicRNN中，返回的output等于state
    # for i in range(num_steps-1):
    #     output,state = cell.call(input_x[i],state)
    #     outputs.append(output)

    outputs,states = tf.nn.dynamic_rnn(cell,x,initial_state=state,sequence_length=seq_lenth)
    #取最后一个时刻的输出,每个样本的序列长度不一样，取每个样本lenth-1的output。
    not_removeing_output = outputs[:,-1,:]
    remove_padding_outputs =[]
    for i in range(batch_size):
        index = seq_lenth[i]-1
        output = tf.reshape(outputs[i][index],[1,-1])
        remove_padding_outputs.append(output)
    remove_padding_outputs = tf.concat(remove_padding_outputs,0)
    weight = tf.Variable(tf.truncated_normal(shape=[num_states[-1],n_classes]))
    bias = tf.Variable(tf.truncated_normal(shape=[n_classes]))
    pred = tf.matmul(remove_padding_outputs,weight)+bias
    tf.summary.histogram('outputs',remove_padding_outputs)
    tf.summary.histogram('weights',weight)
    tf.summary.histogram('bias',bias)
    tf.summary.histogram('pred',pred)
#计算loss，定义优化策略
with tf.name_scope('train'):
    # 定义全局计数变量
    global_step = tf.Variable(0, trainable=False)
    loss = tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits(labels=onehot_y,logits=pred))
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss,global_step=global_step)
    correct_pred = tf.equal(tf.argmax(onehot_y,1),tf.argmax(pred,1))
    accuracy = tf.reduce_mean(tf.cast(correct_pred,dtype=tf.float32))
    tf.summary.scalar('loss',loss)
    tf.summary.histogram('corrected pred',tf.cast(correct_pred,tf.float32))
    tf.summary.scalar('acc',accuracy)

#写入summary
summary_dir='./summary/'
merge = tf.summary.merge_all()
summary_writer = tf.summary.FileWriter(summary_dir,sess.graph)

#每1个epoch保存一次模型
saver = tf.train.Saver()
model_save_dir = './saved_model/'

#graph完成，开始训练
init = tf.global_variables_initializer()
num_batches = int(len(train_x)/batch_size)
sess.run(init)
for epoch in range(10):
    total_cost =0.
    total_acc=0.
    for batch in range(num_batches):
        # batch_x, batch_y = mnist.train.next_batch(batch_size)
        # batch_x = batch_x.reshape([-1, num_steps, input_size])
        batch_x = train_x[batch*batch_size:(batch+1)*batch_size]
        batch_y = train_y[batch*batch_size:(batch+1)*batch_size]
        x_lenth = train_lenth[batch*batch_size:(batch+1)*batch_size]
        _,acc,cost,summary,global_steps = sess.run([optimizer,accuracy,loss,merge,global_step],feed_dict={x:batch_x,y:batch_y,seq_lenth:x_lenth})
        #_, acc, cost,out = sess.run([optimizer, accuracy, loss,output], feed_dict={x: batch_x, y: batch_y})
        total_cost+=cost
        total_acc+=acc
        summary_writer.add_summary(summary,global_step=global_steps)
        print '==========================================='
        print 'epoch:%d batch:%d' % (epoch,batch)
        print 'batch_train_acc:%.4f batch_train_loss:%.4f ' % (acc,cost)
        print 'average_train_acc:%.4f average_train_loss:%.4f ' % (total_acc/(batch+1),total_cost/(batch+1))
        print '==========================================='
    if epoch %1  ==0:
        print '==========================================='
        print 'epoch:%d  train_acc:%.4f train_loss:%.4f '%(epoch,total_acc,total_cost/num_batches)
        print '==========================================='
        saver.save(sess,model_save_dir+'model.ckpt',global_step=epoch)
    summary_writer.close()
print 'optimizer finished! '
#选500个样本进行测试
#
# test_acc,test_cost = sess.run([accuracy,loss],feed_dict={x:test_x[:batch_size],y:test_y[:batch_size],seq_lenth:test_lenth[:batch_size]})
# print 'test_acc:%.4f test_loss:%.4f'%(test_acc,test_cost)