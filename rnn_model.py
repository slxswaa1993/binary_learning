#-*- coding=utf-8 -*-
from sklearn.model_selection import train_test_split
import pickle
import gc
import numpy as np
import tensorflow as tf

#定义一个RNN类，关于RNN需要定义的参数有:rnn神经元数num_states,输入rnn的序列长度num_steps,网络层数num_layers,
class RNN_model:
    #num_steps:输入sequence的长度,num_states:RNN每一层神经元个数,num_layers:RNN层数，与num_states对应。
    #input_dims:输入sequence每一个输入的维度,n_classes:预测的类别数,sequence_lenth是每个输入的实际长度,大于实际长度的数据中已经用0padding
    def __init__(self,num_steps,input_dims,n_classes,num_states=[32,32,64],num_layers=3,learning_rate=0.001,num_epochs=10):
        self.num_steps = num_steps
        self.num_states = num_states
        self.num_layers = num_layers
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs
        self.input_dims = input_dims
        self.n_classes = n_classes

    def build_graph(self,summary_dir,input_is_one_hot,batch_size):
        self.batch_size =batch_size
        #build input
        with tf.name_scope('input'):
            self.input_x = tf.placeholder(dtype=tf.float32,shape=[None,self.num_steps,self.input_dims])
            self.x_lenth = tf.placeholder(dtype=tf.int32,shape=[None])
            if input_is_one_hot:
                self.input_y = tf.placeholder(dtype=tf.int32,shape=[None,self.n_classes])
                self.y = self.input_y
            else:
                self.input_y = tf.placeholder(dtype=tf.int32,shape=[None])
                self.y = tf.one_hot(self.input_y,depth=self.n_classes)
            tf.summary.histogram('x',self.input_x)
            tf.summary.histogram('y',self.input_y)
            tf.summary.histogram('lenth',self.x_lenth)

        #构建RNN层，现在输入的数据shape为[batch_size,num_steps,input_dims]
        #RNN 第i层state的shape是[batch_size,num_states[i]]
        with tf.name_scope('RNN_CELL'):
            self.cell = tf.contrib.rnn.MultiRNNCell([tf.contrib.rnn.BasicLSTMCell(self.num_states[i]) for i in range(self.num_layers)])
            self.init_states = self.cell.zero_state(batch_size=self.batch_size,dtype=tf.float32)
            self.outputs,self.states = tf.nn.dynamic_rnn(self.cell,self.input_x,initial_state=self.init_states,sequence_length=self.x_lenth)
        #此时,batch中的每一个sequence长度不同，超过长度的output都被置为0，超过长度的state都是最后一个有效的state
        #因此要取出实际有效的输出,此时outputs,states的shape[batch_size,num_steps,num_states[-1]]
            valid_outputs = []
            for i in range(self.batch_size):
                output = self.outputs[i,self.x_lenth[i]-1,:]
                output = tf.reshape(output,shape=[1,-1])
                valid_outputs.append(output)
            self.valid_outputs = tf.concat(valid_outputs,0)
            tf.summary.histogram('rnn_outputs',self.valid_outputs)

        #现在要接入一个softmax层进行分类预测
        with tf.name_scope('soft_max'):
            self.weight = tf.Variable(tf.truncated_normal(shape=[self.num_states[-1],self.n_classes]))
            self.bias = tf.Variable(tf.truncated_normal(shape=[self.n_classes]))
            self.pred = tf.matmul(self.valid_outputs,self.weight)+self.bias

        #计算loss,开始训练,加入global_step方便写入summary计步
        with tf.name_scope('train'):
            self.global_step = tf.Variable(0,trainable=False)
            self.loss = tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits(labels=self.y,logits=self.pred))
            self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.loss,global_step=self.global_step)
            self.correct_pred = tf.cast(tf.equal(tf.argmax(self.y,1),tf.argmax(self.pred,1)),dtype=tf.float32)
            self.accuracy = tf.reduce_mean(self.correct_pred)
            tf.summary.histogram('corrected',self.correct_pred)
            tf.summary.scalar('loss',self.loss)
            tf.summary.scalar('accuracy',self.accuracy)

        #创建saver和summary写入
        with tf.name_scope('save_and_summary'):
            self.saver = tf.train.Saver()
            self.merge = tf.summary.merge_all()
            self.sess = tf.Session()
            self.summary_writer = tf.summary.FileWriter(summary_dir,graph=tf.get_default_graph())


    #data的shape是[num_steps,input_dims]
    def train(self,train_data,train_label,train_lenth,model_dir):
        self.sess.run(tf.global_variables_initializer())
        num_batches = int(len(train_data)/self.batch_size)
        for epoch in range(self.num_epochs):
            total_acc = 0.
            total_loss =0.
            for batch in range(num_batches):
                batch_x = train_data[batch*self.batch_size:(batch+1)*self.batch_size]
                batch_y = train_label[batch*self.batch_size:(batch+1)*self.batch_size]
                batch_lenth = train_lenth[batch*self.batch_size:(batch+1)*self.batch_size]
                feed_dict = {self.input_x:batch_x,self.input_y:batch_y,self.x_lenth:batch_lenth}
                _,batch_loss,batch_acc,summary,global_step,output =self.sess.run([self.optimizer,self.loss,self.accuracy,self.merge,self.global_step,self.valid_outputs],feed_dict=feed_dict)
                total_acc += batch_acc
                total_loss += batch_loss
                self.summary_writer.add_summary(summary,global_step=global_step)
                self.summary_writer.flush()
                print '==========================================='
                print 'epoch:%d batch:%d' % (epoch, batch)
                print 'batch_train_acc:%.4f batch_train_loss:%.4f ' % (batch_acc, batch_loss)
                print 'average_train_acc:%.4f average_train_loss:%.4f ' % (total_acc / (batch + 1), total_loss / (batch + 1))
                print '==========================================='

            if epoch %1 == 0:
                print 'Saving model @ epoch %d'%(epoch)
                self.saver.save(self.sess,save_path=model_dir+'model.ckpt',global_step=epoch)
        print 'OPTIMIZER FINISHED ! ! '
        self.summary_writer.close()

    #如果模型存在，载入模型
    def load_model(self,model_dir):
        self.build_graph()
        self.saver.restore(self.sess)

    #定义测试方法
    def evaluate(self,test_data,test_label,test_lenth):
        num_batches = int(len(test_data)/self.batch_size)
        total_acc = 0.
        total_loss = 0.
        for batch in range(num_batches):
            batch_x = test_data[batch*self.batch_size:(batch+1)*self.batch_size]
            batch_y = test_label[batch*self.batch_size:(batch+1)*self.batch_size]
            batch_lenth = test_lenth[batch*self.batch_size:(batch+1)*self.batch_size]
            feed_dict = {self.input_x:batch_x,self.input_y:batch_y,self.x_lenth:batch_lenth}
            batch_loss,batch_acc = self.sess.run([self.loss,self.accuracy],feed_dict=feed_dict)
            total_loss += batch_loss
            total_acc += batch_acc
        print "test accuracy : %.4f  test loss :%.4f"%(total_acc,total_loss)



if __name__ == '__main__':
    print 'Loading Dataset ......'
    data_file = open('./data/train_data.pkl', mode='r')
    label_file = open('./data/train_label.pkl', mode='r')
    lenth_file = open('./data/data_lenth.pkl', mode='r')
    data_set = pickle.load(data_file)
    label_set = pickle.load(label_file)
    data_lenth = pickle.load(lenth_file)
    print 'successfully load dataset !'
    train_x, test_x, train_y, test_y, train_lenth, test_lenth = train_test_split(data_set, label_set, data_lenth,
                                                                                 test_size=0.3, shuffle=True)
    # 回收不用的内存
    del data_set, label_set, data_lenth
    gc.collect()

    model = RNN_model(num_steps=19470,n_classes=104,input_dims=8,num_states=[32,32,64],num_layers=3,learning_rate=0.001,num_epochs=10)
    model.build_graph(batch_size=128,summary_dir='./summary/',input_is_one_hot=False)
    model.train(train_data=train_x,train_label=train_y,train_lenth=train_lenth,model_dir='./rnn_model/')












