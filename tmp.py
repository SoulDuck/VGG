import tensorflow as tf
import matplotlib.pyplot as plt
global_step =tf.placeholder(tf.int32 , name= 'global_step')
lr = tf.train.exponential_decay(0.1 , global_step , 10000 ,0.96 , staircase=False)
sess=tf.Session()
init=tf.global_variables_initializer()
sess.run(init)
xs=[]
for i in range(100000):
    learning_rate=sess.run(lr , feed_dict={global_step:i})
    xs.append(learning_rate)

plt.plot(range(100000) ,xs )
plt.show()