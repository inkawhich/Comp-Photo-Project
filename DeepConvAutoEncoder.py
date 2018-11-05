# Deep Convolutional Autoencoder for Image Denoising
import glob

import tensorflow as tf



x = tf.placeholder(tf.float32,shape=[None,224,224,3],name='InputImage')
y = tf.placeholder(tf.float32,shape=[None,224,224,3],name='OutputImage')

### Build up Encoder
conv1 = tf.layers.conv2d(inputs=x, filters=32, kernel_size=(3,3), padding='same', activation=tf.nn.relu)
# 224*224*32
maxpool1 = tf.layers.max_pooling2d(inputs=conv1,pool_size=(2,2),strides=(2,2),padding='same')
# 112*112*32
conv2 = tf.layers.conv2d(inputs=maxpool1,filters=32,kernel_size=(3,3),padding='same',activation=tf.nn.relu)
# 112*112*32
maxpool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=(2,2), strides=(2,2), padding='same')
# 56*56*32
conv3 = tf.layers.conv2d(inputs=maxpool2,filters=16,kernel_size=(3,3),padding='same',activation=tf.nn.relu)
# 56*56*16
encoded = tf.layers.max_pooling2d(inputs=conv3, pool_size=(2,2), strides=(2,2), padding='same')
# 28*28*16

### Build up Decoder
upsample1 = tf.image.resize_nearest_neighbor(images=encoded,size=(56,56))
# 56*56*16
conv4 = tf.layers.conv2d(inputs=upsample1,filters=16,kernel_size=(3,3),padding='same',activation=tf.nn.relu)
# 56*56*16
upsample2 = tf.image.resize_nearest_neighbor(images=conv4,size=(112,112))
# 112*112*16
conv5 = tf.layers.conv2d(inputs=upsample2,filters=32,kernel_size=(3,3),padding='same',activation=tf.nn.relu)
# 112*112*32
upsample3 = tf.image.resize_nearest_neighbor(images=conv5,size=(224,224))
# 224*224*32
conv6 = tf.layers.conv2d(inputs=upsample3,filters=32,kernel_size=(3,3),padding='same',activation=tf.nn.relu)
# 224*224*32
decoded = tf.layers.conv2d(conv6, 3, (3,3), padding='same', activation=tf.nn.relu, name='decoded')
# 224*224*3

loss = tf.square(y-decoded)
cost = tf.reduce_mean(loss)
opt = tf.train.AdamOptimizer(learning_rate=0.001).minimize(cost)

saver = tf.train.Saver()
# Training
with tf.device('/device:GPU:0'):
    with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
        epochs = 20
        sess.run(tf.global_variables_initializer())
        dark_img = []
        dark_img_msr = []
        dark_img = glob.glob('./Dark-Image-Data/*/*')
        dark_img_msr = glob.glob('./Dark-Image-Data_lime/*/*')
        for e in range(epochs):
            for index, file in enumerate(dark_img):
                dark_image = tf.gfile.FastGFile(file,'rb').read()
                if file.endswith('.jpg'):
                    dark_image = tf.image.decode_jpeg(dark_image,channels=3)
                elif file.endswith('.png'):
                    dark_image = tf.image.decode_png(dark_image,channels=3)
                print file
                if type(dark_image)=='str':
                    print type(dark_image)
                    continue
                if dark_image.get_shape()[2]!=3:
                    continue
                #dark_image = tf.image.resize_images(dark_image,size=[224,224]).eval().reshape(1,224,224,3)
                dark_image = tf.cast(tf.image.resize_images(dark_image,size=[224,224]),tf.float32)
                dark_image = tf.expand_dims(dark_image,0).eval()
                light_image = tf.gfile.FastGFile(dark_img_msr[index],'rb').read()
                if dark_img_msr[index].endswith('.jpg'):
                    light_image = tf.image.decode_jpeg(light_image,channels=3)
                elif file.endswith('.png'):
                    light_image = tf.image.decode_png(light_image,channels=3)
                if type(light_image)=='str':
                    continue
                if light_image.get_shape()[2]!=3:
                    continue
                light_image = tf.cast(tf.image.resize_images(light_image,size=[224,224]),tf.float32)
                light_image = tf.expand_dims(light_image,0).eval()
                batch_cost, batch_opt =sess.run(fetches=[cost,opt],feed_dict={x:dark_image, y:light_image})
                print("Epoch: {}/{}".format(e + 1, epochs),"Training loss: {:.4f}".format(batch_cost))
            writer = tf.summary.FileWriter('./log/', sess.graph)

        saver.save(sess,'./model/msr_autoencoder')
