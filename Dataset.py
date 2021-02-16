from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from IMLib.utils import *
#import tensorflow.compat.v1 as tf
#tf.disable_v2_behavior()

import tensorflow as tf

class Dataset2(object):

    def __init__(self, config):
        super(Dataset2, self).__init__()

        self.data_dir = config.data_dir
        self.dataset_name = 'NewGazeData'
        self.attr_0_txt = 'eye_train.txt'
        self.attr_1_txt = 'eye_test.txt'
        self.height, self.width = config.img_size, config.img_size
        self.channel = config.output_nc
        self.capacity = config.capacity
        self.batch_size = config.batch_size
        self.num_threads = config.num_threads

        self.train_images_list, self.train_eye_pos, self.test_images_list, self.test_eye_pos, self.test_num = self.readfilenames()

    def readfilenames(self):

        train_eye_pos = []
        train_images_list = []
        fh = open(os.path.join(self.data_dir, self.attr_0_txt))

        for f in fh.readlines():
            eye_pos = []
            f = f.strip('\n')
            filenames = f.split(' ', 5)
            if os.path.exists(os.path.join(self.data_dir, "1/"+filenames[0]+".jpg")):
                train_images_list.append(os.path.join(self.data_dir, "1/"+filenames[0]+".jpg"))
                eye_pos.extend([int(value) for value in filenames[1:5]])
                train_eye_pos.append(eye_pos)

        fh.close()

        fh = open(os.path.join(self.data_dir, self.attr_0_txt))
        test_images_list = []
        test_eye_pos = []

        for f in fh.readlines():
            eye_pos = []
            f = f.strip('\n')
            filenames = f.split(' ', 5)
            if os.path.exists(os.path.join(self.data_dir, "0/"+filenames[0]+".jpg")):
                test_images_list.append(os.path.join(self.data_dir,"0/"+filenames[0]+".jpg"))
                eye_pos.extend([int(value) for value in filenames[1:5]])
                test_eye_pos.append(eye_pos)
                #print test_eye_pos

        fh.close()
        return train_images_list, train_eye_pos, test_images_list, test_eye_pos, len(test_images_list)

    @tf.function
    def read_images(self, img_path, eye_pose):
        
        image = tf.io.read_file(img_path)
        image = tf.image.decode_jpeg(image, channels=self.channel)
        image = tf.cast(image, tf.float32)
        image = tf.image.resize(image, size=(self.height, self.width))
        image = image / 127.5 - 1.0
        
        return image, eye_pose

    @tf.function
    def configure_training_specs(self, ds):
        ds = ds.shuffle(10000)
        #ds = ds.repeat()
        ds = ds.batch(self.batch_size, drop_remainder=True)
        ds = ds.prefetch(tf.data.experimental.AUTOTUNE)
        return ds

    def input(self):
        
        train_images = tf.data.Dataset.from_tensor_slices(self.train_images_list)
        train_eye_pos = tf.data.Dataset.from_tensor_slices(self.train_eye_pos)
        train_data = tf.data.Dataset.zip((train_images, train_eye_pos))
        if self.num_threads == -1:
            train_data = train_data.map(self.read_images, num_parallel_calls=tf.data.experimental.AUTOTUNE)
        else:
            train_data = train_data.map(self.read_images, num_parallel_calls=self.num_threads)
        train_data = self.configure_training_specs(train_data)

        test_images = tf.data.Dataset.from_tensor_slices(self.test_images_list)
        test_eye_pos = tf.data.Dataset.from_tensor_slices(self.test_eye_pos)
        test_data = tf.data.Dataset.zip((test_images, test_eye_pos))
        if self.num_threads == -1:
            test_data = test_data.map(self.read_images, num_parallel_calls=tf.data.experimental.AUTOTUNE)
        else:
            test_data = test_data.map(self.read_images, num_parallel_calls=self.num_threads)
        test_data = self.configure_training_specs(test_data)

        return train_data, test_data

class Dataset(object):

    def __init__(self, config):
        super(Dataset, self).__init__()

        self.data_dir = config.data_dir
        self.dataset_name = 'NewGazeData'
        self.attr_0_txt = 'eye_train.txt'
        self.attr_1_txt = 'eye_test.txt'
        self.height, self.width = config.img_size, config.img_size
        self.channel = config.output_nc
        self.capacity = config.capacity
        self.batch_size = config.batch_size
        self.num_threads = config.num_threads

        self.train_images_list, self.train_eye_pos, self.test_images_list, self.test_eye_pos, self.test_num = self.readfilenames()

    def readfilenames(self):

        train_eye_pos = []
        train_images_list = []
        fh = open(os.path.join(self.data_dir, self.attr_0_txt))

        for f in fh.readlines():
            eye_pos = []
            f = f.strip('\n')
            filenames = f.split(' ', 5)
            if os.path.exists(os.path.join(self.data_dir, "1/"+filenames[0]+".jpg")):
                train_images_list.append(os.path.join(self.data_dir, "1/"+filenames[0]+".jpg"))
                eye_pos.extend([int(value) for value in filenames[1:5]])
                train_eye_pos.append(eye_pos)

        fh.close()

        fh = open(os.path.join(self.data_dir, self.attr_0_txt))
        test_images_list = []
        test_eye_pos = []

        for f in fh.readlines():
            eye_pos = []
            f = f.strip('\n')
            filenames = f.split(' ', 5)
            if os.path.exists(os.path.join(self.data_dir, "0/"+filenames[0]+".jpg")):
                test_images_list.append(os.path.join(self.data_dir,"0/"+filenames[0]+".jpg"))
                eye_pos.extend([int(value) for value in filenames[1:5]])
                test_eye_pos.append(eye_pos)
                #print test_eye_pos

        fh.close()
        return train_images_list, train_eye_pos, test_images_list, test_eye_pos, len(test_images_list)

    @tf.function
    def read_images(self, img_path, eye_pose):
        
        image = tf.io.read_file(img_path)
        image = tf.image.decode_jpeg(image, channels=self.channel)
        image = tf.cast(image, tf.float32)
        image = tf.image.resize(image, size=(self.height, self.width))
        image = image / 127.5 - 1.0
        
        return image, eye_pose

    @tf.function
    def configure_training_specs(self, ds):
        ds = ds.shuffle(10000)
        #ds = ds.repeat()
        ds = ds.batch(self.batch_size, drop_remainder=True)
        ds = ds.prefetch(tf.data.experimental.AUTOTUNE)
        return ds

    def input(self):
        
        images = tf.data.Dataset.from_tensor_slices(self.train_images_list + self.test_images_list)
        eye_pos = tf.data.Dataset.from_tensor_slices(self.train_eye_pos + self.test_eye_pos)
        data = tf.data.Dataset.zip((images, eye_pos))
        if self.num_threads == -1:
            data = data.map(self.read_images, num_parallel_calls=tf.data.experimental.AUTOTUNE)
        else:
            data = data.map(self.read_images, num_parallel_calls=self.num_threads)

        train_samples = len(data) - 1000
        train_data = data.take(train_samples)
        test_data = data.skip(train_samples)

        train_data = self.configure_training_specs(train_data)
        test_data = self.configure_training_specs(test_data)

        return train_data, test_data

if __name__ == "__main__":
    from config.train_options import TrainOptions

    opt = TrainOptions().parse()

    dataset = Dataset2(opt)
    train_data, test_data = dataset.input()
    
    for x, y in train_data.take(1):
        print(x.shape, y.shape)
        print(x)
        print(y)