import tensorflow as tf
import numpy as np
import os 
import cv2

class My_Net:
    def __init__(self, layers, shape) -> None:
        self.number_of_layers = layers
        self.input_layer = tf.keras.Input(shape= shape)

    def network_unit(self):
        # cnn unit
        nw = tf.keras.layers.Conv2D(16, kernel_size=(5, 5), activation='relu')(self.input_layer)
        for i in range(self.number_of_layers):
            nw = tf.keras.layers.Conv2D(max(16 - i *4, 4), kernel_size=(5, 5), activation='relu')(nw)
            nw = tf.keras.layers.AvgPool2D(pool_size= (3, 3), strides= 2)(nw)

        # dnn unit
        nw = tf.keras.layers.Flatten()(nw)
        for i in range(self.number_of_layers):
            nw = tf.keras.layers.Dense(max(32 - i*8, 8), activation= 'relu')(nw)
        nw = tf.keras.layers.Dense(8, activation= 'sigmoid')(nw)
        return nw
    
    def build(self):
        output = self.network_unit()
        return tf.keras.Model(self.input_layer, output)

def visual_label(*args):
    """
    use this method to test if the points were correctlly generated
    args : image, points
    """
    assert len(args) <= 2, 'too many variables'
    import matplotlib.pyplot as plt
    _points = np.array(args[1])
    plt.imshow(args[0])
    plt.scatter(_points[:4]*(orinal_shape[0]/SCALER), _points[4:]*(orinal_shape[1]/SCALER), c= 'red')
    plt.show()

class Data_Process:
    """use this class to rectify the dataset
    """
    def __init__(self, train_dir, label_dir, scaler) -> None:
        self.orinal_shape = np.array([0])
        self.files = os.listdir(train_dir)
        self.train_dir = train_dir
        self.label_dir = label_dir
        self.scaler = scaler
        try: 
            self.files.remove('.DS_Store')
        except FileExistsError:
            pass
        self.files.sort()

    def train_data(self):
        """
        training dataset
        """
        self.train_data_x = []
        for file in self.files:
            processed_image = self.image_process(self.train_dir + file).tolist()
            self.train_data_x.append(processed_image)
        return self.train_data_x, self.orinal_shape

    def image_process(self, image_dir):
        """
        change the image to 2d and smaller size, to reduce the memory cost and computation intensity.
        Args:
            image_dir (str): image directory
        Returns:
            matrix: resized gray image
        """
        image = cv2.imread(image_dir)
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        if not self.orinal_shape[0]:
            self.orinal_shape = np.array(np.shape(gray_image))[::-1]
        resized_image = cv2.resize(gray_image, self.orinal_shape//self.scaler)
        return resized_image[:, :, np.newaxis]/255
    
    def label_process(self):
        """ scaling the coordinates to align with resized image

        Returns:
            N*8 matrix: scaled output
        """
        import pandas as pd
        import ast
        import re 
        data = pd.read_csv(self.label_dir)
        data = data['region_shape_attributes']
        modified_data = []
        for i in data:
            i = re.split(':|, ', i)
            label_x = ast.literal_eval(i[2].split('"')[0][:-1])
            label_y = ast.literal_eval(i[3][:-1])
            modified_data.append(np.concatenate([np.array(label_x)/(self.orinal_shape[0])
                                                 , np.array(label_y)/(self.orinal_shape[1])]).tolist())
        return modified_data

if __name__ == "__main__":
    TRAIN = False
    SCALER = 5 #scale down the image size
    path = '/'.join(os.getcwd().split('/')[:-1])
    image_path = path + '/raw4/'  # 'raw4' ; lecture_images
    label_path = path + '/private/via_project_23Jan2024_12h38m_csv.csv'
    dp = Data_Process(image_path, label_path, SCALER)
    data_image, orinal_shape = dp.train_data()
    label = dp.label_process()
    if TRAIN:
        input_shape = np.shape(data_image[0])
        model = My_Net(3, input_shape)
        model = model.build()
        model.summary()
        model.compile(optimizer= tf.keras.optimizers.Adam(), loss= tf.keras.losses.mean_squared_error)
        model.fit(data_image, label, epochs= 250)
        model.save('./cnn_model.h5')
    else:
        model = tf.keras.models.load_model('./cnn_model.h5')
        points = model.predict(data_image)
    visual_label(data_image[4], points[4])

