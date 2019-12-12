from sklearn.model_selection import train_test_split
import os, math, numpy as np

class io_handler():
    def __init__(self, batchsize, data_path, class0_dir_name, class1_dir_name):
        assert batchsize % 2 == 0, 'batchsize has to be divisible by 2'
        self.batchsize = batchsize
        self.data_path = data_path
        self.class0_dir_name = class0_dir_name
        self.class1_dir_name = class1_dir_name
        self.input_mean = 0.004255355
        self.input_std = 0.0039536413

    def get_train_val_names(self):
        class0_names = os.listdir(os.path.join(self.data_path, self.class0_dir_name))
        class0_labels = np.zeros(np.shape(class0_names))
        class1_names = os.listdir(os.path.join(self.data_path, self.class1_dir_name))
        class1_labels = np.ones(np.shape(class1_names))
        class0_names.extend(class1_names)
        inputs = np.array(class0_names)
        labels = np.concatenate((class0_labels, class1_labels), axis=0)
        X_train, X_test, y_train, y_test = train_test_split(inputs, labels, test_size=0.15, random_state=1)
        return X_train, X_test, y_train, y_test

    def sigmoid(self, x):
        return 1/(1+math.exp(-x))

    def load_image_label_batch(self, iter, x_names, y_labels):
        input_batch = []
        label_batch = []
        for i in range(self.batchsize):
            current_label = y_labels[iter + i]
            if current_label:
                with open(os.path.join(self.data_path,
                                       self.class1_dir_name,
                                       x_names[iter + i])) as f:
                    content = f.readlines()
            else:
                with open(os.path.join(self.data_path,
                                       self.class0_dir_name,
                                       x_names[iter + i])) as f:
                    content = f.readlines()
            # current_input = [(np.float32(j.split('\t')[1][:-1])-self.input_mean)/self.input_std for j in content]
            current_input = [np.float32(j.split('\t')[1][:-1]) for j in content]
            current_input = np.expand_dims(current_input, -1)
            input_batch.append(current_input)
            label_batch.append(current_label)

        input_batch = np.array(input_batch)
        input_batch = (input_batch - np.mean(input_batch))/np.std(input_batch)
        label_batch = np.expand_dims(np.array(label_batch), -1)
        return input_batch, label_batch

    def get_inputs_whole_array(self):
        inputs_data = []
        class0_names = os.listdir(os.path.join(self.data_path, self.class0_dir_name))
        class0_labels = np.zeros(np.shape(class0_names))
        class1_names = os.listdir(os.path.join(self.data_path, self.class1_dir_name))
        class1_labels = np.ones(np.shape(class1_names))
        class0_names.extend(class1_names)
        inputs = np.array(class0_names)
        labels = np.concatenate((class0_labels, class1_labels), axis=0)
        for i in range(len(inputs)):
            if labels[i]:
                with open(os.path.join(self.data_path,
                                       self.class1_dir_name,
                                       inputs[i])) as f:
                    content = f.readlines()
            else:
                with open(os.path.join(self.data_path,
                                       self.class0_dir_name,
                                       inputs[i])) as f:
                    content = f.readlines()
            current_input = [np.float32(j.split('\t')[1][:-1]) for j in content]
            inputs_data.append(current_input)

        return inputs_data

if __name__=='__main__':
    io = io_handler(16,
                    '../strawberries',
                    'negatives',
                    'positives')
    inputs_data = io.get_inputs_whole_array()
    print(np.std(inputs_data), np.mean(inputs_data))
    print(np.max(inputs_data), np.min(inputs_data))