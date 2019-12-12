import numpy as np

class metrics():
    def __init__(self, batchsize):
        self.tp0, self.fp0, self.fn0 = 0, 0, 0
        self.tp1, self.fp1, self.fn1 = 0, 0, 0
        self.num_of_call_accumulate = 0
        self.batchsize = batchsize

    def accumulate(self, predict, label):
        '''
        accumulate true positives, false negatives, false postives
        for both classes
        :param predict: model predicted outputs
        :param label:   corresponding labels
        :return:
        '''
        self.num_of_call_accumulate += 1
        predict, label = np.squeeze(predict, 1), np.squeeze(label, 1)
        for i in range(self.batchsize):
            if label[i] >= 0.5:
                if predict[i] >= 0.5:
                    self.tp1 += 1
                else:
                    self.fn1 += 1
                    self.fp0 += 1
            else:
                if predict[i] < 0.5:
                    self.tp0 += 1
                else:
                    self.fn0 += 1
                    self.fp1 += 1

    def get_acc(self):
        # acc = sum of true positives (both classes) / num of samples
        return (self.tp0+self.tp1)/(self.num_of_call_accumulate*self.batchsize)

    def get_precision(self, class_index):
        if class_index:
            return np.divide(self.tp1, (self.tp1 + self.fp1))
        else:
            return np.divide(self.tp0, (self.tp0 + self.fp0))

    def get_recall(self, class_index):
        if class_index:
            return np.divide(self.tp1, (self.tp1 + self.fn1))
        else:
            return np.divide(self.tp0, (self.tp0 + self.fn0))

    def get_mAP(self):
        return np.mean([np.divide(self.tp0, (self.tp0 + self.fp0)),
                        np.divide(self.tp1, (self.tp1 + self.fp1))])

if __name__=='__main__':
    test = metrics(batchsize=4)
    test.accumulate([[0.7], [0.2], [0.8], [0.1]],
                    [[1], [0], [1], [0]])
    test.accumulate([[0.7], [0.2], [0.8], [0.9]],
                    [[1], [0], [1], [1]])
    test.accumulate([[0.8], [0.6], [0.2], [0.9]],
                    [[1], [1], [0], [1]])
    print('total acc:\t', test.get_acc())
    print('class 0 precision:\t', test.get_precision(0))
    print('class 1 precision:\t', test.get_precision(1))
    print('class 0 recall:\t', test.get_recall(0))
    print('class 1 recall:\t', test.get_recall(1))
    print('mAP:\t', test.get_mAP())
