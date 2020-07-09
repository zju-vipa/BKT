import os
import pdb
from scipy import io as sio

class BaseData():
    def __init__(self, is_train,data_dir, archive=True):
        self.is_train = is_train
        self.data_dir = data_dir
        self.archive = archive
        
        self.preprocess()

        self.data = self.get_image_label_pairs()

        self.postprocess()

    def preprocess(self):
        pass
    
    def get_image_label_pairs(self):
        return None

    def postprocess(self):

        if self.archive is True:
            archive_name = "train_data.txt" if self.is_train else "val_data.txt"
            with open(os.path.join(self.data_dir,archive_name),"w",encoding="utf-8") as f:
                for  key, value in self.data:
                    f.write(key + " " + str(int(value)-1) + "\n")


### dataset: CUB_200_2011, different types of birds


class CUB_200(BaseData):
    def __init__(self,is_train, data_dir,archive=True):
        
        self.class_file = os.path.join(data_dir,"classes.txt") # <class_id> <class_name>
        self.image_class_labels_file = os.path.join(data_dir,"image_class_labels.txt") # <image_id> <class_id>
        self.images_file = os.path.join(data_dir,"images.txt") # <image_id> <image_name>(local)
        self.train_test_split_file = os.path.join(data_dir, "train_test_split.txt") #<image_id> <is_training_image>
        self.bounding_boxes_file = os.path.join(data_dir, "bounding_boxes.txt")  #<image_id> <x> <y> <width> <height>
        
        super(CUB_200,self).__init__(is_train,data_dir,archive)

    def preprocess(self):

        self.id_label = {}
        self.id_image = {}
        with open(self.image_class_labels_file,"r") as f:
            for line in f:
                image_id, class_id = line.split()
                self.id_label[image_id] = class_id

        with open(self.images_file,"r") as f:
            for line in f:
                image_id,image_name = line.split()
                self.id_image[image_id] = image_name

        #return  id_label, id_image

    def get_image_label_pairs(self):

        #id_label, id_image = self.get_image_label()

        train_set = []
        test_set = []

        with open(self.train_test_split_file,"r") as  f:
            for line in f: 
                image_id, is_train = line.split()
                if int(is_train):
                    train_set.append([self.id_image[image_id],self.id_label[image_id]])
                else:
                    test_set.append([self.id_image[image_id],self.id_label[image_id]])
        
        return train_set if self.is_train else  test_set

class StanfordDog(BaseData):
    def __init__(self,is_train, data_dir, archive=True):
        super(StanfordDog,self).__init__(is_train,data_dir,archive)

    def get_image_label_pairs(self):

        filename = "train_list.mat" if self.is_train else "test_list.mat"
        data = sio.loadmat(os.path.join(self.data_dir,filename))
        images = data["file_list"]
        labels = data["labels"]
        images = list(map(lambda x: x[0][0], images))
        labels = list(map(lambda x: x[0], labels))

        return list(zip(images, labels))


if __name__ == "__main__":
    root = "/home/clc/data/cub_200_2011/CUB_200_2011/"
    data = CUB_200(is_train=False,data_dir=root,archive=True)
   # import pdb
   # pdb.set_trace()

    for i, (key, value) in enumerate(data.data):
        print("{}----> {} ,  {} \n".format(i, key, value))
