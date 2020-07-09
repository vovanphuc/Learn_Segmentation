import os.path as osp



def make_datapath_list(rootpath): #chứa các đường link dẫn đến các file ảnh và các file s
    original_image_template = osp.join(rootpath, 'JPEGImages', '%s.jpg')
    annotation_image_template = osp.join(rootpath, 'SegmentationClass', '%s.png')

    #train, val

    train_ids = osp.join(rootpath, 'ImageSets/Segmentation/train.txt')
    val_ids = osp.join(rootpath, 'ImageSets/Segmentation/val.txt')

    train_img_list = list()
    train_anno_list = list()

    for line in open(train_ids):
        img_id = line.strip() # xoa khoang trong
        img_path = (original_image_template % img_id)
        anno_path = (annotation_image_template % img_id)

        train_img_list.append(img_path)
        train_anno_list.append(anno_path)

    val_img_list = list()
    val_anno_list = list()

    for line in open(val_ids):
        img_id = line.strip() # xoa khoang trong
        img_path = (original_image_template % img_id)
        anno_path = (annotation_image_template % img_id)

        val_img_list.append(img_path)
        val_anno_list.append(anno_path)

    return train_img_list, train_anno_list, val_img_list, val_anno_list

class DataTransform():
    def __init__(self, input_size, color_mean, color_std):
        self.data_transform = {
            "train" : Compose([
                Scale(scale=[0.5, 1.5]),
                RandomRotation(angle =[-10, 10]),
                RandomMirror(),
                Reszie(input_size),
                Normalize_Tensor(color_mean, color_std)
            ]),
            "val" : Compose([
                Reszie(input_size),
                Normalize_Tensor(color_mean, color_std)
            ])
        }
    def __call__(self, phase, img, anno_class_img):
        return self.data_transform[phase](img, anno_class_img)


if __name__ == "__main__" :

    rootpath = './data/VOC/VOC2012/'
    train_img_list, train_anno_list, val_img_list, val_anno_list = make_datapath_list(rootpath = rootpath)

    print(len(train_img_list))
    print(len(val_img_list))

    print(train_img_list[0]) # path
    print(train_anno_list[0])


