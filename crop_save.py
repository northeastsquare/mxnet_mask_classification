import os
import xml.etree.ElementTree as ET
import cv2

train_path = '/home/silva/work/mask/train'
test_path = '/home/silva/work/mask/val'
save_path = '/home/silva/work/mask/crops'
def cs(dpath):#crop, save
    for root, d, files in os.walk(dpath):
        for f in files:
            fn, ext = os.path.splitext(f)
            if ext != ".xml":
                continue
            xmlfn = os.path.join(root, f)
            tree = ET.parse(xmlfn)
            objs = tree.findall('object')
            img_fn = os.path.join(root, fn+'.jpg')
            img = cv2.imread(img_fn)
            for idx, obj in enumerate(objs):
                cls_name = obj.find('name').text
                bbox = obj.find('bndbox')
                x0 = int(bbox.find('xmin').text)
                y0 = int(bbox.find('ymin').text)
                x1 = int(bbox.find('xmax').text)
                y1 = int(bbox.find('ymax').text)
                crop_img = img[y0:y1+1, x0:x1+1, :]
                dname = os.path.join(save_path, cls_name)
                if not os.path.exists(dname):
                    os.makedirs(dname)
                save_name = os.path.join(dname, fn+'_'+str(idx)+'.jpg')
                cv2.imwrite(save_name, crop_img)
cs(train_path)
cs(test_path)



