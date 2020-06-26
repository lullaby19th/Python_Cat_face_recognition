import random, sys
import dlib, cv2, os
import pandas as pd
import numpy as np

img_size = 224

# preporcess.py 의 resize_img와 같음.
def resize_img(im):
  old_size = im.shape[:2] # old_size is in (height, width) format
  ratio = float(img_size) / max(old_size)
  new_size = tuple([int(x*ratio) for x in old_size])
  # new_size should be in (width, height) format
  im = cv2.resize(im, (new_size[1], new_size[0]))
  delta_w = img_size - new_size[1]
  delta_h = img_size - new_size[0]
  top, bottom = delta_h // 2, delta_h - (delta_h // 2)
  left, right = delta_w // 2, delta_w - (delta_w // 2)
  new_im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT,
      value=[0, 0, 0])
  return new_im, ratio, top, left
  
for i in range(7):
    
    print(i,'번째 전처리..')
    dirname = 'CAT_0' + str(i)
    base_path = 'E:\\cat_hipsterizer-master\\cat-dataset\\%s' % dirname
    file_list = sorted(os.listdir(base_path))
    random.shuffle(file_list)
    
    dataset = {
      'imgs': [],
      'lmks': [],
      'bbs': []
    }
        
    for f in file_list:
      if '.cat' not in f:
        continue
    
      # read landmarks
      pd_frame = pd.read_csv(os.path.join(base_path, f), sep=' ', header=None)
      landmarks = (pd_frame.loc[0][1:-1].to_numpy()).reshape((-1, 2))
      # 연산을 쉽게 하기 위해서 9 x 2의 numpy arrray 형태로 만들어준다.
      
      bb = np.array([np.min(landmarks, axis=0), np.max(landmarks, axis=0)]).astype(np.int)
      center = np.mean(bb, axis=0)
      
      # 얼굴의 크기를 구함
      # np.max -> 랜드마크의 최대값 
      # np.min -> 랜드마크의 최소값 
      face_size = max(np.abs(np.max(landmarks, axis=0) - np.min(landmarks, axis=0)))
      
      # 바운딩 박스(bb)가 너무 타이트하면, 학습이 잘 안되기 때문에 0.6정도의 마진을 줘서 느슨하게 Crop(자름)
      # 바운딩 박스(bb)를 넘어가더라도 랜드마크(landmarks)를 찾을 수 있도록 함.
      
      new_bb = np.array([
        center - face_size * 0.6,
        center + face_size * 0.6
      ]).astype(np.int)
      
      # np.clip을 통하여 최소값이 0보다 작게 가는것을 방지(음수로 넘어가는 것을 방지함.)
      new_bb = np.clip(new_bb, 0, 99999)
      
      # 고양이의 얼굴부분을 추출한 부분에서 새로운 랜드마크 좌표를 설정함.
      new_landmarks = landmarks - new_bb[0] # 새로운 바운딩 박스(bounding box)의 좌상단의 좌표를 뺌.
      
    
      # load image
      img_filename, ext = os.path.splitext(f)
    
      img = cv2.imread(os.path.join(base_path, img_filename))
      
      # new_img로 이미지를 자름.
      new_img = img[new_bb[0][1]:new_bb[1][1], new_bb[0][0]:new_bb[1][0]]
    
      # resize image and relocate landmarks (이미지를 새로 자른만큼 다시 랜드마크를 재조정.)
      img, ratio, top, left = resize_img(new_img)
      new_landmarks = ((new_landmarks * ratio) + np.array([left, top])).astype(np.int)
    
      dataset['imgs'].append(img)
      dataset['lmks'].append(new_landmarks.flatten())
      dataset['bbs'].append(new_bb.flatten())
    
      # for l in new_landmarks:
      #   cv2.circle(img, center=tuple(l), radius=1, color=(255, 255, 255), thickness=2)
    
      # cv2.imshow('img', img)
      # if cv2.waitKey(0) == ord('q'):
      #   sys.exit(1)
    
    np.save('E:\\cat_hipsterizer-master\\dataset\\lmks_%s.npy' % dirname, np.array(dataset))
