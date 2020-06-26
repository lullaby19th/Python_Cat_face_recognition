import random
import dlib, cv2, os
import pandas as pd
import numpy as np

img_size = 224 
# 최종 input image size를 224 x 224 로 setting 함!

# image size를 고정하여 training 하기위해 resize_img 함수를 정의함.
def resize_img(im):
  old_size = im.shape[:2] # old_size is in (height, width) format <- 기존 image 크기
  ratio = float(img_size) / max(old_size)
  new_size = tuple([int(x*ratio) for x in old_size])
  # new_size should be in (width, height) format
  im = cv2.resize(im, (new_size[1], new_size[0])) 
  
  # new_size[1] 은 가로(Width)
  # new_size[0] 은 세로(Height)
  
  delta_w = img_size - new_size[1]
  delta_h = img_size - new_size[0]
  
  # // : 몫을 구하는 연산자 (소숫점 이하를 버리는 연산자)
  top, bottom = delta_h // 2, delta_h - (delta_h // 2)
  left, right = delta_w // 2, delta_w - (delta_w // 2)

  new_im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT,
      value=[0, 0, 0])
  
  # cv2.copyMakeBorder -> 테투리(가장자리)를 만드는 opencv 함수
  # im -> 입력 이미지
  # top, bottom, left, right -> 경계의 폭(해당방향의 픽셀 수)
  # cv2.BORDER_CONSTANT -> 일정한 색상의 테투리를 추가함.
  # value=[0,0,0] -> 테투리 색갈을 설정 (검정색)
  
  return new_im, ratio, top, left
  
for i in range(7):
    
    print(i,'번째 전처리..')
    dirname = 'CAT_0'+str(i)
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
      # sep = ' ' 스페이스바로 구분한다.
      landmarks = (pd_frame.loc[0][1:-1].to_numpy()).reshape((-1, 2))
      # 연산을 쉽게 하기 위해서 9 x 2의 numpy arrray 형태로 만들어준다. 
      
      # load image
      img_filename, ext = os.path.splitext(f)
    
      img = cv2.imread(os.path.join(base_path, img_filename))
    
      # resize image and relocate landmarks
      img, ratio, top, left = resize_img(img) # resize_img 함수를 호출
      landmarks = ((landmarks * ratio) + np.array([left, top])).astype(np.int)
      # 변한 landmark들의 위치를 다시 계산해주는 logic
      
      bb = np.array([np.min(landmarks, axis=0), np.max(landmarks, axis=0)])
      
      # bb -> boundingbox 얼굴의 영역을 지정함
      # np.min -> 좌상단좌표
      # np.max -> 우하단좌표
      
      dataset['imgs'].append(img)
      dataset['lmks'].append(landmarks.flatten())
      dataset['bbs'].append(bb.flatten())
    
      # for l in landmarks:
      #   cv2.circle(img, center=tuple(l), radius=1, color=(255, 255, 255), thickness=2)
    
      # cv2.imshow('img', img)
      # if cv2.waitKey(0) == ord('q'):
      #   break
    
    # dataset 폴더를 만들어 놓은뒤에 np.save 코드 작성
    np.save('E:\\cat_hipsterizer-master\\dataset\\%s.npy' % dirname, np.array(dataset))
