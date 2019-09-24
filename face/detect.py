#OpenCVのインポート
import cv2,os,re
import datetime

#カスケード型分類器に使用する分類器のデータ（xmlファイル）を読み込み
HAAR_FILE = "/usr/local/Cellar/opencv@3/3.4.5_5/share/OpenCV/haarcascades/haarcascade_frontalface_default.xml"
cascade = cv2.CascadeClassifier(HAAR_FILE)
 
#画像ファイルの読み込み
#image_dir = "/Users/yykozy/yolo/darknet/data/nogizaka/saito"
#image_dir = "/Users/yykozy/yolo/darknet/data/nogizaka/saito"
image_dir = "/Users/yykozy/Downloads/files"
 
imglist = os.listdir(image_dir)

count = 0

for imgfile in imglist:
    if(not re.match('^.+jpg$',imgfile)):
        continue
    print(imgfile)
    imgfilename = image_dir + "/" + imgfile
    img = cv2.imread(imgfilename)
    
    #グレースケールに変換する
    img_g = cv2.imread(imgfilename,0)
    
    #カスケード型分類器を使用して画像ファイルから顔部分を検出する
    face = cascade.detectMultiScale(img_g)
    
    #顔の座標を表示する
    print(face)
    if len(face) == 0:
        print("can't detect")
        continue

    #顔部分を切り取る
    for x,y,w,h in face:
        face_cut = img[y:y+h, x:x+w]
    
    #白枠で顔を囲む
    for x,y,w,h in face:
        cv2.rectangle(img,(x,y),(x+w,y+h),(255,255,255),2)
    
    dt_now = datetime.datetime.now()
    dt_str = dt_now.strftime('%Y%m%d%H%M%S001')

    num = str(count).zfill(5)
    count = count + 1
    #画像の出力
    #[age]_[gender]_[race]_[date&time].jpg
    cv2.imwrite('detect3/20_1_4_%s_%s.jpg' % (dt_str , num), face_cut)
    #cv2.imwrite('rect/%s_rect.jpg' % imgfile, img)