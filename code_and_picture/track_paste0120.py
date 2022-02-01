import numpy as np
import cv2

cap = cv2.VideoCapture(0) #カメラを定義

def red_range(img): #赤色の領域をマスクする関数
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV) #BGRをHSV色空間に変換

#赤色の領域を閾値でフィルタリング
#OpenCVのHSV色空間では赤は0~30付近と150~180付近に分かれるが、
#181~255は0からの循環なので180を中心に範囲を取れば赤の閾値を一回で指定できる。たぶん。
    hsv_min = np.array([170,170,60]) #色相(Hue)、彩度(Saturation)、明度(Value)
    hsv_max = np.array([190,255,255])
    mask = cv2.inRange(hsv, hsv_min, hsv_max) #hsvの各ドットについてhsv_minからhsv_maxの範囲内ならtrue

    return mask

#カメラ映像の幅などを取得するコード
# 幅
vidwide = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
# 高さ
vidhit = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
# 総フレーム数
count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
# fps
fps = cap.get(cv2.CAP_PROP_FPS)

#画像を張り付けるコード
def merge_images(bg, fg_alpha, s_x, s_y):
    alpha = fg_alpha[:,:,3]  # アルファチャンネルだけ抜き出す(要は2値のマスク画像)
    alpha = cv2.cvtColor(alpha, cv2.COLOR_GRAY2BGR) # grayをBGRに
    alpha = alpha / 255.0    # 0.0〜1.0の値に変換

    fg = fg_alpha[:,:,:3]

    f_h, f_w, _ = fg.shape # アルファ画像の高さと幅を取得
    b_h, b_w, _ = bg.shape  # 背景画像の高さを幅を取得
    #f_h, f_w, _ = b_h, b_w, _
    #print(f"fh:{f_h}\nfw:{f_w}\nbh:{b_h}\nbw:{b_w}")

   
    
    bg[s_y:f_h+s_y, s_x:f_w+s_x] = (bg[s_y:f_h+s_y, s_x:f_w+s_x] * (1.0 - alpha)).astype('uint8') # アルファ以外の部分を黒で合成
    bg[s_y:f_h+s_y, s_x:f_w+s_x] = (bg[s_y:f_h+s_y, s_x:f_w+s_x] + (fg * alpha)).astype('uint8')  # 合成
    return bg



"""
def blue_range(img): #青色の領域をマスクする関数
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV) #BGRをHSV色空間に変換

#青色の領域を閾値でフィルタリング
#OpenCVのHSV色空間では赤は0~30付近と150~180付近に分かれるが、
#181~255は0からの循環なので180を中心に範囲を取れば赤の閾値を一回で指定できる。たぶん。
    hsv_min = np.array([180,170,60]) #色相(Hue)、彩度(Saturation)、明度(Value)
    hsv_max = np.array([300,255,255])
    mask = cv2.inRange(hsv, hsv_min, hsv_max) #hsvの各ドットについてhsv_minからhsv_maxの範囲内ならtrue

    return mask

def green_range(img): #緑色の領域をマスクする関数
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV) #BGRをHSV色空間に変換

#緑色の領域を閾値でフィルタリング
#OpenCVのHSV色空間では赤は0~30付近と150~180付近に分かれるが、
#181~255は0からの循環なので180を中心に範囲を取れば赤の閾値を一回で指定できる。たぶん。
#色相(Hue)、彩度(Saturation)、明度(Value)
    hsv_min = np.array([90, 64, 0])
    hsv_max = np.array([150,255,255])

    mask = cv2.inRange(hsv, hsv_min, hsv_max) #hsvの各ドットについてhsv_minからhsv_maxの範囲内ならtrue

    return mask
"""
#icondata = "code_and_picture/ojigi_animal_inu.png"
#icondata="code_and_picture\\198.jpg"
icondata = "code_and_picture/daikiti.jpg"

while( cap.isOpened() ): #カメラが使える限りループ

    ret, frame = cap.read() #カメラの情報を取得。frameに640x480x3の配列データが入る。
    frame_np = red_range(np.array(frame)) #frameデータをnp配列に変換。
    

#領域のカタマリである「ブロブ」を識別し、データを格納する。すごくありがたい機能。
    nLabels, labelimages, data, center = cv2.connectedComponentsWithStats(frame_np)

    blob_count = nLabels - 1 #ブロブの数。画面領域全体を1つのブロブとしてカウントするので、-1する。1以上で色を認識

    if blob_count >= 1: #ブロブが1つ以上存在すれば、画面全体を示すブロブデータを削除。
        data = np.delete(data, 0, 0)
        center = np.delete(center, 0, 0)

#認識したブロブの中で最大領域を持つもののインデックスを取得
    tbi = np.argmax(data[:, 4]) #target_blob_index

#最大ブロブの中心に青の円マークを直径10pix、太さ3pixで描く
    #cv2.circle(frame,(int(center[tbi][0]),int(center[tbi][1])),10,(255,0,0),3)
    
#座標が中心でない時に最大ブロブの中心に画像を貼り付ける
    #zahyo = ((int(center[tbi][0]), int(center[tbi][1])))
    zahyo=(160,200)
    #zahyo=(0,0)
    #カメラ映像の幅などを取得するコード
# 幅
    vidwide = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
# 高さ
    vidhit = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    try:
        if blob_count>=1:
            icon = cv2.imread(icondata, -1)  # 画像読み込み
            #frame = merge_images(frame, icon, int(center[tbi][0]), int(center[tbi][1]))  # 画像を中心に貼り付け
            icon=cv2.resize(icon,dsize=(int(vidwide*0.4),int(vidhit*0.3)))#貼り付け画像のサイズ変更
            #x, y = int(center[tbi][0]), int(center[tbi][1])#トラッキング
            x,y=zahyo
            frame[y:icon.shape[0] + y, x:icon.shape[1] + x] = icon[:icon.shape[0], :icon.shape[1], :3]
        #cv2.imshow('RaspiCam_Live', frame)
    except :
        ret,frame=cap.read()


#画像を表示する&画面を移動
    cv2.imshow('RaspiCam_Live',frame)
    cv2.moveWindow('RaspiCam_Live', 320, 240)

#キーが押されたら終了する
    if cv2.waitKey(1) != -1:
        break

#終了処理。カメラを解放し、表示ウィンドウを破棄。
cap.release()
cv2.destroyAllWindows()
