import librosa
import os
import csv

#load music
audio_path = '/kaggle/input/gtzan-dataset-music-genre-classification/Data/genres_original'

def get_d(filepath,savepath):                            #创建信息提取函数，输入分别为包含路径的音频、保存文件的名称
    y,sr = librosa.load(filepath)                           #输出为采样率为22050的单声道从0.0秒开始
    k = [sr]+ y.tolist()                                    #输出格式为第一位为采样频率后面为时间序列的列表
    
    with open(savepath,'w',newline='') as t:                #numline是来控制空的行数的
        writer=csv.writer(t)                                #这一步是创建一个csv的写入器
        writer.writerow(k)                                  #写入样本数据

for root, dirs, files in os.walk(audio_path):                       #读取音频文件并提取、保存其中信息
    for name in files:                                              #遍历文件
        end = name.split('.')
        if end[-1] == 'wav':                                        #判断是否为wav音频文件
            filepath = os.path.join(root,name)                      #获取包含路径的音频名称
            path_list = root.split('/')
            #print(root)
            #break
            i = path_list.index('input')
            path_list[i] ='working'                                  #
            savepath = '/'.join(path_list)+'/'+name.split('.')[0]+name.split('.')[1]+'.csv' #创建包含路径的csv文件名
            if not os.path.exists('/'.join(path_list)):                        #判断保存路径下的文件夹是否存在
                os.makedirs('/'.join(path_list))                    #不存在就创建
            get_d(filepath,savepath)                             #提取信息
            print(name)