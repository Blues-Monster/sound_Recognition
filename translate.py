
import os


def mp3_to_wav(filepath, savedir):
    filename = filepath.replace('.mp3', '.wav')
    savefilename = filename.split('/')
    save_dir = savedir + '/' + savefilename[-1]
    print(save_dir)
    cmd = 'ffmpeg.exe -i ' + filepath + ' ' + save_dir
    os.system(cmd)

def flac_to_wav(filepath, savedir):
    filename = filepath.replace('.flac', '.wav')
    savefilename = filename.split('/')
    save_dir = savedir + '/' + savefilename[-1]
    print(save_dir)
    cmd = 'ffmpeg.exe -i ' + filepath + ' ' + save_dir
    os.system(cmd)


'''
audio_path = r"C:/Users/73714/code/sound_Recognition/music_wav/getvoice.mp3"                #"你的带转换的音频文件路径"
savedir = r"C:/Users/73714/code/sound_Recognition/music_source/music_source"                #"新保存路径"
flac_to_wav(audio_path, savedir)
'''



# 批量处理
savedir = r'../sound_Recognition/music_wav/'
path = r'../sound_Recognition/music_source/'                               #'你的音频文件夹路径'
for root, dirs, files in os.walk(path):
    for name in files:
        filepath = root + "/" + name
        if filepath.split('.')[-1] == "flac":
            flac_to_wav(filepath, savedir)
        if filepath.split('.')[-1] == "mp3":
            mp3_to_wav(filepath, savedir)
