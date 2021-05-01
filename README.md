音乐是我们生活中不可缺少的一部分，不少人都用过网易云、QQ音乐类app，但随着近年版权意识的加强，网上各种音乐纷纷收费或下架，这对于一些用户来说体验并不好，以我个人为例，本人已开始从网上地毯式地下载音乐，并转为同一格式，目前已下载约4800首音乐，但问题也随之而来。本地音乐不能像网上那样模糊搜索，而且音乐流派混乱，对于听歌来说并不方便，故决定开发此项目。

这个项目是为了对音乐流派(包括blues、classical、country、disco、hiphop、jazz、metal、pop、reggae、rock)进行分类而创建，基于python开发。从每个音频中提取出78个特征值，其中：

 - data[i,0] = 过零率(Zero Crossing Rate)
 - data[i,1] = y_harm_mean谐波均值
 - data[i,2] = y_harm_var谐波方差
 - data[i,3] = y_perc_mean感知激波均值
 - data[i,4] = y_perc_var感知激波方差
 - data[i,5] = Tempo BMP (beats per minute)(音乐节拍)
 - data[i,6] = spectral_centroids_mean光谱质心均值
 - data[i,7] = spectral_centroids_var光谱质心方差
 - data[i,8] = spectral_rolloff_mean光谱衰减均值
 - data[i,9] = spectral_rolloff_var光谱衰减方差
 - data[i,10-29] = mfccs_mean#Mel-Frequency Cepstral Coefficients(梅尔频率倒谱系数)均值
 - data[i,30-49] = mfccs_var#Mel-Frequency Cepstral Coefficients(梅尔频率倒谱系数)方差
 - data[i,50-61] = chromagram_mean色度频率均值
 - data[i,62-73] = chromagram_var色度频率方差
 - data[i,74] = chroma_stft_mean频率均值
 - data[i,75] = chroma_stft_var频率方差
 - data[i,76] = 流派(此项只在kaggle提取特征时存在，判别式中不存在)
 - data[i,77] = 编号(此项只在kaggle提取特征时存在，判别式中不存在)


本实验数据库采自kaggle-GTZAN，并在kaggle完成特征提取以即模型计算。

translate为转换文件，将各类型的音频文件转化为librosa库可识别的ＷＡＶ格式（需要安装ffmpeg）目前只支持转化mp3、flac

test内为测试文件

kaggle内为特征提取及模型计算文件，地址：<a href="https://www.kaggle.com/bluesmonster/sound-recognition">https://www.kaggle.com/bluesmonster/sound-recognition</a>
