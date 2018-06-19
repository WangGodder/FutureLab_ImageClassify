# FutureLab_ImageClassify所需第三方框架 <br>
pytorch 0.4.0<br>
torchvison 0.2.1<br>
请访问
https://pytorch.org/
获取pytorch在不同系统下的安装方法
<br>
numpy: anaconda自带或使用命令 pip install numpy 
<br>
<br>
本机运行环境 Windows10 GPU: TITANXP 12G * 2 内存 32768MB 联网环境
<br>
<br>
如何使用<br>
1.创建训练集<br>
将list.csv，categories.csv文件放在DataProcess中，把数据放在imgages/data中，如下图所示<br>
![Alt text](img1.PNG)<br>
在DataProcess中运行ReadDataDir.py<br>
生成data文件夹，里面就是生成的训练集<br>
2.训练    （可直接运行，使用默认参数）<br>
在classy/train_center_loss,py进行训练， <br>
我们使用了ImageNet, 请在联网的环境运行，或http://data.lip6.fr/cadene/pretrainedmodels/resnext101_64x4d-e77a0586.pth下载<br>
我们的训练参数保存在temp文件夹内<br>
log文件保存在logs文件夹内<br>
运行train_center_loss.py时可加参数--data 后选择训练图片路径<br>
运行train_center_loss.py时可加参数--batch_size 后选择训练的batch大小<br>
<br>
3.测试      (可直接运行，使用默认参数)<br>
在classy/test_center_loss.py进行测试，<br>
此时要求图片放在一个文件内即可。默认DataProcess/testB/data<br>
![Alt text](img2.PNG)<br>
输出结果默认保存在result文件夹内<br>
运行test_center_loss.py时可加参数--data 后选择训练图片路径<br>
运行test_center_loss.py时可加参数--model-path 后选择保存的模型路径<br>
运行test_center_loss.py时可加参数'--result-path 后选输出csv文件的路径<br>
