##### 精度统计工具

###### 简要概述：

 - 本工具主要用于将图片流或者视频流算法输出结果和真值结果进行比对，返回结果指标，提高测试效率。

   ```
   ### 图片流格式要求：
   - 算法输出结果以txt文件保存，目前仅支持txt。格式如下(图像名称，类别, 边界框坐标，置信度)，一行表示一张图的信息：
		000006.jpg sedan 1291 567 1981 1279 0.99 sedan 951 78 1122 234 0.18 
   
   - 图像流gt文件是xml文件格式，通常使用标注软件如labelimg等标注的格式
   
   ### 视频流格式要求：
   - 算法输出结果以txt文件保存，目前仅支持txt。格式如下(视频名称，帧号，类别, id, 边界框坐标，置信度) 一行表示一帧的信息：
		1.mp4 7 person 0 308 712 430 953 0.88 person 1 425 727 570 964 0.89 
		
   - 视频流gt文件是xml文件格式，通常使用标注软件如DarkLabel等标注的格式
   ```

   

###### 安装环境(建议在conda环境下安装)
	- pip install -r requirements.txt
   

###### 操作流程

- 本代码提供了以下示例：      
```
图像指标比对：
	- python main.py --Manually-annotate-dir ./test_data/detect/image/gt --source-file ./test_data/detect/image/pred.txt  --data-path ./test/images --tensorboard

视频指标对比：
	- python main.py --Manually-annotate-dir ./test_data/detect/video/gt --source-file ./test_data/detect/video/pred.txt  --data-path ./test_data/detect/video/images --data_type video --tensorboard
```


- 参数解析如下：
  - Manually-annotate-dir: gt文件夹路径，以 xml格式
  - source-file： 算法检测结果
  - conf-thres： 置信度阈值
  - img-path： 图像路径
  - conf-thres-setting: conf 阈值 从0到1 划分为1000份，运行1000次，默认开启; 命令行使用后conf使用默认值0.25，只运行一次
  - save-csv-path: 保存conf阈值 从0到1 划分为1000份后的精度，召回率指标
  - data_type: 数据类型，image或者video
  - tensorboard: 结果和曲线图web端展示
  
###### 结果
  - 图片流和视频流帧上标注预测框和真值框，结果进行保存，默认result文件夹下
  - csv文件存储conf阈值从1到1000(默认)下检测结果，结果默认存储在result文件夹下
  - PR曲线保存,结果默认存储在result文件夹下
  - tensorboard log日志保存，默认result文件夹tensorboard_dirs
