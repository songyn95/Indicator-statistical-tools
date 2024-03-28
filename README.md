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
	- python main.py --Manually-annotate-dir ./test_data/detect/image/gt --source-file ./test_data/detect/image/pred.txt  --data-path ./test/images --tensorboard --plot

视频指标对比：
	- python main.py --Manually-annotate-dir ./test_data/detect/video/gt --source-file ./test_data/detect/video/pred.txt  --data-path ./test_data/detect/video/images --data_type video --tensorboard
```

- 参数解析如下：
    - source-file： 算法检测结果,txt文件
    - Manually-annotate-dir: gt文件夹路径，以 xml格式
    - result-file: 保存csv文件路径名称
    - data_type: 要评估的数据类型，image或者video，不同类型评估指标不同
    - conf-thres： 置信度阈值
    - iou-thres： IOU阈值
    - imgsz： 设置图像大小，功能暂未开放
    - save-data-path： 保存结果图像(jpg/png)的路径
    - save-dir: 保存tensorboard日志文件路径
    - tensorboard: 是否将结果和曲线图web端展示，默认false
    - plot： 是否将预测框和gt框叠加到图像上，默认false
    - plot_evolve: 是否绘制曲线(PR曲线，ROC曲线)，默认True

###### 结果

- 图片流和视频流帧上标注预测框和真值框，结果进行保存，默认save-data-path路径
- csv文件存储conf阈值从1到1000(默认)下检测结果，结果默认存储在result-file
- PR, ROC曲线生成,结果和result-file同一路径
- tensorboard log日志保存, 结果默认存储在save-dir
