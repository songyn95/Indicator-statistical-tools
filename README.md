##### 精度统计工具

###### 简要概述：

 - 本工具主要用于将算法输出结果和真值结果进行比对，提高测试效率。

   ```
   格式要求：
   1. 算法输出结果以txt文件保存，目前仅支持txt。格式如下(图像名称，类别 边界框坐标，置信度)：
	000006.jpg sedan 1291 567 1981 1279 0.99 sedan 951 78 1122 234 0.18 
   
   2. gt文件是xml文件格式，通常使用标注软件如labelimg等标注的格式
   ```

   

###### 依赖环境

 - pandas

 - numpy

 - matplotlib

 - torch

 - xml

 - cv2

 - tqdm

   

###### 操作流程

- 本代码提供了以下示例：      

- python .\main.py --Manually-annotate-dir ./test/gt --source-file .\test\pred.txt --conf-thres 0.25  --img-path .\test\images\   

- python .\main.py --Manually-annotate-dir ./test/gt --source-file .\test\pred.txt --conf-thres 0.25  --img-path .\test\images\  --conf-thres-setting  

- 参数解析如下：
  - Manually-annotate-dir: gt文件夹路径，以 xml格式

  - source-file： 算法检测结果

  - conf-thres： 置信度阈值

  - img-path： 图像路径

  - conf-thres-setting: conf 阈值 从0到1 划分为1000份，运行1000次，默认开启; 命令行使用后conf使用默认值0.25，只运行一次

  - save-csv-path: 保存conf阈值 从0到1 划分为1000份后的精度，召回率指标
  

    

