# RS_designer
 A QGIS plugin to design layouts, calculate parameters and evaluate the capacities of the rainwater drainage system in plain area.
 This plugin includes three main parts: Layout Design, Hydraulic Design and Model Simulation. The layout and hydraulic design section merges the first two parts into one.

一个平原地区雨水排水管网的自动设计、计算及评估的QGIS插件。
主要包括三个部分：平面布局设计、水力设计计算和模型模拟评估。其中布局及水力设计为前两步骤的整合。


 ## Layout Design 平面布局设计
 It generates the pipeline layout and the network structure of the rainwater drainage system according to road network and river data.
 It outputs the drainage nodes, pipes, subcatchments and service regions to represent the layout.
 It mainly consists of these functions:
 - Pre-process pipeline data
 - Create subcatchments
 - Extract outfalls
 - Divide drainage network paths

 根据规划项目范围的道路中心线、河道面和道路面等数据，设计雨水管网的平面布局结构，形成管网的布置形式，导出雨水管网节点、管道、汇水分区及排口系统范围等设计结果。主要包括以下内容：
 - 管网数据预处理
 - 子汇水区划分
 - 排口提取
 - 排水分界

 ## Hydraulic Design 水力设计计算
 It calculates the hydraulic parameters such as diameter, velocity and flow rate according to the network structure.
 It outputs the calculation table (.xlsx), the GIS files (.shp) and the SWMM model (.inp) of the rainwater drainage network.
 It mainly consists of the following contents:
 - Interpolate DEM
 - Topology check
 - Attribute check
 - Hydraulic calculation
 - Vertical elevation design
 - SWMM model formulation

 基于排水管网的平面布局设计结果，采用水力设计参数和地形高程信息对管网的管径流量等设计参数进行计算，导出水力计算表和设计方案GIS文件，并形成SWMM模型的输入文件。主要包括以下内容：
 - DEM生成
 - 拓扑关系梳理
 - 管网属性检查
 - 水力参数计算
 - 竖向高程设计
 - SWMM模型构建

 ## Model Simulation模型模拟评估
 The system is simulated using the console method of SWMM5.exe under the set rainfall scenarios.
 The drainage capacity is evaluated according to the simulation results from the report file (.rpt).

 在设定的降雨场景条件下，调用SWMM Engine进行模拟，从输出的rpt格式文件中读取内涝结果，评估自排系统各汇水区的内涝风险等级和管道承载力，验证规划方案所满足的重现期标准。