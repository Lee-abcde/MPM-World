# MPM-World
MPM-World is an open-source MPM engine for the physically-based simulation of fluid/sand/snow/elastic object. This simulation engine is based on the material point method (MPM) method and [taichi Lang](https://github.com/taichi-dev/taichi).



#### To do list:

* ~~更新一种含有摩擦系数的边界条件，使得弹性体不会跳来跳去 （高优先级）~~（finished 08/04/2023) solution： 使用separate边界即可，不需要添加额外摩擦力
* ~~修复旋度运算网格位置不对可能导致的流体不稳定（高优先级）~~
* 学习并实现沙子的仿真（高优先级）
* 整合加入拉格朗日力的解算
* 实现水体烟雾的压强projection （高优先级）
* 跑benchmark进行结果分析（高优先级）
* 粒子渲染（高优先级）
* 学习并实现刚体的耦合（中优先级）
* 优化烟雾耦合部分的计算（中优先级）
* ~~烟雾模拟长时间会发生渗漏问题，水体模拟长时间会退出（中优先级）~~ （finished 07/04/2023)
* 学习[QuanTaichi](https://github.com/taichi-dev/quantaichi)中fluid solver的写法（似乎是集成MPM,Euler的solver)（中优先级）
* [High-Performance MLS-MPM Solver with Cutting and Coupling (CPIC)](https://github.com/yuanming-hu/taichi_mpm)可以研究一下（低优先级）