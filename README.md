# MPM-World
MPM-World is an open-source fluid engine for the physically-based simulation of fluids. The simulation in this library is based on the material point method (MPM) method.





#### To do list:

* 更新一种含有摩擦系数的边界条件，使得弹性体不会跳来跳去 （高优先级）
* ~~修复旋度运算网格位置不对可能导致的流体不稳定（高优先级）~~
* 优化烟雾耦合部分的计算（中优先级）
* ~~烟雾模拟长时间会发生渗漏问题，水体模拟长时间会退出（中优先级）~~ （finished 07/04/2023)
* 学习[QuanTaichi](https://github.com/taichi-dev/quantaichi)中fluid solver的写法（似乎是集成MPM,Euler的solver)（中优先级）
* [High-Performance MLS-MPM Solver with Cutting and Coupling (CPIC)](https://github.com/yuanming-hu/taichi_mpm)可以研究一下（低优先级）