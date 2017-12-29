# Tetris

这是一个使用Deep Q算法来训练俄罗斯方块ai的程序。

game.py 实现游戏的逻辑，不含UI

model_0.py d_model_1.py 模型文件，其中 d_model_1.py 是最终使用的模型，而前者是最初设计的。

robot.py robot_1.py ai游戏逻辑以及训练逻辑，其中robot_1.py 是最终使用的版本。

play.py 游戏UI，以及启动框架。

## 思路

整体思路使用deepQ的算法，在后期参考了这几篇文章中的算法：

包括优先清扫、启发式奖励函数、启发式奖励函数到分数的迁移等等

http://cs231n.stanford.edu/reports/2016/pdfs/121_Report.pdf

https://codemyroad.wordpress.com/2013/04/14/tetris-ai-the-near-perfect-player/

## 使用方法：

play.py

    直接进行游戏

play.py -a

    使用ai游戏

play.py -A0

    使用ai进行无界面游戏，统计平均分数，可以用来评价ai
    -Ax, x为游戏的次数，如果输入0，则为10次

play.py -t0 [-n [-g]] [-m] [-l0] [-u0]

    训练模型
    -tx, x为训练次数，输入0为10000次
    -n, 创建一个新模型训练（如果没有这个选项，则会自动加载之前保存的模型继续训练）
    -g, 与 -n 一起使用，从golden目录中加载一个模型作为新模型的初始值（可以从中间一个存档开始进行训练）
    -m, 使用master训练模式，用于在后期进行调整。master模式下“随机动作”与“奖励函数”的逻辑有所不同
    -lx, 指定学习率，如果不指定，则使用指数下降的学习率
    -ux, 训练过程显示ui，一般用来调试。x表示UI的帧间隔（毫秒）

## 我的训练步骤：

play.py -n140000   ——这个结果被我保存为golden

play.py -n10000 -m -l0.0001

play.py -n10000 -m -l0.00008

后面的步骤可以酌情测试调整，包括学习率。
