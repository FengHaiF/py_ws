""" 使用Python中NetworkX包绘制深度神经网络结构图 """
# 导入相应包
import networkx as nx
import matplotlib.pyplot as plt

# 创建DAG
G = nx.DiGraph()

# 创建结构图顶点列表
vertex_list = ['v'+str(i) for i in range(1, 12)]

# 添加结构图顶点
G.add_nodes_from(vertex_list)

# 创建边列表
edge_list = [
             ('v1', 'v5'), ('v1', 'v6'), ('v1', 'v7'), ('v1', 'v8'), ('v1', 'v9'),('v1', 'v10'),
             ('v2', 'v5'), ('v2', 'v6'), ('v2', 'v7'), ('v2', 'v8'), ('v2', 'v9'),('v2', 'v10'),
             ('v3', 'v5'), ('v3', 'v6'), ('v3', 'v7'), ('v3', 'v8'), ('v3', 'v9'),('v3', 'v10'),
             ('v4', 'v5'), ('v4', 'v6'), ('v4', 'v7'), ('v4', 'v8'), ('v4', 'v9'),('v4', 'v10'),
             ('v5', 'v11'),
             ('v6', 'v11'),
             ('v7', 'v11'),
             ('v8', 'v11'),
             ('v9', 'v11'),
             ('v10', 'v11'),

            ]

# 通过列表形式来添加边
G.add_edges_from(edge_list)

# 指定绘制DAG图时每个顶点的位置
pos = {
        'v1': (-2, 1.5),
        'v2': (-2, 0.5),
        'v3': (-2, -0.5),
        'v4': (-2, -1.5),
        'v5': (0, 2.5),
        'v6': (0, 1.5),
        'v7': (0, 0.5),
        'v8': (0, -0.5),
        'v9': (0, -1.5),
        'v10': (0, -2.5),
        'v11': (2, 0),

       }

# 绘制DAG图
plt.title('Neural Network Structure')  # 神经网络结构图标题
plt.xlim(-2.2, 2.2)  # 设置X轴坐标范围
plt.ylim(-3, 3)  # 设置Y轴坐标范围
nx.draw(
        G,
        pos=pos,  # 点的位置
        node_color='red',  # 顶点颜色
        edge_color='black',  # 边的颜色
        with_labels=False,  # 不显示顶点标签
        font_size=10,  # 文字大小
        node_size=500,  # 顶点大小
       )

# 保存图片，图片大小为640*480
plt.savefig('./DNN_chart.png')

# 显示图片
plt.show()

