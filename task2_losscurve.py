import pandas as pd
import matplotlib.pyplot as plt
import os

project_dir = r"D:\course\Neural_Network_Project\PJ2"
results_file = os.path.join(project_dir, "task2_training_results.txt")
plot_dir = os.path.join(project_dir, "plots")
os.makedirs(plot_dir, exist_ok=True)

# 读取带正确标题行的文件
df = pd.read_csv(
    results_file,
    sep='\t',        # 制表符分隔
    header=0,        # 第一行为标题行
    engine='python'
)

# 打印列名验证（应显示正确列名）
print("列名:", df.columns)  

# 查看 Epoch 列数据情况，调试用
print("Epoch 列的数据情况：")
print(df['Epoch'].value_counts(dropna=False))
print("Epoch 列的空值数量：", df['Epoch'].isnull().sum())

# 处理 Epoch 列可能存在的 NaN，这里选择删除含 NaN 的行，可根据实际调整
df = df.dropna(subset=['Epoch'])

# 数据类型转换（处理可能的NaN或非数值问题）
# 单独转换数值列（避免影响 Model 等字符串列）
df['Epoch'] = df['Epoch'].astype(int)  # Epoch 转整数
# 处理带百分号的 Test Accuracy：先删百分号，再转浮点数
df['Test Accuracy'] = df['Test Accuracy'].str.rstrip('%').astype(float)  
df['Train Loss'] = df['Train Loss'].astype(float)  # 训练损失转浮点数
df['Test Loss'] = df['Test Loss'].astype(float)    # 测试损失转浮点数

# 提取模型数据
vgg_a = df[df['Model'] == 'vgg_a']
vgg_a_bn = df[df['Model'] == 'vgg_a_bn']

# 绘制图表
plt.figure(figsize=(12, 6))
# 左侧：准确率对比
plt.subplot(1, 2, 1)
plt.plot(vgg_a['Epoch'], vgg_a['Test Accuracy'], 'b-o', label='VGG-A (No BN)')
plt.plot(vgg_a_bn['Epoch'], vgg_a_bn['Test Accuracy'], 'r-s', label='VGG-A+BN (With BN)')
plt.xlabel('Epoch')
plt.ylabel('Test Accuracy')
plt.title('BN对测试准确率的影响')
plt.legend()

# 右侧：损失对比
plt.subplot(1, 2, 2)
plt.plot(vgg_a['Epoch'], vgg_a['Test Loss'], 'b--o', label='VGG-A (No BN)')
plt.plot(vgg_a_bn['Epoch'], vgg_a_bn['Test Loss'], 'r--s', label='VGG-A+BN (With BN)')
plt.xlabel('Epoch')
plt.ylabel('Test Loss')
plt.title('BN对测试损失的影响')
plt.legend()

plt.tight_layout()
plt.savefig(os.path.join(plot_dir, "bn_performance_comparison.png"))
plt.show()
