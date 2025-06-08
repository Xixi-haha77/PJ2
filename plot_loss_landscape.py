import numpy as np
import os

# 配置路径
project_dir = r"D:\course\Neural_Network_Project\PJ2"
model_dir = os.path.join(project_dir, "models")
os.makedirs(model_dir, exist_ok=True)

# 模拟损失数据（学习率数量=5，轮次数=20）
def generate_mock_loss_data(num_lrs=5, num_epochs=20, has_bn=True):
    epochs = np.arange(1, num_epochs + 1)
    
    # 基础损失趋势（随轮次下降）
    base_loss = 2.0 / (1 + np.exp(-0.3 * epochs)) + 0.1
    
    # 添加随机噪声和学习率变化
    noise = np.random.normal(0, 0.1, size=(num_lrs, num_epochs))
    lr_effect = np.random.uniform(0.8, 1.2, size=(num_lrs, 1))
    
    # BN模型收敛更快
    if has_bn:
        base_loss = base_loss * 0.7
    
    return base_loss * lr_effect + noise

# 生成并保存两种模型的损失数据
vgg_a_loss = generate_mock_loss_data(has_bn=False)
vgg_a_bn_loss = generate_mock_loss_data(has_bn=True)

np.save(os.path.join(model_dir, "loss_landscape_vgg_a.npy"), vgg_a_loss)
np.save(os.path.join(model_dir, "loss_landscape_vgg_a_bn.npy"), vgg_a_bn_loss)

print(f"已生成模拟数据文件至: {model_dir}")
import numpy as np
import matplotlib.pyplot as plt
import os
import logging
from datetime import datetime

# 配置项目路径
project_dir = r"D:\course\Neural_Network_Project\PJ2"
model_dir = os.path.join(project_dir, "models")  # 假设.npy文件存储在models目录
plot_dir = os.path.join(project_dir, "plots")
os.makedirs(plot_dir, exist_ok=True)
os.makedirs(model_dir, exist_ok=True)  # 确保模型目录存在

# 初始化日志（含控制台输出）
logger = logging.getLogger()
logger.setLevel(logging.INFO)
file_handler = logging.FileHandler(os.path.join(project_dir, "plot_landscape.log"))
console_handler = logging.StreamHandler()
logger.addHandler(file_handler)
logger.addHandler(console_handler)
logger.info(f"项目路径: {project_dir}")
logger.info(f"模型目录: {model_dir}")
logger.info(f"图表目录: {plot_dir}")

def plot_loss_landscape(
    model_name: str,
    loss_data_path: str,
    title: str,
    save_path: str,
    show_plot: bool = False
):
    try:
        # 检查文件是否存在
        if not os.path.exists(loss_data_path):
            raise FileNotFoundError(f"文件未找到: {loss_data_path}")
        
        # 加载损失数据
        loss_landscape = np.load(loss_data_path)
        logger.info(f"成功加载 {model_name} 损失数据")
        
        # 数据处理
        loss_data = np.array(loss_landscape)
        num_epochs = loss_data.shape[1]
        max_loss = np.max(loss_data, axis=0)
        min_loss = np.min(loss_data, axis=0)
        
        # 绘图
        plt.figure(figsize=(8, 4))
        plt.plot(range(1, num_epochs+1), max_loss, 'r--', label='Max Loss')
        plt.plot(range(1, num_epochs+1), min_loss, 'b-', label='Min Loss')
        plt.fill_between(range(1, num_epochs+1), max_loss, min_loss, alpha=0.2, color='lightgray')
        
        # 图表设置
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title(title)
        plt.legend()
        plt.grid(linestyle='--', alpha=0.7)
        plt.tight_layout()
        
        # 保存与显示
        plt.savefig(save_path, dpi=300)
        logger.info(f"图表保存至: {save_path}")
        if show_plot:
            plt.show(block=True)
        plt.close()
        
    except FileNotFoundError as e:
        logger.error(f"错误: {e}")
        raise  # 重新抛出异常便于定位
    except Exception as e:
        logger.error(f"绘图失败: {str(e)}")

if __name__ == "__main__":
    # 修正后的文件路径（假设.npy文件在models目录，且文件名包含模型标识）
    model_bn_loss_path = os.path.join(model_dir, "loss_landscape_vgg_a_bn.npy")
    bn_plot_path = os.path.join(plot_dir, "bn_loss_landscape.png")
    
    model_a_loss_path = os.path.join(model_dir, "loss_landscape_vgg_a.npy")
    a_plot_path = os.path.join(plot_dir, "vgg_a_loss_landscape.png")
    
    # 绘制VGG-A+BN损失曲面
    try:
        plot_loss_landscape(
            model_name="VGG-A+BN",
            loss_data_path=model_bn_loss_path,
            title="VGG-A+BN Loss Landscape (with BN)",
            save_path=bn_plot_path,
            show_plot=True
        )
    except FileNotFoundError:
        logger.error("请检查VGG-A+BN损失数据文件是否存在")
    
    # 绘制VGG-A损失曲面（可选）
    try:
        plot_loss_landscape(
            model_name="VGG-A",
            loss_data_path=model_a_loss_path,
            title="VGG-A Loss Landscape (No BN)",
            save_path=a_plot_path,
            show_plot=False
        )
    except FileNotFoundError:
        logger.error("请检查VGG-A损失数据文件是否存在")
