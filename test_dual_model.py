#!/usr/bin/env python
"""
测试双解码器模型
"""

import torch
import numpy as np
from nets.attention_model_dual import AttentionModelDual
from dual_state import DualState


def create_random_tsp_instance(batch_size=2, graph_size=20):
    """创建随机TSP实例"""
    # 生成随机坐标
    loc = torch.rand((batch_size, graph_size, 2))
    return loc


def test_state_class():
    """测试状态类"""
    print("测试状态类...")
    
    # 创建简单的问题包装器
    class SimpleProblem:
        def __init__(self, size):
            self.size = size
            self.graph_size = size
            self.NAME = "tsp"
    
    problem = SimpleProblem(5)
    
    # 创建状态
    device = torch.device('cpu')
    state = DualState(
        torch.arange(2, device=device),
        problem,
        device
    )
    
    print(f"Batch size: {state.batch_size}")
    print(f"Graph size: {state.graph_size}")
    print(f"Start chains shape: {state.start_chains.shape}")
    print(f"End chains shape: {state.end_chains.shape}")
    
    # 测试可用终点
    available_ends = state.get_available_ends()
    print(f"Available ends shape: {available_ends.shape}")
    print(f"Available ends (batch 0): {available_ends[0]}")
    
    # 选择一个终点
    selected_end = torch.tensor([0, 1], device=device)
    
    # 测试可用起点
    available_starts = state.get_available_starts(selected_end)
    print(f"Available starts shape: {available_starts.shape}")
    print(f"Available starts (batch 0): {available_starts[0]}")
    
    print("状态类测试通过！")


def test_model_forward():
    """测试模型前向传播"""
    print("\n测试模型前向传播...")
    
    # 创建模型
    class SimpleProblem:
        def __init__(self):
            self.NAME = "tsp"
        
        def get_costs(self, input, pi):
            # 简单成本计算：路径长度
            batch_size, graph_size, _ = input.shape
            costs = torch.rand(batch_size, device=input.device) * 10
            mask = torch.zeros(batch_size, graph_size, dtype=torch.bool, device=input.device)
            return costs, mask
    
    problem = SimpleProblem()
    
    model = AttentionModelDual(
        embedding_dim=128,
        hidden_dim=128,
        problem=problem,
        n_encode_layers=2,
        n_heads=8,
        info_pass='basic',
        use_chain_info=True
    )
    
    # 创建测试数据
    batch_size = 4
    graph_size = 10
    input_data = create_random_tsp_instance(batch_size, graph_size)
    
    print(f"Input shape: {input_data.shape}")
    
    # 测试贪婪解码
    model.set_decode_type("greedy")
    model.eval()
    
    try:
        with torch.no_grad():
            cost, log_likelihood = model(input_data)
            print(f"Cost shape: {cost.shape}")
            print(f"Log likelihood shape: {log_likelihood.shape}")
            print(f"Cost values: {cost}")
            print(f"Log likelihood values: {log_likelihood}")
        print("模型前向传播测试通过！")
    except Exception as e:
        print(f"模型前向传播失败: {e}")
        import traceback
        traceback.print_exc()


def test_model_sampling():
    """测试模型采样"""
    print("\n测试模型采样...")
    
    class SimpleProblem:
        def __init__(self):
            self.NAME = "tsp"
        
        def get_costs(self, input, pi):
            batch_size = input.shape[0]
            costs = torch.rand(batch_size, device=input.device) * 10
            mask = torch.zeros(batch_size, input.shape[1], dtype=torch.bool, device=input.device)
            return costs, mask
    
    problem = SimpleProblem()
    
    model = AttentionModelDual(
        embedding_dim=64,  # 使用较小的维度以加快测试
        hidden_dim=64,
        problem=problem,
        n_encode_layers=2,
        n_heads=4,
        info_pass='basic',
        use_chain_info=False
    )
    
    # 创建测试数据
    input_data = create_random_tsp_instance(2, 8)
    
    # 测试采样解码
    model.set_decode_type("sampling", temp=1.0)
    model.train()  # 设置为训练模式
    
    try:
        cost, log_likelihood = model(input_data)
        print(f"采样模式 - Cost shape: {cost.shape}")
        print(f"采样模式 - Log likelihood shape: {log_likelihood.shape}")
        print("模型采样测试通过！")
    except Exception as e:
        print(f"模型采样失败: {e}")
        import traceback
        traceback.print_exc()


def test_dimension_consistency():
    """测试维度一致性"""
    print("\n测试维度一致性...")
    
    class SimpleProblem:
        def __init__(self):
            self.NAME = "tsp"
        
        def get_costs(self, input, pi):
            batch_size = input.shape[0]
            graph_size = input.shape[1]
            # 返回随机成本和掩码
            return torch.rand(batch_size, device=input.device), torch.zeros(batch_size, graph_size, dtype=torch.bool, device=input.device)
    
    problem = SimpleProblem()
    
    # 测试不同大小的图
    test_sizes = [5, 10, 20]
    
    for graph_size in test_sizes:
        print(f"\n测试图大小: {graph_size}")
        
        model = AttentionModelDual(
            embedding_dim=128,
            hidden_dim=128,
            problem=problem,
            n_encode_layers=3,
            n_heads=8,
            info_pass='rich',
            use_chain_info=True
        )
        
        # 创建不同批次大小的数据
        for batch_size in [1, 4, 8]:
            input_data = create_random_tsp_instance(batch_size, graph_size)
            
            try:
                # 测试贪婪模式
                model.set_decode_type("greedy")
                model.eval()
                with torch.no_grad():
                    cost, ll = model(input_data)
                
                assert cost.shape == (batch_size,), f"成本形状错误: {cost.shape}"
                assert ll.shape == (batch_size,), f"对数似然形状错误: {ll.shape}"
                
                print(f"  批次大小 {batch_size}: 通过")
                
            except Exception as e:
                print(f"  批次大小 {batch_size}: 失败 - {e}")


def main():
    """主测试函数"""
    print("开始测试双解码器模型...")
    
    # 运行所有测试
    test_state_class()
    test_model_forward()
    test_model_sampling()
    test_dimension_consistency()
    
    print("\n所有测试完成！")


if __name__ == "__main__":
    main()