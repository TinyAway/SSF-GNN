import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv, GraphSAGE, GATv2Conv, EdgeConv
from PSPNet import PSPNet_vig, PSPNet_efficientnet


class MLP(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        num_layers: int,
        sigmoid_output: bool = False,
    ) -> None:
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(
            nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim])
        )
        self.sigmoid_output = sigmoid_output

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        if self.sigmoid_output:
            x = F.sigmoid(x)
        return x


class IndexSelector(nn.Module):
    def __init__(self, num_indices):
        super().__init__()
        # 每个指数一个可学习参数，初始为1
        self.weights = nn.Parameter(torch.ones(num_indices))

    def forward(self, index_tensor):
        # index_tensor: shape [B, C=num_indices, H, W]
        # softmax 或 sigmoid 控制每个通道的使用程度
        gated_weights = torch.sigmoid(self.weights).view(1, -1, 1, 1)
        return index_tensor * gated_weights


def segmentation_to_adj_matrix(segmentation):
    """
    将分割图转换为稠密邻接矩阵（仅同类像素相连）

    Args:
        segmentation (torch.Tensor): 分割图, shape=(H, W)

    Returns:
        X (torch.Tensor): 节点特征矩阵, shape=(num_nodes, 1)
        adj_matrix (torch.Tensor): 稠密邻接矩阵, shape=(num_nodes, num_nodes)
    """
    B, H, W = segmentation.shape
    N = H * W  # 像素数
    segmentation = segmentation.view(B, -1)  # 转为1D

    # 构建稠密邻接矩阵
    idx = torch.arange(N)  # 像素索引
    idx1, idx2 = torch.meshgrid(idx, idx, indexing='ij')  # 所有像素对

    # 筛选同类像素对
    same_class = segmentation[:, idx1] == segmentation[:, idx2]
    adj_matrix = same_class.float()

    return adj_matrix


def dense_to_sparse(adj):
   """将稠密邻接矩阵转换为PyG的edge_index和edge_attr格式"""
   assert adj.dim() == 2
   index = adj.nonzero(as_tuple=False).t()
   value = adj[index[0], index[1]]
   return index, value


class SSFGNN(nn.Module):
    def __init__(self, num_classes, GNN_depth=5):
        super(SSFGNN, self).__init__()
        self.pspnet = PSPNet_vig(n_classes=10)
        self.pspnet.load_state_dict(torch.load('checkpoints/pspnet-vig.pt'))
        self.layer3, self.layer4 = self.pspnet.layer3, self.pspnet.layer4

        self.feat_dim = 128
        self.hidden_dim = 64
        self.GNN_depth = GNN_depth

        self.node_conv = nn.Sequential(
            nn.Conv2d(384 + 20, self.feat_dim * 2, 3, 1, padding=1, bias=False),
            nn.BatchNorm2d(self.feat_dim * 2),
            nn.ReLU(),
            nn.Dropout2d(0.2)
        )

        self.edge_conv = nn.Sequential(
            nn.Conv2d(self.feat_dim, 64, 3, 1, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 8, 3, 1, 1),
            nn.BatchNorm2d(8),
            nn.ReLU()
        )

        self.pos_embedding = nn.Parameter(torch.randn(1, self.feat_dim, 8, 8))

        self.gcnconv_c = nn.ModuleList()
        self.gcnconv_s = nn.ModuleList()

        self.gcnconv_c.append(GATv2Conv(self.feat_dim, self.hidden_dim, heads=4))
        self.gcnconv_c.append(nn.BatchNorm1d(self.hidden_dim * 4))
        self.gcnconv_s.append(GATv2Conv(self.feat_dim, self.hidden_dim, heads=4, edge_dim=1))
        self.gcnconv_s.append(nn.BatchNorm1d(self.hidden_dim * 4))

        for i in range(1, GNN_depth - 1):
            self.gcnconv_c.append(GATv2Conv(self.hidden_dim * 4, self.hidden_dim, heads=4))
            self.gcnconv_c.append(nn.BatchNorm1d(self.hidden_dim * 4))

            self.gcnconv_s.append(GATv2Conv(self.hidden_dim * 4, self.hidden_dim, heads=4, edge_dim=1))
            self.gcnconv_s.append(nn.BatchNorm1d(self.hidden_dim * 4))

        self.gcnconv_c.append(GATv2Conv(self.hidden_dim * 4, self.hidden_dim, heads=1))
        self.gcnconv_s.append(GATv2Conv(self.hidden_dim * 4, self.hidden_dim, heads=1, edge_dim=1))


        self.avg_pool = nn.AvgPool2d(8)
        self.decoder = MLP(self.hidden_dim * 2, 256, num_classes, 3)
        self.index_selctor = IndexSelector(20)

    def forward(self, x):
        # segmentation branch
        img = x[:, :12, :, :]
        indices = x[:, 12:, :, :]
        with torch.no_grad():
            seg_logit, feat = self.pspnet(img)
            seg_logit = F.interpolate(seg_logit, (8, 8), mode='bilinear', align_corners=True)
            seg_map = seg_logit.argmax(1)
        adj_c = segmentation_to_adj_matrix(seg_map)
        adj_c = torch.block_diag(*adj_c)
        adj_c_index, _ = dense_to_sparse(adj_c)

        # graph.branch
        feat = self.layer3(feat)
        feat = self.layer4(feat)

        indices = F.interpolate(indices, (8, 8), mode='bilinear', align_corners=True)
        indices = self.index_selctor(indices)
        x_all = self.node_conv(torch.cat([feat, indices], dim=1))

        # x_all = self.node_conv(feat)
        x_node = x_all[:, :self.feat_dim, :, :]
        # x_node = torch.cat([x_node, indices], dim=1)
        B, C, H, W = x_node.shape
        x_node = x_node.permute(0, 2, 3, 1).reshape(B * H * W, C)  # B*H*W, C
        x_edge = x_all[:, self.feat_dim:, :, :]
        x_edge = x_edge + self.pos_embedding
        x_edge = self.edge_conv(x_edge)  # B, 8, H, W
        adj_s = create_adjacency_matrix(x_edge)  # B, HW, HW
        adj_s = torch.block_diag(*adj_s)         # BHW, BHW
        adj_s_index, adj_s_weight = dense_to_sparse(adj_s)

        x_node_s1 = self.gcnconv_s[0](x_node, adj_s_index, adj_s_weight)
        x_node_s1 = self.gcnconv_s[1](x_node_s1)
        x_node_s1 = F.relu(x_node_s1)
        x_node_s = x_node_s1.clone()

        x_node_c1 = self.gcnconv_c[0](x_node, adj_c_index)
        x_node_c1 = self.gcnconv_c[1](x_node_c1)
        x_node_c1 = F.relu(x_node_c1)
        x_node_c = x_node_c1.clone()

        for i in range(1, self.GNN_depth - 1):
            x_node_s = self.gcnconv_s[i * 2](x_node_s, adj_s_index, adj_s_weight)
            x_node_s = self.gcnconv_s[i * 2 + 1](x_node_s)
            x_node_s = F.relu(x_node_s)

            x_node_c = self.gcnconv_c[i * 2](x_node_c, adj_c_index)
            x_node_c = self.gcnconv_c[i * 2 + 1](x_node_c)
            x_node_c = F.relu(x_node_c)

        x_node_s = self.gcnconv_s[-1](x_node_s1 + x_node_s, adj_s_index, adj_s_weight)
        x_s = x_node_s.reshape(B, H, W, self.hidden_dim).permute(0, 3, 1, 2)

        x_node_c = self.gcnconv_c[-1](x_node_c1 + x_node_c, adj_c_index)
        x_c = x_node_c.reshape(B, H, W, self.hidden_dim).permute(0, 3, 1, 2)

        x = torch.cat([x_c, x_s], dim=1)
        x = self.avg_pool(x).view(B, -1)
        x = self.decoder(x)

        return x

    def train_mode(self):
        self.train()
        self.pspnet.eval()
        self.layer3.train()
        self.layer4.train()


def create_adjacency_matrix(edge_weights):
    B, _, H, W = edge_weights.shape
    A = torch.eye((H + 2) * (W + 2), device=edge_weights.device)
    A = A.unsqueeze(0).repeat(B, 1, 1)

    # 定义八邻域的偏移量
    offsets = [(-1, -1), (-1, 0), (-1, 1),
               (0, -1), (0, 1),
               (1, -1), (1, 0), (1, 1)]

    # 预计算所有节点的索引
    node_indices = torch.arange((H+2)*(W+2)).view(H+2, W+2)  # 节点索引矩阵

    h_neigh = torch.arange(1, H + 1).unsqueeze(1)
    w_neigh = torch.arange(1, W + 1).unsqueeze(0)
    neigh_indices1 = node_indices[h_neigh, w_neigh].view(-1)  # 邻域节点索引

    for k, (dh, dw) in enumerate(offsets):
        # 计算邻域节点的索引
        h_neigh = torch.arange(1, H+1).unsqueeze(1) + dh
        w_neigh = torch.arange(1, W+1).unsqueeze(0) + dw
        neigh_indices2 = node_indices[h_neigh, w_neigh].view(-1)  # 邻域节点索引

        # 填充邻接矩阵
        A[:, neigh_indices1, neigh_indices2] = edge_weights[:, k].view(B, -1)
    x, y = torch.meshgrid(neigh_indices1, neigh_indices1, indexing='ij')

    return A[:, x, y]


