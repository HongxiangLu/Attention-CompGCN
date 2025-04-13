from helper import *
from model.message_passing import MessagePassing

class CompGCNConv(MessagePassing):
	def __init__(self, in_channels, out_channels, num_rels, act=lambda x:x, params=None):
		super(self.__class__, self).__init__()

		self.p 			= params
		self.in_channels	= in_channels
		self.out_channels	= out_channels
		self.num_rels 		= num_rels
		self.act 		= act
		self.device		= None

		self.w_loop		= get_param((in_channels, out_channels))
		self.w_in		= get_param((in_channels, out_channels))
		self.w_out		= get_param((in_channels, out_channels))
		self.w_rel 		= get_param((in_channels, out_channels))
		self.loop_rel 		= get_param((1, in_channels));

		self.drop		= torch.nn.Dropout(self.p.dropout)
		self.bn			= torch.nn.BatchNorm1d(out_channels)

		if self.p.bias: self.register_parameter('bias', Parameter(torch.zeros(out_channels)))

	def forward(self, x, edge_index, edge_type, rel_embed): 
		if self.device is None:
			self.device = edge_index.device

		rel_embed = torch.cat([rel_embed, self.loop_rel], dim=0)
		num_edges = edge_index.size(1) // 2
		num_ent   = x.size(0)

		self.in_index, self.out_index = edge_index[:, :num_edges], edge_index[:, num_edges:]
		self.in_type,  self.out_type  = edge_type[:num_edges], 	 edge_type [num_edges:]

		self.loop_index  = torch.stack([torch.arange(num_ent), torch.arange(num_ent)]).to(self.device)
		self.loop_type   = torch.full((num_ent,), rel_embed.size(0)-1, dtype=torch.long).to(self.device)

		self.in_norm     = self.compute_norm(self.in_index,  num_ent)
		self.out_norm    = self.compute_norm(self.out_index, num_ent)
		
		in_res		= self.propagate('add', self.in_index,   x=x, edge_type=self.in_type,   rel_embed=rel_embed, edge_norm=self.in_norm, 	mode='in')
		loop_res	= self.propagate('add', self.loop_index, x=x, edge_type=self.loop_type, rel_embed=rel_embed, edge_norm=None, 		mode='loop')
		out_res		= self.propagate('add', self.out_index,  x=x, edge_type=self.out_type,  rel_embed=rel_embed, edge_norm=self.out_norm,	mode='out')
		out		= self.drop(in_res)*(1/3) + self.drop(out_res)*(1/3) + loop_res*(1/3)

		if self.p.bias: out = out + self.bias
		out = self.bn(out)

		return self.act(out), torch.matmul(rel_embed, self.w_rel)[:-1]		# Ignoring the self loop inserted

	class RelTransform(torch.nn.Module):
		def __init__(self, num_feats):
			super(self.__class__, self).__init__()
			self.num_feats = num_feats
			self.weights = torch.nn.Parameter(torch.zeros(size=(2 * num_feats, 1)))
			torch.nn.init.xavier_uniform_(self.weights.data, gain=1.414)
			self.Weights = torch.nn.Parameter(torch.zeros(size=(3 * num_feats, num_feats)))
			torch.nn.init.xavier_uniform_(self.Weights.data, gain=1.414)

			self.dropout = 0
			self.alpha = 0.01
			self.LeakyReLU = torch.nn.LeakyReLU(self.alpha)

		def forward(self, ent_embed, rel_embed):
			trans_0 = ccorr(ent_embed, rel_embed)
			trans_1 = ent_embed - rel_embed
			trans_2 = ent_embed * rel_embed

			'''
			num_nodes = ent_embed.size(0)
			# 将三个特征矩阵堆叠为三维张量（形状 3×N×F）
			trans_stack = torch.stack([trans_0, trans_1, trans_2], dim=0)  # 3×N×F
			# 构建广播组合维度
			trans_i = trans_stack.unsqueeze(1).expand(3, 3, num_nodes, self.num_feats)  # 3×1×N×F
			trans_j = trans_stack.unsqueeze(0).expand(3, 3, num_nodes, self.num_feats)  # 1×3×N×F
			# 拼接特征并执行矩阵乘法
			concat_feats = torch.cat([trans_i, trans_j], dim=-1)  # 3×3×N×2F
			coefficients = self.LeakyReLU(torch.matmul(concat_feats, self.weights).squeeze(-1))  # 3×3×N
			'''
			coefficients = []
			for i, trans_i in enumerate([trans_0, trans_1, trans_2]):
				coefficients.append([])
				for trans_j in (trans_0, trans_1, trans_2):
					coefficients[i].append(self.LeakyReLU(torch.matmul(torch.cat([trans_i, trans_j], dim=1), self.weights).squeeze(1)))
			coefficients_tensor = torch.stack([torch.stack(row, dim=0) for row in coefficients], dim=0)

			# attentions = F.dropout(F.softmax(coefficients, dim=1), self.dropout, training=self.training).unsqueeze(-1)
			attentions = []
			for i in range(3):
				attentions.append(F.dropout(F.softmax(coefficients_tensor[i], dim=0), self.dropout, training=self.training))
			attentions = torch.stack(attentions, dim=0)

			'''
			# 将 trans_0/1/2 堆叠为三维张量（形状 3×N×F）
			trans = torch.stack([trans_0, trans_1, trans_2], dim=0)
			# 张量乘法与求和（利用广播机制）
			summed = (trans.unsqueeze(0) * attentions).sum(dim=1)
			# 维度重组（将结果拼接为 N×3F）
			results = summed.permute(1, 0, 2).reshape(-1, 3 * trans_0.size(1))
			'''
			results = []
			for i in range(3):
				results.append(trans_0 * attentions[i][0].unsqueeze(1) + trans_1 * attentions[i][1].unsqueeze(1) + trans_2 * attentions[i][2].unsqueeze(1))
			results = torch.cat(results, dim=1)

			return torch.matmul(results, self.Weights)

	def message(self, x_j, edge_type, rel_embed, edge_norm, mode):
		weight 	= getattr(self, 'w_{}'.format(mode))
		rel_emb = torch.index_select(rel_embed, 0, edge_type)
		trans_model = self.RelTransform(self.in_channels).to(torch.device("cuda"))
		xj_rel = trans_model(x_j, rel_emb)
		out	= torch.mm(xj_rel, weight)

		return out if edge_norm is None else out * edge_norm.view(-1, 1)

	def update(self, aggr_out):
		return aggr_out

	def compute_norm(self, edge_index, num_ent):
		row, col	= edge_index
		edge_weight 	= torch.ones_like(row).float()
		deg		= scatter_add( edge_weight, row, dim=0, dim_size=num_ent)	# Summing number of weights of the edges
		deg_inv		= deg.pow(-0.5)							# D^{-0.5}
		deg_inv[deg_inv	== float('inf')] = 0
		norm		= deg_inv[row] * edge_weight * deg_inv[col]			# D^{-0.5}

		return norm

	def __repr__(self):
		return '{}({}, {}, num_rels={})'.format(
			self.__class__.__name__, self.in_channels, self.out_channels, self.num_rels)
