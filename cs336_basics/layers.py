import torch
import torch.nn as nn
from einops import rearrange, einsum
import einx
from jaxtyping import Int

class Linear(nn.Module):
    def __init__(self, in_features: int, out_features: int, device=None, dtype=None):
        """
        Construct a linear transformation module. 
        in_features: int  final dimension of the input
        out_features: int  final dimension of the output
        device: torch.device | None = None  Device to store the parameters on
        dtype: torch.dtype | None = None  Data type of the parameters
        """
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.device = device
        self.dtype = dtype

        sigma = (2 / (in_features + out_features)) ** 0.5
        W_tensor = torch.empty(out_features, in_features)
        W = nn.init.trunc_normal_(W_tensor, mean=0, std=sigma, a=-3*sigma, b=3*sigma)
        self.W = nn.Parameter(W)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply the linear transformation to the input.
        """
        x = einsum(x, self.W, "... d_in, d_out d_in -> ... d_out")
        return x

class Embedding(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, device=None, dtype=None):
        """
        Construct an embedding module. 
        num_embeddings: int  Size of the vocabulary
        embedding_dim: int  Dimension of the embedding vectors, i.e., d_model
        device: torch.device | None = None  Device to store the parameters on
        dtype: torch.dtype | None = None  Data type of the parameters
        """
        super().__init__()
        self.vocab_size = num_embeddings
        self.d_model = embedding_dim
        self.device = device
        self.dtype = dtype

        sigma = 1
        M_tensor = torch.empty(num_embeddings, embedding_dim)
        M = nn.init.trunc_normal_(M_tensor, mean=0, std=sigma, a=-3*sigma, b=3*sigma)
        self.M = nn.Parameter(M)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        """
        Lookup the embedding vectors for the given token IDs
        """
        token_ids = torch.LongTensor(token_ids) # [batch_size, sequence_length]
        result = torch.empty(len(token_ids), len(token_ids[0]), self.d_model) # [batch_size, sequence_length, d_model]
        for i in range(len(result)):
            for j in range(len(result[0])):
                token_id = token_ids[i][j]
                result[i][j] = self.M[token_id]
        return result

class RMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-5, device=None, dtype=None):
        """
        Construct the RMSNorm module. 
        d_model: int  Hidden dimension of the model
        eps: float = 1e-5  Epsilon value for numerical stability
        device: torch.device | None = None  Device to store the parameters on
        dtype: torch.dtype | None = None  Data type of the parameters
        """
        super().__init__()
        self.d_model = d_model
        self.eps = eps
        self.device = device
        self.dtype = dtype

        G_tensor = torch.ones([d_model], dtype=dtype, device=device)
        self.G = nn.Parameter(G_tensor)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Process an input tensor of shape (batch_size, sequence_length, d_model) 
        and return a tensor of the same shape.
        """
        in_dtype = x.dtype
        x = x.to(torch.float32)

        # dim=-1：对[batch seq d_model]内部相加得到[batch seq 1], 其size(-1)则为d_model
        # keepdim=True用于保留[batch seq 1]中的1，否则得到[batch seq]无法正确广播
        RMS_x = ((x*x).sum(dim=-1, keepdim=True) / x.size(dim=-1) + self.eps)** 0.5
        result = (x / RMS_x) * self.G
        return result.to(in_dtype)

class SwiGLU(nn.Module):
    def __init__(self, d_model: int, d_ff: int, device=None, dtype=None):
        super().__init__()
        self.d_model = d_model
        self.device = device
        self.dtype = dtype

        raw_d_ff = 8 * d_model / 3
        caculated_d_ff = 64 * round(raw_d_ff / 64)
        if d_ff:
            self.d_ff = d_ff
        else:
            self.ff = caculated_d_ff

        sigma = (2 / (d_ff + d_model)) ** 0.5
        W1_tensor, W3_tensor = torch.empty(d_ff, d_model), torch.empty(d_ff, d_model)
        W2_tensor = torch.empty(d_model, d_ff)
        W1 = nn.init.trunc_normal_(W1_tensor, mean=0, std=sigma, a=-3*sigma, b=3*sigma)
        self.W1 = nn.Parameter(W1)
        W2 = nn.init.trunc_normal_(W2_tensor, mean=0, std=sigma, a=-3*sigma, b=3*sigma)
        self.W2 = nn.Parameter(W2)
        W3 = nn.init.trunc_normal_(W3_tensor, mean=0, std=sigma, a=-3*sigma, b=3*sigma)
        self.W3 = nn.Parameter(W3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        W1x = einx.dot("d_ff d_model, ... d_model -> ... d_ff", self.W1, x)
        W3x = einx.dot("d_ff d_model, ... d_model -> ... d_ff", self.W3, x)
        silu_W1x = W1x * torch.sigmoid(W1x)
        result = silu_W1x * W3x
        result = einx.dot("d_model d_ff, ... d_ff -> ... d_model", self.W2, result)
        return result