# mambair_arch
## 1 Dimension Attention
- Query, Key and Values are computed using convolution layers instead of linear transformations which are found in the traditional attention mechanism
- Attention scores are computed over 1 dimension only
- The softmax normalization and weighted sum are applied to the single-dimensional attention scores. In normal attention, these operations would be applied to the scores across all positions in the sequence.
```python
def oneDimensionAttention(self, x:torch.Tensor):
        B, C, d1, d2 = x.shape
        query = self.query_conv(x)
        key = self.key_conv(x)
        value = x
        scale_factor = C ** 0.5
        query = query.permute(0, 3, 2, 1)  # (B, W, H, C) or (B, H, W,C)
        key = key.permute(0, 3, 1, 2)  # (B, W, C, H) or (B, H, C, W)
        value = value.permute(0, 3, 2, 1)  # (B, W, H, C) or (B, H, W, C)
        scores = torch.matmul(query, key) / scale_factor  # (B, W, H, H) or (B,H,W,W)
        attention = torch.softmax(scores, dim=-1)
        out = torch.matmul(attention, value)  # (B, W, H, C) or (B,H,W,C)
        out = out.permute(0, 3, 2, 1).contiguous()  # (B, C, H, W) or (B,C,W,H)
        return out
```
## Linear Interpolation, Mean and Concatenation Convolution
- Mean is computed over both dimensions after the 1 dimension attention phase
- We feed this to 2 mamba blocks based on both dimensions
- The Mamba block requires a linear vector to be passed to it, so after that we need a method to get back to the original dimension.
-  For that purpose we use linear interpolation in the form of pytorch's expand function. This gets both horizontal and vertical outputs of mamba blocks into the same dimension. 
-  We then vertically stack both of these and use our concatenation convolution block to get back to the same dimension of the image.
```python
x1 = x.mean(dim=1)  # (B, W, C)
x2 = x.mean(dim=2)  # (B, H, C)

y1 = self.mamba(x1)  # (B, W, C)
y2 = self.mamba(x2)  # (B, H, C)

y1 = y1.unsqueeze(1)  # (B, 1, W, C)
y2 = y2.unsqueeze(2)  # (B, H, 1, C)

y1 = y1.expand(B, H, W, 2*C)
y2 = y2.expand(B, H, W, 2*C)

y = torch.cat((y1, y2), dim=3)  # (B, H, W, 4C)
y = y.permute(0,3,1,2) # (B, 4C, H, W)

y = self.concat_conv(y)  # (B, 2C, H, W)

y = y.permute(0,2,3,1) #(B, H, W, 2C)
```
## Normalization, Activation and Dropout
- Here the output projection is a linear layer.
- We use the Silu Activation function as well. 
- Dropout and out_norm is self explanatory
```python
y = self.out_norm(y)
print(f"After out norm: {y.shape}")
y = y * F.silu(z)
print(f"After silu: {y.shape}")
out = self.out_proj(y)
print(f"Ouput: {y.shape}")
if self.dropout is not None:
    out = self.dropout(out)
print(f"Ouput: {y.shape}")
return out
```
