import torch
from ..cpp_extensions import Tensor, DType


def tensor_repr(tensor: Tensor):
    if tensor.dtype == DType.Float8E4M3 or DType.Float8E5M2:
        conv_table = (
            torch.tensor(ALL_FP8E4M3_VALUES, device="cpu")
            if tensor.dtype == DType.Float8E4M3
            else torch.tensor(ALL_FP8E5M2_VALUES, device="cpu")
        )
        fp32_values = conv_table[tensor.data.cpu().int()]
        data_repr = repr(fp32_values)
    else:
        data_repr = repr(tensor.data)
    data_repr = data_repr[::-1][data_repr[::-1].find("]") :][::-1]
    data_repr = "T" + data_repr[1:]
    return f"""\
{data_repr},
       dtype={tensor.dtype.name},\
 amax={tensor.amax[0].item() if tensor.amax.numel() else None},\
 scale={tensor.scale.item() if tensor.scale.numel() else None},\
 scale_inv={tensor.scale_inv.item() if tensor.scale_inv.numel() else None}\
)"""

setattr(Tensor, "__repr__", tensor_repr)


# fmt: off
nan = float("nan")
inf = float("inf")
ALL_FP8E4M3_VALUES = [
   0.         ,    0.001953125,    0.00390625 ,    0.005859375,    0.0078125  ,    0.009765625,    0.01171875 ,    0.013671875,
   0.015625   ,    0.017578125,    0.01953125 ,    0.021484375,    0.0234375  ,    0.025390625,    0.02734375 ,    0.029296875,
   0.03125    ,    0.03515625 ,    0.0390625  ,    0.04296875 ,    0.046875   ,    0.05078125 ,    0.0546875  ,    0.05859375 ,
   0.0625     ,    0.0703125  ,    0.078125   ,    0.0859375  ,    0.09375    ,    0.1015625  ,    0.109375   ,    0.1171875  ,
   0.125      ,    0.140625   ,    0.15625    ,    0.171875   ,    0.1875     ,    0.203125   ,    0.21875    ,    0.234375   ,
   0.25       ,    0.28125    ,    0.3125     ,    0.34375    ,    0.375      ,    0.40625    ,    0.4375     ,    0.46875    ,
   0.5        ,    0.5625     ,    0.625      ,    0.6875     ,    0.75       ,    0.8125     ,    0.875      ,    0.9375     ,
   1.         ,    1.125      ,    1.25       ,    1.375      ,    1.5        ,    1.625      ,    1.75       ,    1.875      ,
   2.         ,    2.25       ,    2.5        ,    2.75       ,    3.         ,    3.25       ,    3.5        ,    3.75       ,
   4.         ,    4.5        ,    5.         ,    5.5        ,    6.         ,    6.5        ,    7.         ,    7.5        ,
   8.         ,    9.         ,   10.         ,   11.         ,   12.         ,   13.         ,   14.         ,   15.         ,
  16.         ,   18.         ,   20.         ,   22.         ,   24.         ,   26.         ,   28.         ,   30.         ,
  32.         ,   36.         ,   40.         ,   44.         ,   48.         ,   52.         ,   56.         ,   60.         ,
  64.         ,   72.         ,   80.         ,   88.         ,   96.         ,  104.         ,  112.         ,  120.         ,
 128.         ,  144.         ,  160.         ,  176.         ,  192.         ,  208.         ,  224.         ,  240.         ,
 256.         ,  288.         ,  320.         ,  352.         ,  384.         ,  416.         ,  448.         ,  nan          ,
  -0.         ,   -0.001953125,   -0.00390625 ,   -0.005859375,   -0.0078125  ,   -0.009765625,   -0.01171875 ,   -0.013671875,
  -0.015625   ,   -0.017578125,   -0.01953125 ,   -0.021484375,   -0.0234375  ,   -0.025390625,   -0.02734375 ,   -0.029296875,
  -0.03125    ,   -0.03515625 ,   -0.0390625  ,   -0.04296875 ,   -0.046875   ,   -0.05078125 ,   -0.0546875  ,   -0.05859375 ,
  -0.0625     ,   -0.0703125  ,   -0.078125   ,   -0.0859375  ,   -0.09375    ,   -0.1015625  ,   -0.109375   ,   -0.1171875  ,
  -0.125      ,   -0.140625   ,   -0.15625    ,   -0.171875   ,   -0.1875     ,   -0.203125   ,   -0.21875    ,   -0.234375   ,
  -0.25       ,   -0.28125    ,   -0.3125     ,   -0.34375    ,   -0.375      ,   -0.40625    ,   -0.4375     ,   -0.46875    ,
  -0.5        ,   -0.5625     ,   -0.625      ,   -0.6875     ,   -0.75       ,   -0.8125     ,   -0.875      ,   -0.9375     ,
  -1.         ,   -1.125      ,   -1.25       ,   -1.375      ,   -1.5        ,   -1.625      ,   -1.75       ,   -1.875      ,
  -2.         ,   -2.25       ,   -2.5        ,   -2.75       ,   -3.         ,   -3.25       ,   -3.5        ,   -3.75       ,
  -4.         ,   -4.5        ,   -5.         ,   -5.5        ,   -6.         ,   -6.5        ,   -7.         ,   -7.5        ,
  -8.         ,   -9.         ,  -10.         ,  -11.         ,  -12.         ,  -13.         ,  -14.         ,  -15.         ,
 -16.         ,  -18.         ,  -20.         ,  -22.         ,  -24.         ,  -26.         ,  -28.         ,  -30.         ,
 -32.         ,  -36.         ,  -40.         ,  -44.         ,  -48.         ,  -52.         ,  -56.         ,  -60.         ,
 -64.         ,  -72.         ,  -80.         ,  -88.         ,  -96.         , -104.         , -112.         , -120.         ,
-128.         , -144.         , -160.         , -176.         , -192.         , -208.         , -224.         , -240.         ,
-256.         , -288.         , -320.         , -352.         , -384.         , -416.         , -448.         ,  nan          ,
]

ALL_FP8E5M2_VALUES = [
      0.                ,      0.0000152587890625,      0.000030517578125 ,      0.0000457763671875,      0.00006103515625  ,     0.0000762939453125,      0.000091552734375 ,      0.0001068115234375,
      0.0001220703125   ,      0.000152587890625 ,      0.00018310546875  ,      0.000213623046875 ,      0.000244140625    ,     0.00030517578125  ,      0.0003662109375   ,      0.00042724609375  ,
      0.00048828125     ,      0.0006103515625   ,      0.000732421875    ,      0.0008544921875   ,      0.0009765625      ,     0.001220703125    ,      0.00146484375     ,      0.001708984375    ,
      0.001953125       ,      0.00244140625     ,      0.0029296875      ,      0.00341796875     ,      0.00390625        ,     0.0048828125      ,      0.005859375       ,      0.0068359375      ,
      0.0078125         ,      0.009765625       ,      0.01171875        ,      0.013671875       ,      0.015625          ,     0.01953125        ,      0.0234375         ,      0.02734375        ,
      0.03125           ,      0.0390625         ,      0.046875          ,      0.0546875         ,      0.0625            ,     0.078125          ,      0.09375           ,      0.109375          ,
      0.125             ,      0.15625           ,      0.1875            ,      0.21875           ,      0.25              ,     0.3125            ,      0.375             ,      0.4375            ,
      0.5               ,      0.625             ,      0.75              ,      0.875             ,      1.                ,     1.25              ,      1.5               ,      1.75              ,
      2.                ,      2.5               ,      3.                ,      3.5               ,      4.                ,     5.                ,      6.                ,      7.                ,
      8.                ,     10.                ,     12.                ,     14.                ,     16.                ,    20.                ,     24.                ,     28.                ,
     32.                ,     40.                ,     48.                ,     56.                ,     64.                ,    80.                ,     96.                ,    112.                ,
    128.                ,    160.                ,    192.                ,    224.                ,    256.                ,   320.                ,    384.                ,    448.                ,
    512.                ,    640.                ,    768.                ,    896.                ,   1024.                ,  1280.                ,   1536.                ,   1792.                ,
   2048.                ,   2560.                ,   3072.                ,   3584.                ,   4096.                ,  5120.                ,   6144.                ,   7168.                ,
   8192.                ,  10240.                ,  12288.                ,  14336.                ,  16384.                , 20480.                ,  24576.                ,  28672.                ,
  32768.                ,  40960.                ,  49152.                ,  57344.                ,    inf                 ,   nan                 ,    nan                 ,    nan                 ,
     -0.                ,     -0.0000152587890625,     -0.000030517578125 ,     -0.0000457763671875,     -0.00006103515625  ,    -0.0000762939453125,     -0.000091552734375 ,     -0.0001068115234375,
     -0.0001220703125   ,     -0.000152587890625 ,     -0.00018310546875  ,     -0.000213623046875 ,     -0.000244140625    ,    -0.00030517578125  ,     -0.0003662109375   ,     -0.00042724609375  ,
     -0.00048828125     ,     -0.0006103515625   ,     -0.000732421875    ,     -0.0008544921875   ,     -0.0009765625      ,    -0.001220703125    ,     -0.00146484375     ,     -0.001708984375    ,
     -0.001953125       ,     -0.00244140625     ,     -0.0029296875      ,     -0.00341796875     ,     -0.00390625        ,    -0.0048828125      ,     -0.005859375       ,     -0.0068359375      ,
     -0.0078125         ,     -0.009765625       ,     -0.01171875        ,     -0.013671875       ,     -0.015625          ,    -0.01953125        ,     -0.0234375         ,     -0.02734375        ,
     -0.03125           ,     -0.0390625         ,     -0.046875          ,     -0.0546875         ,     -0.0625            ,    -0.078125          ,     -0.09375           ,     -0.109375          ,
     -0.125             ,     -0.15625           ,     -0.1875            ,     -0.21875           ,     -0.25              ,    -0.3125            ,     -0.375             ,     -0.4375            ,
     -0.5               ,     -0.625             ,     -0.75              ,     -0.875             ,     -1.                ,    -1.25              ,     -1.5               ,     -1.75              ,
     -2.                ,     -2.5               ,     -3.                ,     -3.5               ,     -4.                ,    -5.                ,     -6.                ,     -7.                ,
     -8.                ,    -10.                ,    -12.                ,    -14.                ,    -16.                ,   -20.                ,    -24.                ,    -28.                ,
    -32.                ,    -40.                ,    -48.                ,    -56.                ,    -64.                ,   -80.                ,    -96.                ,   -112.                ,
   -128.                ,   -160.                ,   -192.                ,   -224.                ,   -256.                ,  -320.                ,   -384.                ,   -448.                ,
   -512.                ,   -640.                ,   -768.                ,   -896.                ,  -1024.                , -1280.                ,  -1536.                ,  -1792.                ,
  -2048.                ,  -2560.                ,  -3072.                ,  -3584.                ,  -4096.                , -5120.                ,  -6144.                ,  -7168.                ,
  -8192.                , -10240.                , -12288.                , -14336.                , -16384.                , 20480.                , -24576.                , -28672.                ,
 -32768.                , -40960.                , -49152.                , -57344.                ,   -inf                 ,   nan                 ,    nan                 ,    nan                 ,
]
