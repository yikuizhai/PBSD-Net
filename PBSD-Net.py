import torch, math, copy
import torch.nn as nn

def autopad(k, p=None, d=1):  # kernel, padding, dilation
    """Pad to 'same' shape outputs."""
    if d > 1:
        k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]  # actual kernel-size
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
    return p

class Conv(nn.Module):
    # Standard convolution
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=True):  # ch_in, ch_out, kernel, stride, padding, groups
        super(Conv, self).__init__()
        # print(c1, c2, k, s,)
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p), groups=g, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.SiLU() if act is True else (act if isinstance(act, nn.Module) else nn.Identity())

    def forward(self, x):
        # print("Conv", x.shape)
        res= self.act(self.bn(self.conv(x)))
        return res

    def fuseforward(self, x):
        res = self.act(self.conv(x))

        return res

class Bottleneck(nn.Module):
    # Standard bottleneck
    def __init__(self, c1, c2, shortcut=True, g=1, e=0.5):  # ch_in, ch_out, shortcut, groups, expansion
        super(Bottleneck, self).__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_, c2, 3, 1, g=g)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))

class DySnakeConv(nn.Module):
    def __init__(self, inc, ouc, k=3) -> None:
        super().__init__()

        self.conv_0 = Conv(inc, ouc, k)
        self.conv_x = DSConv(inc, ouc, 0, k)
        self.conv_y = DSConv(inc, ouc, 1, k)

    def forward(self, x):
        return torch.cat([self.conv_0(x), self.conv_x(x), self.conv_y(x)], dim=1)

class DSConv(nn.Module):
    def __init__(self, in_ch, out_ch, morph, kernel_size=3, if_offset=True, extend_scope=1):

        super(DSConv, self).__init__()

        self.offset_conv = nn.Conv2d(in_ch, 2 * kernel_size, 3, padding=1)
        self.bn = nn.BatchNorm2d(2 * kernel_size)
        self.kernel_size = kernel_size


        self.dsc_conv_x = nn.Conv2d(
            in_ch,
            out_ch,
            kernel_size=(kernel_size, 1),
            stride=(kernel_size, 1),
            padding=0,
        )
        self.dsc_conv_y = nn.Conv2d(
            in_ch,
            out_ch,
            kernel_size=(1, kernel_size),
            stride=(1, kernel_size),
            padding=0,
        )

        self.gn = nn.GroupNorm(out_ch // 4, out_ch)
        self.act = Conv.default_act

        self.extend_scope = extend_scope
        self.morph = morph
        self.if_offset = if_offset

    def forward(self, f):
        offset = self.offset_conv(f)
        offset = self.bn(offset)

        offset = torch.tanh(offset)
        input_shape = f.shape
        dsc = DSC(input_shape, self.kernel_size, self.extend_scope, self.morph)
        deformed_feature = dsc.deform_conv(f, offset, self.if_offset)
        if self.morph == 0:
            x = self.dsc_conv_x(deformed_feature.type(f.dtype))
            x = self.gn(x)
            x = self.act(x)
            return x
        else:
            x = self.dsc_conv_y(deformed_feature.type(f.dtype))
            x = self.gn(x)
            x = self.act(x)
            return x

class DSC(object):
    def __init__(self, input_shape, kernel_size, extend_scope, morph):
        self.num_points = kernel_size
        self.width = input_shape[2]
        self.height = input_shape[3]
        self.morph = morph
        self.extend_scope = extend_scope
        self.num_batch, self.num_channels = input_shape[0], input_shape[1]

    def _coordinate_map_3D(self, offset, if_offset):
        device = offset.device
        y_offset, x_offset = torch.split(offset, self.num_points, dim=1)

        # Create center grid
        y_center = torch.arange(self.width).repeat(self.height).view(self.height, self.width).T.unsqueeze(0).repeat(
            self.num_points, 1, 1).float()
        x_center = torch.arange(self.height).repeat(self.width).view(self.width, self.height).T.unsqueeze(0).repeat(
            self.num_points, 1, 1).float()

        # Initialize kernel
        if self.morph == 0:
            y = torch.tensor([[0]])
            x = torch.linspace(-self.num_points // 2, self.num_points // 2, self.num_points)
        else:
            y = torch.linspace(-self.num_points // 2, self.num_points // 2, self.num_points)
            x = torch.tensor([[0]])

        y, x = torch.meshgrid(y.squeeze(), x.squeeze())
        y_grid, x_grid = y.reshape(-1, 1).repeat(1, self.width * self.height).view(self.num_points, self.width,
                                                                                   self.height), x.reshape(-1,
                                                                                                           1).repeat(1,
                                                                                                                     self.width * self.height).view(
            self.num_points, self.width, self.height)

        y_new = y_center + y_grid
        x_new = x_center + x_grid

        # Repeat for batch
        y_new = y_new.unsqueeze(0).repeat(self.num_batch, 1, 1, 1).to(device)
        x_new = x_new.unsqueeze(0).repeat(self.num_batch, 1, 1, 1).to(device)

        # Apply offset if necessary
        if if_offset:
            offset_new = self._apply_offset(y_offset if self.morph == 0 else x_offset, y_new, self.num_points,
                                            self.extend_scope, device)
            y_new.add_(offset_new)

        return self._reshape_outputs(y_new), self._reshape_outputs(x_new)

    def _apply_offset(self, offset, new_grid, num_points, extend_scope, device):
        center = num_points // 2
        offset_new = offset.detach().clone().permute(1, 0, 2, 3)

        offset_new[center] = 0
        for index in range(1, center):
            offset_new[center + index] = offset_new[center + index - 1] + offset[center + index]
            offset_new[center - index] = offset_new[center - index + 1] + offset[center - index]

        return offset_new.permute(1, 0, 2, 3).to(device).mul(extend_scope)

    def _reshape_outputs(self, tensor):
        return tensor.reshape(self.num_batch, self.num_points, 1, self.width, self.height).permute(0, 3, 1, 4,
                                                                                                   2).reshape(
            self.num_batch, self.num_points * self.width, self.height)

    def _bilinear_interpolate_3D(self, input_feature, y, x):
        device = input_feature.device
        y = y.reshape([-1]).float()
        x = x.reshape([-1]).float()

        zero = torch.zeros([], dtype=torch.int)
        max_y, max_x = self.width - 1, self.height - 1

        # Find grid locations
        y0, y1 = torch.clamp(torch.floor(y).int(), zero, max_y), torch.clamp(torch.floor(y).int() + 1, zero, max_y)
        x0, x1 = torch.clamp(torch.floor(x).int(), zero, max_x), torch.clamp(torch.floor(x).int() + 1, zero, max_x)

        input_feature_flat = input_feature.flatten(start_dim=0, end_dim=1).reshape(self.num_batch, self.num_channels,
                                                                                   self.width, self.height).permute(0,
                                                                                                                    2,
                                                                                                                    3,
                                                                                                                    1).reshape(
            -1, self.num_channels)

        # Calculate base indices
        base = (torch.arange(self.num_batch).unsqueeze(-1) * (self.height * self.width)).float().to(device)
        repeat = torch.ones([self.num_points * self.width * self.height]).unsqueeze(0).float()

        base = base.matmul(repeat).reshape([-1]).to(device)
        base_y0, base_y1 = base + y0 * self.height, base + y1 * self.height

        # Get indices for interpolation
        index_a0 = base_y0 - base + x0
        index_c0 = base_y0 - base + x1
        index_a1 = base_y1 - base + x0
        index_c1 = base_y1 - base + x1

        # Get 8 grid values
        value_a0 = input_feature_flat[index_a0.long()]
        value_c0 = input_feature_flat[index_c0.long()]
        value_a1 = input_feature_flat[index_a1.long()]
        value_c1 = input_feature_flat[index_c1.long()]

        # Calculate volumes
        y0_float, y1_float = y0.float(), y1.float()
        x0_float, x1_float = x0.float(), x1.float()

        vol_a0 = ((y1_float - y) * (x1_float - x)).unsqueeze(-1).to(device)
        vol_c0 = ((y1_float - y) * (x - x0_float)).unsqueeze(-1).to(device)
        vol_a1 = ((y - y0_float) * (x1_float - x)).unsqueeze(-1).to(device)
        vol_c1 = ((y - y0_float) * (x - x0_float)).unsqueeze(-1).to(device)

        outputs = (value_a0 * vol_a0 + value_c0 * vol_c0 + value_a1 * vol_a1 + value_c1 * vol_c1)

        # Reshape outputs
        if self.morph == 0:
            outputs = outputs.reshape(self.num_batch, self.num_points * self.width, self.height,
                                      self.num_channels).permute(0, 3, 1, 2)
        else:
            outputs = outputs.reshape(self.num_batch, self.width, self.num_points * self.height,
                                      self.num_channels).permute(0, 3, 1, 2)

        return outputs

    def deform_conv(self, input, offset, if_offset):
        y, x = self._coordinate_map_3D(offset, if_offset)
        return self._bilinear_interpolate_3D(input, y, x)

class Bottleneck_DySnakeConv(Bottleneck):

    def __init__(self, c1, c2, shortcut=True, g=1, k=(3, 3), e=0.5):  # ch_in, ch_out, shortcut, groups, kernels, expand
        super().__init__(c1, c2, shortcut, g, k, e)
        c_ = int(c2 * e)  # hidden channels
        self.cv2 = DySnakeConv(c_, c2, k[1])
        self.cv3 = Conv(c2 * 3, c2, k=1)
    def forward(self, x):

        return x + self.cv3(self.cv2(self.cv1(x))) if self.add else self.cv3(self.cv2(self.cv1(x)))

class C3(nn.Module):

    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):

        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c1, c_, 1, 1)
        self.cv3 = Conv(2 * c_, c2, 1)  # optional act=FReLU(c2)
        self.m = nn.Sequential(*(Bottleneck(c_, c_, shortcut, g, k=((1, 1), (3, 3)), e=1.0) for _ in range(n)))

    def forward(self, x):

        return self.cv3(torch.cat((self.m(self.cv1(x)), self.cv2(x)), 1))

class SDSC(C3):
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        super().__init__(c1, c2, n, shortcut, g, e)
        self.m = nn.ModuleList(Bottleneck_DySnakeConv(self.c, self.c, shortcut, g, k=(3, 3), e=1.0) for _ in range(n))

class FM(nn.Module):
    def __init__(self, dim, focal_window=3, focal_level=2, focal_factor=2, bias=True, proj_drop=0.,
                 use_postln_in_modulation=False, normalize_modulator=False):
        super().__init__()

        self.focal_layers = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(dim, dim, kernel_size=focal_factor * k + focal_window, stride=1,
                          groups=dim, padding=(focal_factor * k + focal_window) // 2, bias=False),
                nn.GELU()
            ) for k in range(focal_level)
        ])
        self.f_linear = nn.Conv2d(dim, 2 * dim + (focal_level + 1), kernel_size=1, bias=bias)
        self.h = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        self.proj = nn.Conv2d(dim, dim, kernel_size=1)
        self.proj_drop = nn.Dropout(proj_drop)
        self.act = nn.GELU()
        self.use_postln_in_modulation = use_postln_in_modulation
        self.ln = nn.LayerNorm(dim) if use_postln_in_modulation else None
        self.normalize_modulator = normalize_modulator

    def forward(self, x):
        C = x.shape[1]
        x = self.f_linear(x).contiguous()
        q, ctx, gates = torch.split(x, (C, C, self.focal_level + 1), 1)

        ctx_all = sum(self.focal_layers[l](ctx) * gates[:, l:l + 1] for l in range(self.focal_level))
        ctx_global = self.act(ctx.mean(dim=(2, 3), keepdim=True))
        ctx_all += ctx_global * gates[:, self.focal_level:]

        if self.normalize_modulator:
            ctx_all /= (self.focal_level + 1)

        x_out = q * self.h(ctx_all).contiguous()
        if self.use_postln_in_modulation and self.ln:
            x_out = self.ln(x_out)

        return self.proj_drop(self.proj(x_out))

class Concat(nn.Module):
    def __init__(self, dimension=1):
        super().__init__()
        self.d = dimension

    def forward(self, x):

        sizes = [feature.size() for feature in x]
        max_height = max([size[2] for size in sizes])
        max_width = max([size[3] for size in sizes])

        resized_features = []
        for feature in x:
            if feature.size(2) != max_height or feature.size(3) != max_width:

                upsampled = nn.functional.interpolate(feature, size=(max_height, max_width), mode='bilinear',
                                                      align_corners=False)
                resized_features.append(upsampled)
            else:
                resized_features.append(feature)

        return torch.cat(resized_features, self.d)


######################################anchor based##################################
class Detect0(nn.Module):
    stride = None  # strides computed during build
    export = False  # onnx export

    def __init__(self, nc=80, anchors=(), ch=()):  # detection layer
        super(Detect0, self).__init__()
        self.nc = nc  # number of classes
        self.no = nc + 5  # number of outputs per anchor
        self.nl = len(anchors)  # number of detection layers
        self.na = len(anchors[0]) // 2  # number of anchors
        self.grid = [torch.zeros(1)] * self.nl  # init grid
        a = torch.tensor(anchors).float().view(self.nl, -1, 2)
        self.register_buffer('anchors', a)  # shape(nl,na,2)
        self.register_buffer('anchor_grid', a.clone().view(self.nl, 1, -1, 1, 1, 2))  # shape(nl,1,na,1,1,2)
        self.m = nn.ModuleList(nn.Conv2d(x, self.no * self.na, 1) for x in ch)  # output conv

    def forward(self, x):
        # x = x.copy()  # for profiling
        z = []  # inference output
        self.training |= self.export
        for i in range(self.nl):
            x[i] = self.m[i](x[i])  # conv
            bs, _, ny, nx = x[i].shape  # x(bs,255,20,20) to x(bs,3,20,20,85)
            x[i] = x[i].view(bs, self.na, self.no, ny, nx).permute(0, 1, 3, 4, 2).contiguous()

            if not self.training:  # inference
                if self.grid[i].shape[2:4] != x[i].shape[2:4]:
                    self.grid[i] = self._make_grid(nx, ny).to(x[i].device )

                y = x[i].sigmoid()
                y[..., 0:2] = (y[..., 0:2] * 2. - 0.5 + self.grid[i]) * self.stride[i]  # xy
                y[..., 2:4] = (y[..., 2:4] * 2) ** 2 * self.anchor_grid[i]  # wh
                z.append(y.view(bs, -1, self.no))

        return x if self.training else (torch.cat(z, 1), x)

    @staticmethod
    def _make_grid(nx=20, ny=20):
        yv, xv = torch.meshgrid([torch.arange(ny), torch.arange(nx)])
        return torch.stack((xv, yv), 2).view((1, 1, ny, nx, 2)).float()
######################################anchor based#################################

######################################anchor free#################################
class DFL(nn.Module):

    def __init__(self, c1=16):
        """Initialize a convolutional layer with a given number of input channels."""
        super().__init__()
        self.conv = nn.Conv2d(c1, 1, 1, bias=False).requires_grad_(False)
        x = torch.arange(c1, dtype=torch.float)
        self.conv.weight.data[:] = nn.Parameter(x.view(1, c1, 1, 1))
        self.c1 = c1

    def forward(self, x):
        """Applies a transformer layer on input tensor 'x' and returns a tensor."""
        b, _, a = x.shape  # batch, channels, anchors
        return self.conv(x.view(b, 4, self.c1, a).transpose(2, 1).softmax(1)).view(b, 4, a)
        # return self.conv(x.view(b, self.c1, 4, a).softmax(1)).view(b, 4, a)

def make_anchors(feats, strides, grid_cell_offset=0.5):
    """Generate anchors from features."""
    anchor_points, stride_tensor = [], []
    assert feats is not None
    dtype, device = feats[0].dtype, feats[0].device
    for i, stride in enumerate(strides):
        _, _, h, w = feats[i].shape
        sx = torch.arange(end=w, device=device, dtype=dtype) + grid_cell_offset  # shift x
        sy = torch.arange(end=h, device=device, dtype=dtype) + grid_cell_offset  # shift y
        sy, sx = torch.meshgrid(sy, sx, indexing="ij")
        anchor_points.append(torch.stack((sx, sy), -1).view(-1, 2))
        stride_tensor.append(torch.full((h * w, 1), stride, dtype=dtype, device=device))
    return torch.cat(anchor_points), torch.cat(stride_tensor)

def dist2bbox(distance, anchor_points, xywh=True, dim=-1):
    """Transform distance(ltrb) to box(xywh or xyxy)."""
    lt, rb = distance.chunk(2, dim)
    x1y1 = anchor_points - lt
    x2y2 = anchor_points + rb
    if xywh:
        c_xy = (x1y1 + x2y2) / 2
        wh = x2y2 - x1y1
        return torch.cat((c_xy, wh), dim)  # xywh bbox
    return torch.cat((x1y1, x2y2), dim)  # xyxy bbox

class Detect1(nn.Module):

    dynamic = False  # force grid reconstruction
    export = False  # export mode
    end2end = False  # end2end
    max_det = 300  # max_det
    shape = None
    anchors = torch.empty(0)  # init
    strides = torch.empty(0)  # init

    def __init__(self, nc=80, ch=()):

        super().__init__()
        self.nc = nc  # number of classes
        self.nl = len(ch)  # number of detection layers
        self.reg_max = 16  # DFL channels (ch[0] // 16 to scale 4/8/12/16/20 for n/s/m/l/x)
        self.no = nc + self.reg_max * 4  # number of outputs per anchor
        self.stride = torch.zeros(self.nl)  # strides computed during build
        c2, c3 = max((16, ch[0] // 4, self.reg_max * 4)), max(ch[0], min(self.nc, 100))  # channels
        self.cv2 = nn.ModuleList(
            nn.Sequential(Conv(x, c2, 3), Conv(c2, c2, 3), nn.Conv2d(c2, 4 * self.reg_max, 1)) for x in ch
        )
        self.cv3 = nn.ModuleList(nn.Sequential(Conv(x, c3, 3), Conv(c3, c3, 3), nn.Conv2d(c3, self.nc, 1)) for x in ch)
        self.dfl = DFL(self.reg_max) if self.reg_max > 1 else nn.Identity()

        if self.end2end:
            self.one2one_cv2 = copy.deepcopy(self.cv2)
            self.one2one_cv3 = copy.deepcopy(self.cv3)

    def forward(self, x):
        """Concatenates and returns predicted bounding boxes and class probabilities."""
        if self.end2end:
            return self.forward_end2end(x)

        for i in range(self.nl):
            x[i] = torch.cat((self.cv2[i](x[i]), self.cv3[i](x[i])), 1)
        if self.training:  # Training path
            return x
        y = self._inference(x)
        return y if self.export else (y, x)

    def forward_end2end(self, x):

        one2one = [
            torch.cat((self.one2one_cv2[i](x[i]), self.one2one_cv3[i](x[i])), 1) for i in range(self.nl)
        ]
        if hasattr(self, 'cv2') and hasattr(self, 'cv3'):
            for i in range(self.nl):
                x[i] = torch.cat((self.cv2[i](x[i]), self.cv3[i](x[i])), 1)
        if self.training:  # Training path
            return {"one2many": x, "one2one": one2one}

        y = self._inference(one2one)
        y = self.postprocess(y.permute(0, 2, 1), self.max_det, self.nc)
        return y if self.export else (y, {"one2many": x, "one2one": one2one})

    def _inference(self, x):
        """Decode predicted bounding boxes and class probabilities based on multiple-level feature maps."""
        # Inference path
        shape = x[0].shape  # BCHW
        x_cat = torch.cat([xi.view(shape[0], self.no, -1) for xi in x], 2)
        if self.dynamic or self.shape != shape:
            self.anchors, self.strides = (x.transpose(0, 1) for x in make_anchors(x, self.stride, 0.5))
            self.shape = shape

        if self.export and self.format in {"saved_model", "pb", "tflite", "edgetpu", "tfjs"}:  # avoid TF FlexSplitV ops
            box = x_cat[:, : self.reg_max * 4]
            cls = x_cat[:, self.reg_max * 4 :]
        else:
            box, cls = x_cat.split((self.reg_max * 4, self.nc), 1)

        if self.export and self.format in {"tflite", "edgetpu"}:

            grid_h = shape[2]
            grid_w = shape[3]
            grid_size = torch.tensor([grid_w, grid_h, grid_w, grid_h], device=box.device).reshape(1, 4, 1)
            norm = self.strides / (self.stride[0] * grid_size)
            dbox = self.decode_bboxes(self.dfl(box) * norm, self.anchors.unsqueeze(0) * norm[:, :2])
        else:
            dbox = self.decode_bboxes(self.dfl(box), self.anchors.unsqueeze(0)) * self.strides

        return torch.cat((dbox, cls.sigmoid()), 1)

    def bias_init(self):
        """Initialize Detect() biases, WARNING: requires stride availability."""
        m = self  # self.model[-1]  # Detect() module

        for a, b, s in zip(m.cv2, m.cv3, m.stride):  # from
            a[-1].bias.data[:] = 1.0  # box
            b[-1].bias.data[: m.nc] = math.log(5 / m.nc / (640 / s) ** 2)  # cls (.01 objects, 80 classes, 640 img)
        if self.end2end:
            for a, b, s in zip(m.one2one_cv2, m.one2one_cv3, m.stride):  # from
                a[-1].bias.data[:] = 1.0  # box
                b[-1].bias.data[: m.nc] = math.log(5 / m.nc / (640 / s) ** 2)  # cls (.01 objects, 80 classes, 640 img)

    def decode_bboxes(self, bboxes, anchors):
        """Decode bounding boxes."""
        return dist2bbox(bboxes, anchors, xywh=not self.end2end, dim=1)

    @staticmethod
    def postprocess(preds: torch.Tensor, max_det: int, nc: int = 80):

        assert 4 + nc == preds.shape[-1]
        boxes, scores = preds.split([4, nc], dim=-1)
        max_scores = scores.amax(dim=-1)
        max_scores, index = torch.topk(max_scores, min(max_det, max_scores.shape[1]), axis=-1)
        index = index.unsqueeze(-1)
        boxes = torch.gather(boxes, dim=1, index=index.repeat(1, 1, boxes.shape[-1]))
        scores = torch.gather(scores, dim=1, index=index.repeat(1, 1, scores.shape[-1]))

        scores, index = torch.topk(scores.flatten(1), max_det, axis=-1)
        labels = index % nc
        index = index // nc
        boxes = boxes.gather(dim=1, index=index.unsqueeze(-1).repeat(1, 1, boxes.shape[-1]))

        return torch.cat([boxes, scores.unsqueeze(-1), labels.unsqueeze(-1).to(boxes.dtype)], dim=-1)
######################################anchor free#################################