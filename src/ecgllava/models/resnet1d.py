"""1D ResNet 인코더 + 분류 헤드.

배포 제약 (DESIGN.md 3절): Conv1d, BatchNorm1d, ReLU, MaxPool1d, AdaptiveAvgPool1d,
Dropout, Linear 만 쓴다. RNN 계열과 커스텀 활성화는 배제한다. ONNX opset 17 로 뽑았을 때
TensorRT 가 바로 먹는 그래프를 목표로 한다.

2단계 연결: forward_features 가 반환하는 (B, 512, 32) 가 프로젝터 입력이 된다.
시간축 32 스텝이 곧 32개의 ECG 토큰이다. LLaVA 에서 CLIP 패치 토큰이 프로젝터를 거쳐
LLM 임베딩 공간으로 들어가는 자리와 같다. 1단계 체크포인트를 그대로 재사용하고
pooling 과 분류 헤드만 떼어낸다.
"""

import torch
import torch.nn as nn


class BasicBlock1d(nn.Module):
    def __init__(self, cin: int, cout: int, stride: int = 1, kernel_size: int = 7):
        super().__init__()
        pad = kernel_size // 2
        self.conv1 = nn.Conv1d(cin, cout, kernel_size, stride=stride, padding=pad,
                               bias=False)
        self.bn1 = nn.BatchNorm1d(cout)
        self.conv2 = nn.Conv1d(cout, cout, kernel_size, stride=1, padding=pad,
                               bias=False)
        self.bn2 = nn.BatchNorm1d(cout)
        self.relu = nn.ReLU(inplace=True)

        if stride != 1 or cin != cout:
            self.downsample = nn.Sequential(
                nn.Conv1d(cin, cout, 1, stride=stride, bias=False),
                nn.BatchNorm1d(cout),
            )
        else:
            self.downsample = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x if self.downsample is None else self.downsample(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return self.relu(out + identity)


class ResNet1d(nn.Module):
    def __init__(
        self,
        in_channels: int = 12,
        stem_channels: int = 64,
        stem_kernel: int = 15,
        stage_channels: tuple = (64, 128, 256, 512),
        blocks_per_stage: tuple = (2, 2, 2, 2),
        stage_strides: tuple = (1, 2, 2, 2),
        kernel_size: int = 7,
        dropout: float = 0.5,
        num_classes: int = 5,
    ):
        super().__init__()
        assert len(stage_channels) == len(blocks_per_stage) == len(stage_strides)

        self.stem = nn.Sequential(
            nn.Conv1d(in_channels, stem_channels, stem_kernel, stride=2,
                      padding=stem_kernel // 2, bias=False),
            nn.BatchNorm1d(stem_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(3, stride=2, padding=1),
        )

        stages = []
        cin = stem_channels
        for cout, n_blocks, stride in zip(stage_channels, blocks_per_stage,
                                          stage_strides):
            blocks = [BasicBlock1d(cin, cout, stride, kernel_size)]
            blocks += [BasicBlock1d(cout, cout, 1, kernel_size)
                       for _ in range(n_blocks - 1)]
            stages.append(nn.Sequential(*blocks))
            cin = cout
        self.stages = nn.Sequential(*stages)
        self.feature_dim = cin

        self.pool = nn.AdaptiveAvgPool1d(1)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(cin, num_classes)

        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out",
                                        nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
        # residual 분기의 마지막 BN 을 0 으로 두면 초기에 identity 로 동작해 학습이 안정된다.
        for m in self.modules():
            if isinstance(m, BasicBlock1d):
                nn.init.zeros_(m.bn2.weight)

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        """(B, 12, 1000) -> (B, 512, 32). 2단계 프로젝터 입력."""
        return self.stages(self.stem(x))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """(B, 12, 1000) -> (B, 5) 로짓. sigmoid 는 손실/평가 쪽에서 적용한다."""
        f = self.forward_features(x)
        f = self.pool(f).flatten(1)
        return self.fc(self.dropout(f))


def build_model(model_cfg) -> ResNet1d:
    if model_cfg.name != "resnet1d":
        raise ValueError(f"알 수 없는 모델: {model_cfg.name}")
    return ResNet1d(
        in_channels=model_cfg.in_channels,
        stem_channels=model_cfg.stem_channels,
        stem_kernel=model_cfg.stem_kernel,
        stage_channels=tuple(model_cfg.stage_channels),
        blocks_per_stage=tuple(model_cfg.blocks_per_stage),
        stage_strides=tuple(model_cfg.stage_strides),
        kernel_size=model_cfg.kernel_size,
        dropout=model_cfg.dropout,
        num_classes=model_cfg.num_classes,
    )
