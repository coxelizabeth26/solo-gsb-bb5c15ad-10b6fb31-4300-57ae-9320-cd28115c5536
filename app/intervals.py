"""数值区间预测的原文解析与命中判定（纯函数，使用十进制精确计算）。

揭示原文按「下界..上界」解析为两个有限十进制数：

    <lower>..<upper>

两侧各为一个可选符号、整数/小数（可带指数）的有限十进制字面量，
中间必须恰好有一个 ``..`` 分隔符；不允许空白、下划线或 ``Infinity`` /
``NaN`` 等非有限值。解析成功的区间还须满足「下界不得大于上界」。

区间覆盖（命中）为闭区间判定：``lower <= standard <= upper``，全部比较
与区间宽度（``upper - lower``）均用 :class:`decimal.Decimal` 精确计算，
不经过二进制浮点。
"""

import re
from decimal import Decimal, InvalidOperation
from dataclasses import dataclass

# 非法格式：无法按「下界..上界」切分，或任一侧不是有限十进制字面量
INVALID_FORMAT = "INVALID_FORMAT"
# 格式可解析，但下界严格大于上界
INVALID_ORDER = "INVALID_ORDER"

# 可选符号 + 至少一位 ASCII 数字（可带小数点与指数）；显式拒绝空白、
# 下划线、Unicode 数字、Infinity、NaN——这些即使 Decimal() 能接受也不
# 属于本接口的合法字面量
_FINITE_DECIMAL_PATTERN = re.compile(
    r"^[+-]?(?:[0-9]+(?:\.[0-9]*)?|\.[0-9]+)(?:[eE][+-]?[0-9]+)?$"
)


@dataclass(frozen=True)
class Interval:
    """解析成功的闭区间预测。"""

    lower: Decimal
    upper: Decimal

    @property
    def width(self) -> Decimal:
        """区间宽度（非负，十进制精确）。"""
        return self.upper - self.lower

    def covers(self, value: Decimal) -> bool:
        """闭区间覆盖判定：value 落在 [lower, upper]（含边界）。"""
        return self.lower <= value <= self.upper


def parse_decimal(text: str) -> Decimal | None:
    """把单侧边界解析为有限 Decimal；非有限十进制字面量返回 None。"""
    if not _FINITE_DECIMAL_PATTERN.fullmatch(text):
        return None
    try:
        value = Decimal(text)
    except InvalidOperation:
        return None
    if not value.is_finite():
        return None
    return value


def parse_interval(original_text: str) -> Interval | str:
    """把揭示原文解析为区间。

    成功返回 :class:`Interval`；失败返回原因代码
    （``INVALID_FORMAT`` / ``INVALID_ORDER``）。
    """
    parts = original_text.split("..")
    if len(parts) != 2:
        return INVALID_FORMAT
    lower = parse_decimal(parts[0])
    upper = parse_decimal(parts[1])
    if lower is None or upper is None:
        return INVALID_FORMAT
    if lower > upper:
        return INVALID_ORDER
    return Interval(lower=lower, upper=upper)
