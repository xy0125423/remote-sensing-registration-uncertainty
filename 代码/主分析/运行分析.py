"""对用户提供的原生 NBR 窗口运行冻结计算核心。"""
from __future__ import annotations
import argparse
import csv
from pathlib import Path
import numpy as np
from 响应边界 import SUPPORTS, evaluate


def main() -> None:
    """Read a documented numeric input and create a new result CSV."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error('输出文件已存在；请选择新文件。')
    with np.load(args.input, allow_pickle=False) as data:
        a = np.asarray(data['A'], dtype=float)
        b = np.asarray(data['B'], dtype=float)
        px, py = float(data['px']), float(data['py'])
        rv = np.array([float(data['r_east_m']), -float(data['r_north_m'])])
    if a.shape != (11, 11) or b.shape != (11, 11):
        parser.error('A、B必须为11×11的原生20米NBR窗口。')
    if not np.isfinite([px, py, *rv]).all() or not np.isfinite(a).all() or not np.isfinite(b).all():
        parser.error('输入必须为有限数值。')
    if not (40 <= px < 180 and 40 <= py < 180):
        parser.error('名义样地坐标须保留局部拟合与支持边距。')
    rows = [row for support in SUPPORTS for row in evaluate(a, b, px, py, rv, support)]
    fields = list(dict.fromkeys(key for row in rows for key in row))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x', encoding='utf-8-sig', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    print(f'已写入 {len(rows)} 个条件配置。')


if __name__ == '__main__':
    main()
