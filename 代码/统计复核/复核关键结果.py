"""从公开的逐火场汇总重建主切片数量及五场等权比例。"""
from pathlib import Path
import csv
import math
import hashlib


def main() -> None:
    """Check aggregation independently without rerunning research experiments."""
    root = Path(__file__).resolve().parents[2]
    def read(name):
        with (root / '关键结果' / name).open(encoding='utf-8-sig', newline='') as stream:
            return list(csv.DictReader(stream))
    rows = [r for r in read('逐火场判别汇总.csv')
            if r['support'] == 'circle15' and r['index'] == 'dNBR'
            and float(r['lambda']) == 0.5]
    expected = read('主分析与严格质量子集对照.csv')
    for radius in [0, 2, 5, 10, 20]:
        selected = [r for r in rows if float(r['C_m']) == radius]
        reference = next(r for r in expected if r['组别'] == '主分析140' and float(r['C_m']) == radius)
        assert len(selected) == 5
        total = sum(int(r['records']) for r in selected)
        guaranteed = sum(int(r['positive']) + int(r['negative']) for r in selected)
        equal = sum((int(r['positive']) + int(r['negative'])) / int(r['records']) for r in selected) / 5
        assert total == int(reference['n']) == 140
        assert guaranteed == int(reference['可保证'])
        assert math.isclose(equal, float(reference['原五场等权']), abs_tol=1e-12)
        assert sum(int(r['newly_guaranteed']) for r in selected) == 0
        print(f'C={radius:2d} m: {guaranteed}/{total}, equal-fire={equal:.6%}, newly guaranteed=0')
    for row in read('文件校验.csv'):
        assert hashlib.sha256((root / row['相对路径']).read_bytes()).hexdigest() == row['SHA256']
    print('关键统计与冻结图表文件校验通过。严格子集数据为封版汇总，未重算原生质量规则。')


if __name__ == '__main__':
    main()
