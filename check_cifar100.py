#!/usr/bin/env python3
import os
import sys
import argparse
import pickle


def load_pickle(file_path):
    with open(file_path, 'rb') as f:
        # CIFAR-100 使用 Python2 协议进行序列化，需指定 encoding='latin1' 才能在 Python3 下正确反序列化
        return pickle.load(f, encoding='latin1')


def validate_labels_in_range(labels, expected_count):
    if not hasattr(labels, '__len__'):
        return False, 'labels 无长度属性'
    if len(labels) != expected_count:
        return False, f'labels 数量不匹配: 期望 {expected_count}, 实际 {len(labels)}'
    # 标签应为 0..n-1
    uniq = set(int(x) for x in labels)
    if min(uniq) < 0 or max(uniq) >= 100:
        # 对 fine_labels 期望 < 100，coarse < 20 的检查在外层进行
        return False, '标签超出预期范围'
    return True, ''


def check_split(split_path, expected_num, split_name):
    issues = []
    try:
        d = load_pickle(split_path)
    except Exception as e:
        return False, [f'{split_name} 反序列化失败: {e}']

    # 关键键检查
    required_keys = {'data', 'fine_labels', 'coarse_labels', 'filenames'}
    missing = required_keys - set(d.keys())
    if missing:
        issues.append(f'{split_name} 缺少键: {sorted(list(missing))}')

    # 数量一致性
    for key in ['data', 'fine_labels', 'coarse_labels', 'filenames']:
        if key in d:
            try:
                n = len(d[key])
            except Exception:
                n = None
            if n != expected_num:
                issues.append(f'{split_name} {key} 数量不匹配: 期望 {expected_num}, 实际 {n}')

    # 标签范围
    if 'fine_labels' in d:
        fine = d['fine_labels']
        if not all((isinstance(x, int) and 0 <= x < 100) for x in fine):
            issues.append(f'{split_name} fine_labels 存在越界或非整型值')
    if 'coarse_labels' in d:
        coarse = d['coarse_labels']
        if not all((isinstance(x, int) and 0 <= x < 20) for x in coarse):
            issues.append(f'{split_name} coarse_labels 存在越界或非整型值')

    return len(issues) == 0, issues


def check_meta(meta_path):
    issues = []
    try:
        meta = load_pickle(meta_path)
    except Exception as e:
        return False, [f'meta 反序列化失败: {e}']

    if 'fine_label_names' not in meta or 'coarse_label_names' not in meta:
        issues.append('meta 缺少 fine_label_names 或 coarse_label_names')
        return False, issues

    fine_names = meta['fine_label_names']
    coarse_names = meta['coarse_label_names']
    if len(fine_names) != 100:
        issues.append(f'fine_label_names 数量不为 100: 实际 {len(fine_names)}')
    if len(coarse_names) != 20:
        issues.append(f'coarse_label_names 数量不为 20: 实际 {len(coarse_names)}')

    # 名称应为非空字符串
    if not all(isinstance(x, str) and len(x) > 0 for x in fine_names):
        issues.append('fine_label_names 存在空或非字符串项')
    if not all(isinstance(x, str) and len(x) > 0 for x in coarse_names):
        issues.append('coarse_label_names 存在空或非字符串项')

    return len(issues) == 0, issues


def check_cifar100(root_dir):
    ds_dir = os.path.join(root_dir, 'cifar-100-python')
    if not os.path.isdir(ds_dir):
        return False, [f'未找到目录: {ds_dir}']

    train_path = os.path.join(ds_dir, 'train')
    test_path = os.path.join(ds_dir, 'test')
    meta_path = os.path.join(ds_dir, 'meta')

    missing_files = [p for p in [train_path, test_path, meta_path] if not os.path.isfile(p)]
    if missing_files:
        return False, [f'缺少必要文件: {", ".join(missing_files)}']

    ok_train, issues_train = check_split(train_path, 50000, 'train')
    ok_test, issues_test = check_split(test_path, 10000, 'test')
    ok_meta, issues_meta = check_meta(meta_path)

    all_ok = ok_train and ok_test and ok_meta
    all_issues = []
    all_issues.extend(issues_train)
    all_issues.extend(issues_test)
    all_issues.extend(issues_meta)

    return all_ok, all_issues


def main():
    parser = argparse.ArgumentParser(description='CIFAR-100 数据集完整性检查')
    parser.add_argument('--root', type=str, default='.', help='数据根目录，包含 cifar-100-python 子目录')
    args = parser.parse_args()

    root_dir = os.path.abspath(args.root)
    ok, issues = check_cifar100(root_dir)

    if ok:
        print('CIFAR-100 完整性检查: 通过')
        sys.exit(0)
    else:
        print('CIFAR-100 完整性检查: 失败')
        for i, msg in enumerate(issues, 1):
            print(f'[{i}] {msg}')
        sys.exit(1)


if __name__ == '__main__':
    main()


