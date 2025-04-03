# -*- encoding: utf-8 -*-
"""
@File: ascii_art.py
@Modify Time: 2025/4/3 15:04       
@Author: Kevin-Chen
@Descriptions: 
"""
import argparse
from PIL import Image


def main(the_input='input.jpg', output='output.txt', aspect=0.35, width=150, reverse=False, save_txt=False):
    """
    - description: 该函数是一个图片转字符画的工具，可以将输入的图片转换为ASCII字符画，并保存为文本文件。

    参数:
    - the_input: str, 输入图片路径，默认为 'input.txt'
    - output: str, 输出文件路径，默认为 'output.txt'
    - aspect: float, 高宽补偿系数，默认值为 0.35
    - width: int, 字符画宽度，默认值为 150
    - reverse: bool, 是否反转字符颜色，默认为 False
    - save_txt: bool, 是否保存为文本文件，默认为 False
    """

    chars = ['@', '#', 'S', '%', '?', '*', '+', ';', ':', ',', '.']
    if reverse:
        chars = chars[::-1]

    try:
        img = Image.open(the_input).convert('L')  # 直接转为灰度图
    except Exception as e:
        print(f"打开图片失败：{e}")
        return

    # 精确计算高度（考虑字符高宽比）
    orig_width, orig_height = img.size
    aspect_ratio = orig_height / orig_width

    # 新高度 = 字符画宽度 * 原图高宽比 * 补偿系数
    new_height = int(width * aspect_ratio * aspect)
    img = img.resize((width, new_height))

    # 像素到字符的映射
    pixels = img.getdata()
    scale = (len(chars) - 1) / 255
    chars = [chars[int(p * scale)] for p in pixels]

    # 生成结果
    result = '\n'.join(''.join(chars[i * width:(i + 1) * width])
                       for i in range(new_height))
    print(result)

    if save_txt:
        with open(output, 'w') as f:
            f.write(result)
    print(f"字符画已生成：{output} (尺寸 {width}x{new_height} 字符)")


if __name__ == '__main__':
    main('NBI.png')

