# -*- coding: utf-8 -*-
# @Time : 2021/12/3 9:49
# @Author : Administrator
# @Email : jakesun2020@163.com
# @File : mylog.py
# @Project : 02-paper-REIA-relation-extraction-qa

import logging
import sys

def MyLog(args):
    logger = logging.getLogger("My")

    # 设置日志输出格式

    formatter = logging.Formatter('%(asctime)s [%(filename)s:%(lineno)s][%(levelname)s] %(message)s')
    # 指定日志的最低输出级别，默认为WARN级别
    logger.setLevel(logging.INFO)

    # 文件日志
    file_handler = logging.FileHandler("输出验证.log")
    file_handler.setFormatter(formatter)

    # 控制台日志
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)

    # 为logger添加的日志处理器
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    logger.info(args)
    logger.removeHandler(file_handler)
    logger.removeHandler(console_handler)

# if __name__ == '__main__':
#     MyLog("woshishei")