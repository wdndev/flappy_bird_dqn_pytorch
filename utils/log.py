# -*- coding: utf-8 -*-
#  @file        - log.py
#  @author      - dongnian.wang
#  @brief       - databus log
#  @version     - 1.0
#  @date        - 2022-09-29
#  @copyright   - Copyright (c) 2021 
""""
log

usage:
    logger.info("info")
    logger.debug("debug")
    logger.warning("warn")
    logger.error("error")
"""

import logging
import os
import sys
from datetime import datetime

class LogLevelException(Exception):
    """
        日志格式异常异常
    """
    pass

LOG_LEVELS = {
    'DEBUG': logging.DEBUG,
    'INFO': logging.INFO,
    'ERROR': logging.ERROR,
    'WARN': logging.WARN
}

def logger_init(log_level='DEBUG', model_name=__name__, log_dir="logs/", is_file=True):
    """ databus日志输出
    Note: 
        日志输出功能在python运行时只需调用一次.
        
    Args:
        log_level   : 日志等级 DEBUG, INFO, ERROR or WARN (str).
        model_name  : 运行日志的模块.
    
    Return:
        logger      : 日志对象.

    Raise:
        如果给定的日志级别未知.
    """
    if log_level not in LOG_LEVELS:
        raise LogLevelException("Unknown log level: {}".format(log_level))

    # 指定路径
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    log_path = os.path.join(log_dir, "log" + '_' + str(datetime.now())[:10] + '.txt')
    fmt_str = '[%(asctime)s][%(funcName)s %(lineno)d][%(levelname)s]: %(message)s'
    log_lvl = LOG_LEVELS[log_level]

    logging.basicConfig(format=fmt_str, level=log_lvl, datefmt='%Y-%d-%m %H:%M:%S')
    logger = logging.getLogger(model_name)
    logger.setLevel(log_lvl)
    
    # 执行Logging的基本配置（仅适用于STDOUT CONFIG）
    logging.basicConfig(format=fmt_str, level=log_lvl, datefmt='%Y-%d-%m %H:%M:%S')
    
    logger = logging.getLogger()
    logger.setLevel(log_lvl)
    handler = logging.StreamHandler(sys.stdout)
    formatter = logging.Formatter(fmt_str)
    if is_file:
        log_stream = logging.StreamHandler(log_path)
        handler.setStream(log_stream)

    handler.setFormatter(formatter)

    # # 删除getlogger添加的默认处理程序以避免重复的日志
    # if(logger.hasHandlers()):
    #     logger.handlers.clear()
    # logger.addHandler(handler)

    return logger

    # if only_file:
    #     logging.basicConfig(filename=log_path, level=log_lvl, format=fmt_str, datefmt='%Y-%d-%m %H:%M:%S')
    # else:
    #     logging.basicConfig(level=log_lvl, format=fmt_str, datefmt='%Y-%d-%m %H:%M:%S',
    #                         handlers=[logging.FileHandler(log_path), logging.StreamHandler(sys.stdout)])
    
