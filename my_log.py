import logging
import os.path
import time


logger = logging.getLogger()
logger.setLevel(logging.INFO)

def creat_log():
    # 创建一个logger
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
def save_log(tips=''):
    # 创建一个handler用来写入日志文件
    rq = time.strftime('%Y %m %d %H %M', time.localtime(time.time()))
    log_path = './logs'  ##os.getcwd()返回当前工作路径
    filename = rq + tips + '.log'
    logfile = os.path.join(log_path, filename)
    fh = logging.FileHandler(logfile, mode='w')
    fh.setLevel(logging.DEBUG)  # 输出file的log等级开关
    # 定义handler的输出格式
    formatter = logging.Formatter('%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s')
    fh.setFormatter(formatter)
    # 将logger添加到handler里面
    logger.addHandler(fh)
    # 日志

