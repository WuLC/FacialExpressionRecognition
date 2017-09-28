# -*- coding: utf-8 -*-
# @Author: lc
# @Date:   2017-09-24 17:58:25
# @Last Modified by:   WuLC
# @Last Modified time: 2017-09-24 21:28:54

import multiprocessing as mp
from multiprocessing import Pool
import os, time, random

"""
def long_time_task(name):
    print('Run task %s (%s)...' % (name, os.getpid()))
    start = time.time()
    time.sleep(random.random() * 3)
    end = time.time()
    print('Task %s runs %0.2f seconds.' % (name, (end - start)))


class Calculator():
    def __init__(self):
        print('creating a calculator')
    def calculate(self, x):
        return x, x*x

ca = Calculator()

def f(x):
    global ca 
    return ca.calculate(x)

if __name__=='__main__':
    print('Parent process %s.' % os.getpid())
    p = Pool(3)
    while True:
        results = [p.apply_async(f, args = (i, )) for i in range(100)]
        output = [x.get() for x in results]
        print(output)
        print('sleep for 5s')
        time.sleep(5)
"""

import multiprocessing as mp
import time

def foo_pool(x):
    time.sleep(2)
    return x*x

result_list = []
def log_result(result):
    # This is called whenever foo_pool(i) returns a result.
    # result_list is modified only by the main process, not the pool workers.
    result_list.append(result)

def apply_async_with_callback():
    pool = mp.Pool()
    for i in range(10):
        pool.apply_async(foo_pool, args = (i, ), callback = log_result)
    pool.close()
    pool.join()
    print(result_list)

if __name__ == '__main__':
    apply_async_with_callback()