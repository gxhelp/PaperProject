#!/user/bin/env python
# -*- coding:utf-8 -*-
# author :Guoz time:2019/7/19

def frange(x, y, jump):
    while x < y:
        yield x
        x += jump


def data_save_write(data,fig_name_title):
    import time
    cur_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    file_name = 'test_data_saved.txt'
    file_temp = open(file_name, 'a')
    write_content = '{}>{}>>test_data_ber>>>{}\n'.format(cur_time,fig_name_title,data)
    file_temp.write(write_content)
    file_temp.close()
    print("test_data_saved.txt数据成功保存！")
