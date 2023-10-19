"""
将m4a格式转换成wav格式
"""
import os

# path = 'D:/hnu/python/sens/m4aFiles/'
path = 'D:/hnu/python/sens/iPhone/'
filter = [".m4a"]


def all_path(file_path):
    result = []  # 所有文件
    for maindir, subdir, file_name_list in os.walk(file_path):
        for filename in file_name_list:
            apath = os.path.join(maindir, filename)  # 合并成一个完整路径
            ext = os.path.splitext(apath)[1]  # 获取文件后缀 [0]获取的是除了文件名以外的内容

            if ext in filter:
                result.append(apath)

    return result


filenames = all_path(path)

for filename in filenames:
    filename = str(filename)
    temp = filename.split('.')

    # 将.m4a格式转为wav格式的命令
    cmd_command = "ffmpeg -i {0} -acodec pcm_s16le -ac 1 -ar 16000 -y {1}.wav && del {0}".format(filename, temp[0])
    # 将.mp3格式转为wav格式的命令
    # cmd_command = "ffmpeg -loglevel quiet -y -i {0} -ar 16000 -ac 1 {1}.wav && del {0}".format(filename, temp[0])

    # print(cmd_command)
    os.system(cmd_command)
    os.remove(filename)  # 删除源文件


