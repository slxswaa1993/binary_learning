import os
from subprocess import Popen,PIPE

source_path = "/home/junjunshi/binarylearning/bisheData/binary_source/"
binary_path = "/home/junjunshi/binarylearning/bisheData/binary_out/"
source_nohead_path = '/home/junjunshi/binarylearning/bisheData/binary_source_nohead/'
binary_result_path = '/home/junjunshi/binarylearning/bisheData/binary_result/'
ida_path = "/home/junjunshi/IDA/idaq64"
python_script_path = "/home/junjunshi/PycharmProjects/binarylearning/test.py"


def main():
    dirs = os.listdir(binary_path)
    dirs.sort(key=lambda x: int(x))
    for dir in dirs:
        files = os.listdir(binary_path + dir)
        files.sort(key=lambda x: str(x).split('.')[0])
        if (not os.path.isdir(binary_result_path + dir)):
            os.mkdir(binary_result_path + dir)
        for f in files:
            file_name = f.split(".")[0]
            if f.split('.')[1] == "out":
                source_nohead_file = source_nohead_path+dir+'/'+file_name+'.cpp'

                cmd = ida_path + ' -c -A -S"%s %s %s" %s' % (python_script_path, binary_result_path+dir+'/', source_nohead_file, binary_path+dir+'/'+f)
                p = Popen(cmd, shell=True)
                stdout, stderr = p.communicate()
                os.remove(binary_path+dir+'/'+file_name+'.i64')


    # binary_file = binary_path+'/1/1.out'
    # source_file = source_nohead_path+'1/1.cpp'
    # result_file = binary_result_path+'1/'
    # cmd = ida_path + ' -c -A -S"%s %s %s" %s' % (python_script_path, result_file, source_file, binary_file)
    # print cmd
    # p = Popen(cmd, shell=True, stdout=PIPE)

if __name__=='__main__':
    main()