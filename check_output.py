import os
import collections

output_path = '/home/junjunshi/binarylearning/bisheData/binary_result/'

def main():
    dir_list = os.listdir(output_path)
    for dir in dir_list:
        file_list = os.listdir(os.path.join(output_path,dir))
        for file in file_list:
            file_path = output_path+dir+'/'+file
            f = open(file_path,mode='r')
            content = f.read()
            word_count = collections.Counter(content)
            print word_count

if __name__ == '__main__':
    main()
