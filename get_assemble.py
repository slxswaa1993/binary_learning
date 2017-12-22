import os
FILE_PATH = '/home/junjunshi/binarylearning/bisheData/binary_result/'

def main():
    dir_list = os.listdir(FILE_PATH)
    for dir in dir_list:
        file_list = os.listdir(FILE_PATH+dir)
        assemble = []
        for file in file_list:
            file_path = FILE_PATH+dir+'/'+file
            print file_path
            f = open(file_path,mode='r')
            content = f.read()
            blocks = content.split('@')
            for i in range(len(blocks)-1):
                ass = blocks[i].split('#')[-1]
                assemble.append(ass)
            f.close()
        new_file = open(FILE_PATH+dir+'/all_assemble.txt',mode='w')
        new_file.write(''.join(assemble))
        new_file.close()
if __name__ == '__main__':
    main()