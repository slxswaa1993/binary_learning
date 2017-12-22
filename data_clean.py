import os
import re
import collections
INPUT_PATH = '/home/zju/slx/binarylearning/bisheData/binary_result/'
OUT_PATH ='/home/zju/slx/binarylearning/bisheData/assemble_code/'

#get assemble code (remove binary code)
def get_assemble(file_path,output_path):
    f = open(file_path,mode='r')
    content = f.read()
    blocks = content.split('@')
    assemble=[]
    for i in range(len(blocks)-1):
        ass = blocks[i].split('#')[-1]
        assemble.append(ass)
    assemble = "$".join(assemble)
    assemble = assemble.split("$")
    clean_ass = [re.split(r'([\s\[\]\+\*_$-,<>\(\):;])',line) for line in assemble]
    clean_ass = [filter(remove_noise,line) for line in clean_ass]
    clean_ass = filter(remove_empty_list,clean_ass)
    new_content = ["$".join(line) for line in clean_ass]
    new_content ="!".join(new_content)
    f.close()
    out_file = open(output_path,mode='w+')
    out_file.write(new_content)
    out_file.close()
    print 'complete process ',output_path

#remove noise,save pure assmeble code
def remove_noise(s):
    is_digit = any(char.isdigit() for char in s)
    is_empty = not (s and s.strip())
    contain_underline = '_' in s
    contain_perc = '%' in s
    contain_comma = "," in s
    has_upper = any(char.isupper() for char in s)
    contain_ques = '?' in s
    contain_ref = '"' in s
    contain_sep = ';' in s
    return not (is_digit or is_empty or contain_perc or contain_ques or contain_underline or has_upper or contain_ref or contain_sep or contain_comma)

def remove_empty_list(l):
    is_empty_list = not ''.join(l).strip()
    zero_lenth = len(l)==0
    return not (is_empty_list or zero_lenth)

def main():
    dir_list = os.listdir(INPUT_PATH)
    for dir in dir_list:
        file_name = INPUT_PATH+dir+'/all_assemble.txt'
        # try:
        #     os.remove(file_name)
        #     print file_name
        # except:
        #     print 'error'
        file_list = os.listdir(INPUT_PATH+dir)
        output_path = OUT_PATH+dir
        if not os.path.exists((output_path)):
            os.makedirs(output_path)
        for file_name in file_list:
            file_path = INPUT_PATH+dir+'/'+file_name
            outfile_path = output_path+'/'+file_name
            get_assemble(file_path,outfile_path)


if __name__ == '__main__':
    main()
