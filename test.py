from idautils import *
from idaapi import *
from idc import *
#from pycparser import c_ast, parse_file





def query_ida():

    output_path = idc.ARGV[1]
    source_nohead_file = idc.ARGV[2]
    file_name = str(source_nohead_file).split("/")[-1].split(".")[0]
    #func_list = get_func_name(source_nohead_file)

    idaapi.autoWait()

    for seg in Segments():
        if SegName(seg) == ".text":
            functions_list = []
            #output_handler = open('/home/junjunsh=i/main.txt', 'w')
            output_handler = open(output_path +file_name+'.txt', 'w')

            for func_ea in Functions(seg):

                # get function name
                name = GetFunctionName(func_ea)
                if name=='main':
                #if(name in func_list):

                    basic_block_list = []
                    for bb in FlowChart(get_func(func_ea), flags=FC_PREDS):
                        disasm_list = []
                        bytes_list = []
                        mnen_list = []
                        for head in Heads(bb.startEA, bb.endEA):
                            if isCode(getFlags(head)):

                                # get mnemonic
                                mnemonic = GetMnem(head)

                                # get disassembly
                                disasm = GetDisasm(head)

                                # get bytes
                                next = NextHead(head, bb.endEA + 1)
                                if next:
                                    length = next - head
                                else:
                                    length = bb.endEA - head
                                # get bytes
                                if length<1000 and length>=0:
                                    bytes = GetManyBytes(head, length, False)
                                else:
                                    bytes = ""

                                # store the info to the list
                                disasm_list.append(disasm)
                                mnen_list.append(mnemonic)
                                if bytes:
                                    bytes = bytes.encode('hex')
                                    bytes_list.append(bytes)
                                    # store the info to the list

                        # flatten the list to string
                        disasm_str = ' $ '.join(disasm_list)
                        bytes_str = ' $ '.join(bytes_list)
                        basic_block_list.append(disasm_str+' @ '+bytes_str)

                    function_content = ' # '.join(basic_block_list)
                    functions_list.append(function_content)
            output_handler.write('\n\n'.join(functions_list))

    # exit ida pro
    Exit(1)
query_ida()